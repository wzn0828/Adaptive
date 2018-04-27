import argparse
import math
import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
from code_src.tools.utils import coco_eval, to_var
from code_src.data.data_loader import get_loader
import code_src.models as atten_models
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence


def main_train(cf):
    # To reproduce training results
    torch.manual_seed(cf.train_random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cf.train_random_seed)

    # Create model directory
    cf.trained_model_path = os.path.join(cf.exp_dir, 'trained_models')
    if not os.path.exists(cf.trained_model_path):
        os.makedirs(cf.trained_model_path)

    # Image Preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
        transforms.RandomCrop(cf.train_crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper.
    with open(cf.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    cf.vocab_length = len(vocab)

    # Build training data loader
    data_loader = get_loader(cf.resized_image_dir, cf.train_anno_path, vocab,
                             transform, cf.train_batch_size,
                             shuffle=True, num_workers=cf.dataloader_num_workers)

    # build model
    adaptive, start_epoch, params = get_model(cf)

    # Constructing CNN parameters for optimization, only fine-tuning higher layers
    cnn_optimizer = get_cnn_optimizer(adaptive, cf)

    # # Will decay later
    # learning_rate = cf.adam_learning_rate

    # Language Modeling Loss
    LMcriterion = nn.CrossEntropyLoss()


    # Change to GPU mode if available
    if torch.cuda.is_available():
        adaptive.cuda()
        LMcriterion.cuda()

    # Train the Models
    total_step = len(data_loader)

    cider_scores = []
    best_cider = 0.0
    best_epoch = 0

    train_losses = []
    # Start Training
    for epoch in range(start_epoch, cf.train_num_epochs + 1):

        # # Start Learning Rate Decay
        # learning_rate = lr_decay(cf, epoch, learning_rate)

        # print('Learning Rate for Epoch %d: %.6f' % (epoch, learning_rate))

        optimizer = get_optimizer(cf, params)

        # Language Modeling Training
        print('------------------Training for Epoch %d----------------' % (epoch))

        train_batch_losses = []
        for i, (images, captions, lengths, _, _) in enumerate(data_loader):

            # Set mini-batch dataset
            images = to_var(images)     # size of [cf.train_batch_size, 3, 224, 224]
            captions = to_var(captions)     # size of [cf.train_batch_size, maxlength of current batch]
            lengths = [cap_len - 1 for cap_len in lengths]     # size of cf.train_batch_size
            targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]       # size of sum(lengths)

            # preparation for train
            adaptive.train()
            adaptive.zero_grad()

            # Forward
            packed_scores = adaptive(images, captions, lengths)     # size of packed_scores[0] is [sum(lengths), 10141(vocab_size)]

            # Compute loss and backprop
            loss = LMcriterion(packed_scores[0], targets)
            train_batch_losses.append(loss.data.cpu().numpy()[0])
            loss.backward()

            # Gradient clipping for gradient exploding problem in LSTM
            for p in adaptive.decoder.LSTM.parameters():
                p.data.clamp_(-cf.train_clip, cf.train_clip)

            # Optimize
            optimizer.step()
            # Start CNN fine-tuning
            if epoch > cf.opt_fine_tune_cnn_start_epoch:
                cnn_optimizer.step()

            # Print log info
            if i % cf.train_log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], CrossEntropy Loss: %.4f, Perplexity: %5.4f' % (epoch,
                                                                                                   cf.train_num_epochs,
                                                                                                   i, total_step,
                                                                                                   loss.data,
                                                                                                   np.exp(
                                                                                                       loss.data)))

        # Save the Adaptive Attention model after each epoch
        torch.save(adaptive.state_dict(),
                   os.path.join(cf.trained_model_path,
                                'adaptive-%d.pkl' % (epoch)))

        train_loss = np.array(train_batch_losses).mean()
        print('Train Loss', epoch, train_loss)
        train_losses.append(train_loss)

        # Evaluation on validation set
        cider = coco_eval(cf, model=adaptive, epoch=epoch)

        # plot figure losses
        figure_loss(cf, epoch, train_losses)

        cider_scores.append(cider)
        print('#---printing cider_scores---#')
        print(cider_scores)

        # record the best cider and best epoch
        if cider > best_cider:
            best_cider = cider
            best_epoch = epoch

        # judge whether early stop
        whether_early_stop = early_stop_Ornot(cf, cider_scores, best_cider)
        if whether_early_stop:
            break

    print('Model of best epoch #: %d with CIDEr score %.2f' % (best_epoch, best_cider))


def get_model(cf):
    # build model
    if cf.atten_model_name == 'adaptive':
        adaptive = atten_models.adaptive.Encoder2Decoder(cf.lstm_embed_size, cf.vocab_length, cf.lstm_hidden_size)
    elif cf.atten_model_name == 'rnn_attention':
        adaptive = atten_models.rnn_attention.Encoder2Decoder(cf)

    # load pretrained model or not, and get start_epoch
    if cf.train_pretrained:
        adaptive.load_state_dict(torch.load(cf.train_pretrained_model))
        # Get starting epoch #, note that model is named as '...your path to model/algoname-epoch#.pkl'
        # A little messy here.
        start_epoch = int(cf.train_pretrained_model.split('/')[-1].split('-')[1].split('.')[0]) + 1
    else:
        start_epoch = 1

    # Other parameter optimization
    params = list(adaptive.encoder.affine_a.parameters()) + list(adaptive.encoder.affine_b.parameters()) \
             + list(adaptive.decoder.parameters())

    return adaptive, start_epoch, params


def get_optimizer(cf, params):
    if cf.opt_rnn_optimization == 'adam':
        optimizer = torch.optim.Adam(params, lr=cf.opt_rnn_adam_learning_rate, betas=(cf.opt_rnn_adam_alpha, cf.opt_rnn_adam_beta))
    elif cf.opt_rnn_optimization == 'sgd':
        optimizer = torch.optim.SGD(params, lr=cf.opt_rnn_sgd_learning_rate, momentum=cf.opt_rnn_sgd_momentum, nesterov=True)

    return optimizer


def lr_decay(cf, epoch, learning_rate):
    '''
    Learning Rate Decay
    :param cf: config file
    :param epoch: current epoch
    :param learning_rate: current learning_rate
    :return:
    '''

    if epoch > cf.train_lr_decay:
        frac = float(epoch - cf.train_lr_decay) / cf.train_lr_decay_every
        decay_factor = math.pow(0.5, frac)

        # Decay the learning rate
        learning_rate = cf.adam_learning_rate * decay_factor

    return learning_rate


def get_cnn_optimizer(adaptive, cf):
    """
    Constructing CNN parameters for optimization, only fine-tuning higher layers
    :param adaptive: the encoder2decoder model
    :param cf: config file
    :return: parameters of cnn needed to be optimized
    """
    cnn_subs = list(adaptive.encoder.resnet_conv.children())[cf.opt_fine_tune_cnn_start_layer:]
    cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
    cnn_params = [item for sublist in cnn_params for item in sublist]

    if cf.opt_cnn_optimization == 'adam':
        cnn_optimizer = torch.optim.Adam(cnn_params, lr=cf.opt_cnn_adam_learning_rate, betas=(cf.opt_cnn_adam_alpha, cf.opt_cnn_adam_beta))
    elif cf.opt_cnn_optimization == 'sgd':
        cnn_optimizer = torch.optim.SGD(cnn_params, lr=cf.opt_cnn_sgd_learning_rate, momentum=cf.opt_cnn_sgd_momentum, nesterov=True)

    return cnn_optimizer


def early_stop_Ornot(cf, cider_scores, best_cider):
    '''
    judge whether early stop
    :param cf: config file
    :param cider_scores: history of cider_scores
    :param best_cider: the best cider score
    :return: early stop or not
    '''
    flag = False
    if cf.train_early_stop and len(cider_scores) > 5:
        last_6 = cider_scores[-6:]
        last_6_max = max(last_6)

        # Test if there is improvement, if not do early stopping
        if last_6_max != best_cider:
            print('No improvement with CIDEr in the last 6 epochs...Early stopping triggered.')
            flag = True

    return flag


def figure_loss(cf, epoch, train_losses):
    if epoch > 0 and epoch % cf.train_figure_epoch == 0:
        print('---> Train losses:')
        print(train_losses)
        # losses figure
        plt.figure()
        plt.title('Train Losses')
        plt.xlabel('epochs')
        plt.ylabel('losses')
        plt.plot(train_losses, color='b', label='train losses')
        plt.legend()
        figure_name = 'loss_figure_' + str(epoch) + '.jpg'
        figure_path = os.path.join(cf.exp_dir, figure_name)
        plt.savefig(figure_path)