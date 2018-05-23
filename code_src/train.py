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
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from tensorboardX import SummaryWriter
writer = SummaryWriter()
from code_src.models.model_factory import get_model, get_encoder_optimizer, get_decoder_optimizer

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
    model, start_epoch = get_model(cf)

    # Constructing optimizer for encoder and decoder
    encoder_optimizer, encoder_lbfgs_flag = get_encoder_optimizer(cf, model)
    decoder_optimizer, decoder_lbfgs_flag = get_decoder_optimizer(cf, model)

    # Language Modeling Loss
    LMcriterion = nn.CrossEntropyLoss()

    # Change to GPU mode if available
    if torch.cuda.is_available():
        model.cuda()
        LMcriterion.cuda()

    # Train the Models
    total_step = len(data_loader)

    cider_scores = []
    cider_scores_train_eval = []
    best_cider = 0.0
    best_epoch = 0

    train_losses = []
    # Start Training
    for epoch in range(start_epoch, cf.train_num_epochs + 1):

        # # Start Learning Rate Decay
        # learning_rate = lr_decay(cf, epoch, learning_rate)

        # print('Learning Rate for Epoch %d: %.6f' % (epoch, learning_rate))


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
            model.train()
            model.zero_grad()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Gradient clipping for gradient exploding problem in LSTM
            for p in model.decoder.LSTM.parameters():
                p.data.clamp_(-cf.train_clip, cf.train_clip)

            # check overfit tiny dataset
            if cf.train_overfit_check:
                params, model = L_BFGS(params, model, images, captions, lengths, targets, LMcriterion, cf, epoch, i, total_step, train_batch_losses)
                continue


            if decoder_lbfgs_flag==False or (epoch > cf.opt_fine_tune_cnn_start_epoch and encoder_lbfgs_flag==False):
                # Forward
                packed_scores = model(images, captions, lengths)  # size of packed_scores[0] is [sum(lengths), 10141(vocab_size)]
                # Compute loss and backprop
                loss = LMcriterion(packed_scores[0], targets)
                loss_data = loss.data.cpu().item()
                loss.backward()

            # decoder optimize
            if decoder_lbfgs_flag:
                loss_data = L_BFGS_optimize(decoder_optimizer, model, images, captions, lengths, targets, LMcriterion)
            else:
                decoder_optimizer.step()

            # encoder optimize
            if epoch > cf.opt_fine_tune_cnn_start_epoch:
                if encoder_lbfgs_flag:
                    loss_data = L_BFGS_optimize(encoder_optimizer, model, images, captions, lengths, targets,
                                                        LMcriterion)
                else:
                    encoder_optimizer.step()

            train_batch_losses.append(loss_data)

            # Print log info
            if i % cf.train_log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], CrossEntropy Loss: %.4f, Perplexity: %5.4f' % (epoch,
                                                                                                   cf.train_num_epochs,
                                                                                                   i, total_step,
                                                                                                   loss_data,
                                                                                                   np.exp(loss_data)))

        # Save the Adaptive Attention model after each epoch
        torch.save(model.state_dict(),
                   os.path.join(cf.trained_model_path,
                                'attention_model-%d.pkl' % (epoch)))

        train_loss = np.array(train_batch_losses).mean()
        print('Train Loss', epoch, train_loss)
        train_losses.append(train_loss)
        print('Train Losses:')
        print(train_losses)
        # plot figure losses
        figure_loss(cf, epoch, train_losses)

        if cf.train_evalOrnot:
            # Evaluation on train_eval set
            cider_train_eval = coco_eval(cf, model=model, epoch=epoch, train_mode=True)
            cider_scores_train_eval.append(cider_train_eval)
            print('#---printing train_eval cider_scores---#')
            print(cider_scores_train_eval)

            # Evaluation on validation set
            cider = coco_eval(cf, model=model, epoch=epoch)
            cider_scores.append(cider)
            print('#---printing validation cider_scores---#')
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


def L_BFGS(params, adaptive, images, captions, lengths, targets, LMcriterion, cf, epoch, step, total_step, train_batch_losses):
    optimizer = torch.optim.LBFGS(params, lr=0.8)
    def closure():
        # Forward
        optimizer.zero_grad()
        packed_scores = adaptive(images, captions,
                                 lengths)  # size of packed_scores[0] is [sum(lengths), 10141(vocab_size)]
        # Compute loss and backprop
        loss = LMcriterion(packed_scores[0], targets)
        train_batch_losses.append(loss.data.cpu().item())
        loss.backward()
        return loss
    optimizer.step(closure)

    print('Epoch [%d/%d], Step [%d/%d], CrossEntropy Loss: %.4f, Perplexity: %5.4f' % (epoch,
                                                                                       cf.train_num_epochs,
                                                                                       step, total_step,
                                                                                       train_batch_losses[-1],
                                                                                       np.exp(
                                                                                           train_batch_losses[-1])))
    return params, adaptive


def L_BFGS_optimize(optimizer, model, images, captions, lengths, targets, LMcriterion):
    model.zero_grad()
    optimizer.zero_grad()

    batch_losses = []
    def closure():
        # Forward
        optimizer.zero_grad()
        packed_scores = model(images, captions,
                                 lengths)  # size of packed_scores[0] is [sum(lengths), 10141(vocab_size)]
        # Compute loss and backprop
        loss = LMcriterion(packed_scores[0], targets)
        batch_losses.append(loss.data.cpu().item())
        loss.backward()
        return loss
    optimizer.step(closure)

    return batch_losses[0]


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
        plt.close()