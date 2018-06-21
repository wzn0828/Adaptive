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

    # tensorboard plot
    logdir = os.path.join(cf.exp_dir, 'tensorboard')
    writer = SummaryWriter(logdir+cf.tensorboard)

    # build model
    model, start_epoch = get_model(cf)

    # Constructing optimizer and scheduler for encoder and decoder
    encoder_optimizer, encoder_lbfgs_flag = get_encoder_optimizer(cf, model)
    decoder_optimizer = get_decoder_optimizer(cf, model)
    decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, factor=cf.opt_lrdecay_factor, patience=cf.opt_lrdecay_patience,
                                                                   threshold=0.02, threshold_mode='abs', min_lr=1e-6)
    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, factor=cf.opt_lrdecay_factor, patience=cf.opt_lrdecay_patience,
                                                                   threshold=0.02, threshold_mode='abs', min_lr=1e-7)

    # Language Modeling Loss
    LMcriterion = nn.CrossEntropyLoss()

    # Change to GPU mode if available
    if torch.cuda.is_available():
        model.cuda()
        LMcriterion.cuda()

    # some statics during training
    total_step = len(data_loader)

    cider_scores = []
    cider_scores_train_eval = []
    best_cider = 0.0
    best_epoch = 0

    train_epoch_losses = []
    # initial train_epoch_loss for lr scheduler
    train_epoch_loss = 100

    # Start Training
    global_n_iter = 0
    encoder_opt_flag = False
    for epoch in range(start_epoch, cf.train_num_epochs + 1):
        # Model Training
        print('#------------------Training for Epoch %d----------------#' % (epoch))

        if epoch > cf.opt_fine_tune_cnn_start_epoch:
            encoder_opt_flag = True

        # implementing lr scheduler
        lr_scheduler(decoder_scheduler, encoder_scheduler, encoder_opt_flag, epoch, train_epoch_loss, writer)

        train_batch_losses = []
        for i, (images, captions, lengths, _, _) in enumerate(data_loader):

            # Set mini-batch dataset
            images = to_var(images)     # size of [cf.train_batch_size, 3, 224, 224]
            captions = to_var(captions)     # size of [cf.train_batch_size, maxlength of current batch]
            lengths = [cap_len - 1 for cap_len in lengths]     # size of cf.train_batch_size
            targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]       # size of sum(lengths)

            # preparation for train
            model.train()

            # decoder optimize
            loss_data, total_norm = model_optimize(decoder_optimizer, model, images, captions, lengths, targets, LMcriterion, cf, True)

            # encoder optimize
            if encoder_opt_flag:
                if encoder_lbfgs_flag:
                    model_optimize(encoder_optimizer, model, images, captions, lengths, targets, LMcriterion, cf, False)
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

            # tensorboard: histogram of parameters and gradients, train losses
            if global_n_iter % cf.train_tb_interval_batches == 0:
                for name, param in model.named_parameters():
                    if 'resnet' not in name:
                        writer.add_histogram('Weights_' + name.replace('.', '/'), param, global_n_iter)
                        if cf.train_tb_gradOrnot:
                            writer.add_histogram('Grads_' + name.replace('.', '/'), param.grad, global_n_iter)
                writer.add_scalar('loss-performance/train loss per batches', loss_data, global_n_iter)

                # tensorboard: scalars of lstm norm
                if cf.train_tb_lstm_clip_grad:
                    writer.add_scalar('decoder_norm/decoder_lstm_norm', total_norm, global_n_iter)

            global_n_iter += 1


        train_epoch_loss = np.array(train_batch_losses).mean()
        writer.add_scalar('loss-performance/train loss per epoch', train_epoch_loss, epoch)
        print('Train Loss: epoch', epoch, train_epoch_loss)
        train_epoch_losses.append(train_epoch_loss)
        print('Train epoch losses:')
        print(train_epoch_losses)

        cider = 0
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

            writer.add_scalars('loss-performance/Cider per epoch', {"train": cider_train_eval, "valid": cider}, epoch)

            # record the best cider and best epoch
            if cider > best_cider:
                best_cider = cider
                best_epoch = epoch

            # judge whether early stop
            whether_early_stop = early_stop_Ornot(cf, cider_scores, best_cider)
            if whether_early_stop:
                break

        # Save the Adaptive Attention model after each epoch
        torch.save(model.state_dict(),
                    os.path.join(cf.trained_model_path, 'cider-%.4f_model-%d.pkl' % (cider, epoch)))
    writer.close()

    print('Model of best epoch #: %d with CIDEr score %.2f' % (best_epoch, best_cider))


def lr_scheduler(decoder_scheduler, encoder_scheduler, encoder_opt_flag, epoch, train_epoch_loss, writer):
    decoder_scheduler.step(train_epoch_loss)
    decoder_lr = decoder_scheduler.optimizer.param_groups[0]['lr']
    print('learning rate of Decoder is:', decoder_lr)
    writer.add_scalars('learning_rate_per_epoch', {"decoder": decoder_lr}, epoch)

    if encoder_opt_flag:
        encoder_scheduler.step(train_epoch_loss)
        encoder_lr = encoder_scheduler.optimizer.param_groups[0]['lr']
        print('learning rate of Encoder is:', encoder_lr)
        writer.add_scalars('learning_rate_per_epoch', {"encoder": encoder_lr}, epoch)


def model_optimize(optimizer, model, images, captions, lengths, targets, LMcriterion, cf, lstm_clip_grad):

    batch_losses = []
    total_norm = [0]
    def closure():
        # Forward
        model.zero_grad()
        optimizer.zero_grad()
        packed_scores = model(images, captions,
                                 lengths)  # size of packed_scores[0] is [sum(lengths), 10141(vocab_size)]
        # Compute loss and backprop
        loss = LMcriterion(packed_scores[0], targets)
        batch_losses.append(loss.data.cpu().item())
        loss.backward()

        # clip lstm grad or just for computing the norm of lstm's grad
        if lstm_clip_grad:
            total_norm.append(torch.nn.utils.clip_grad_norm_(model.decoder.LSTM.parameters(), cf.train_lstm_maxnormal))

        return loss
    optimizer.step(closure)

    return batch_losses[0], total_norm[-1]




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
    if cf.train_early_stop and len(cider_scores) > cf.train_early_stop_patience:
        last_ciders = cider_scores[-(cf.train_early_stop_patience+1):]
        last_ciders_max = max(last_ciders)

        # Test if there is improvement, if not do early stopping
        if last_ciders_max != best_cider:
            print('No improvement with CIDEr in the last %d epochs...Early stopping triggered.' % (cf.train_early_stop_patience+1))
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