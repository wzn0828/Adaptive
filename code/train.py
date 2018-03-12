import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import pickle
from code.tools.utils import coco_eval, to_var
from code.data.data_loader import get_loader
from code.models.adaptive import Encoder2Decoder
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence


def main(cf):
    # To reproduce training results
    torch.manual_seed(cf.train_random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cf.train_random_seed)

    # Create model directory
    cf.model_path = os.path.join(cf.exp_dir, 'models')
    if not os.path.exists(cf.model_path):
        os.makedirs(cf.model_path)

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

    # Build training data loader
    data_loader = get_loader(cf.image_dir, cf.train_anno_path, vocab,
                             transform, cf.train_batch_size,
                             shuffle=True, num_workers=cf.num_workers)

    # Load pretrained model or build from scratch
    adaptive = Encoder2Decoder(cf.lstm_embed_size, len(vocab), cf.lstm_hidden_size)

    if cf.train_pretrained_model:
        adaptive.load_state_dict(torch.load(cf.train_pretrained_model))
        # Get starting epoch #, note that model is named as '...your path to model/algoname-epoch#.pkl'
        # A little messy here.
        start_epoch = int(cf.train_pretrained_model.split('/')[-1].split('-')[1].split('.')[0]) + 1

    else:
        start_epoch = 1

    # Constructing CNN parameters for optimization, only fine-tuning higher layers
    cnn_subs = list(adaptive.encoder.resnet_conv.children())[cf.fine_tune_cnn_start_layer:]
    cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
    cnn_params = [item for sublist in cnn_params for item in sublist]

    cnn_optimizer = torch.optim.Adam(cnn_params, lr=cf.adam_learning_rate_cnn,
                                     betas=(cf.adam_alpha, cf.adam_beta))

    # Other parameter optimization
    params = list(adaptive.encoder.affine_a.parameters()) + list(adaptive.encoder.affine_b.parameters()) \
             + list(adaptive.decoder.parameters())

    # Will decay later    
    learning_rate = cf.adam_learning_rate

    # Language Modeling Loss
    LMcriterion = nn.CrossEntropyLoss()

    # adaptive = torch.nn.DataParallel(adaptive, device_ids=[0, 1])   # by wzn, use multi-GPU
    # Change to GPU mode if available
    if torch.cuda.is_available():
        adaptive.cuda()
        LMcriterion.cuda()

    # Train the Models
    total_step = len(data_loader)

    cider_scores = []
    best_cider = 0.0
    best_epoch = 0

    # Start Training 
    for epoch in range(start_epoch, cf.train_num_epochs + 1):

        # Start Learning Rate Decay
        if epoch > cf.train_lr_decay:
            frac = float(epoch - cf.train_lr_decay) / cf.train_lr_decay_every
            decay_factor = math.pow(0.5, frac)

            # Decay the learning rate
            learning_rate = cf.adam_learning_rate * decay_factor

        print('Learning Rate for Epoch %d: %.6f' % (epoch, learning_rate))

        optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(cf.adam_alpha, cf.adam_beta))

        # Language Modeling Training
        print('------------------Training for Epoch %d----------------' % (epoch))
        if epoch != start_epoch:
            for i, (images, captions, lengths, _, _) in enumerate(data_loader):

                # Set mini-batch dataset
                images = to_var(images)
                captions = to_var(captions)
                lengths = [cap_len - 1 for cap_len in lengths]
                targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]

                # Forward, Backward and Optimize
                adaptive.train()
                adaptive.zero_grad()

                # lengths = torch.Tensor(lengths)   # by wzn,new added
                # lengths = to_var(lengths)         # by wzn,new added
                packed_scores = adaptive(images, captions, lengths)

                # Compute loss and backprop
                loss = LMcriterion(packed_scores[0], targets)
                loss.backward()

                # Gradient clipping for gradient exploding problem in LSTM
                for p in adaptive.decoder.LSTM.parameters():
                    p.data.clamp_(-cf.train_clip, cf.train_clip)

                optimizer.step()

                # Start CNN fine-tuning
                if epoch > cf.fine_tune_cnn_start_epoch:
                    cnn_optimizer.step()

                # Print log info
                if i % cf.train_log_step == 0:
                    print('Epoch [%d/%d], Step [%d/%d], CrossEntropy Loss: %.4f, Perplexity: %5.4f' % (epoch,
                                                                                                       cf.train_num_epochs,
                                                                                                       i, total_step,
                                                                                                       loss.data[0],
                                                                                                       np.exp(
                                                                                                           loss.data[0])))

                    # Save the Adaptive Attention model after each epoch
            torch.save(adaptive.state_dict(),
                       os.path.join(cf.model_path,
                                    'adaptive-%d.pkl' % (epoch)))

        # Evaluation on validation set
        cider = coco_eval(adaptive, cf, epoch)
        cider_scores.append(cider)

        if cider > best_cider:
            best_cider = cider
            best_epoch = epoch

        if len(cider_scores) > 5:

            last_6 = cider_scores[-6:]
            last_6_max = max(last_6)

            # Test if there is improvement, if not do early stopping
            if last_6_max != best_cider:
                print('No improvement with CIDEr in the last 6 epochs...Early stopping triggered.')
                print('Model of best epoch #: %d with CIDEr score %.2f' % (best_epoch, best_cider))
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='self', help='To make it runnable in jupyter')
    parser.add_argument('--model_path', type=str, default='./models',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/media/samsumg_1tb/Image_Caption/Datasets/MSCOCO/resized',
                        help='directory for resized training images')
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/karpathy_split_train.json',
                        help='path for train annotation json file')
    parser.add_argument('--caption_val_path', type=str,
                        default='./data/annotations/karpathy_split_val.json',
                        help='path for validation annotation json file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing log info')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for model reproduction')

    # ---------------------------Hyper Parameter Setup------------------------------------

    # CNN fine-tuning
    parser.add_argument('--fine_tune_start_layer', type=int, default=5,
                        help='CNN fine-tuning layers from: [0-7]')
    parser.add_argument('--cnn_epoch', type=int, default=20,
                        help='start fine-tuning CNN after')

    # Optimizer Adam parameter
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='alpha in Adam')
    parser.add_argument('--beta', type=float, default=0.999,
                        help='beta in Adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help='learning rate for the whole model')
    parser.add_argument('--learning_rate_cnn', type=float, default=1e-4,
                        help='learning rate for fine-tuning CNN')

    # LSTM hyper parameters
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors, also dimension of v_g')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')

    # Training details
    parser.add_argument('--pretrained', type=str, default='models/adaptive-1.pkl', help='start from checkpoint or scratch')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=70)  # on cluster setup, 60 each x 4 for Huckle server

    # For eval_size > 30, it will cause cuda OOM error on Huckleberry
    parser.add_argument('--eval_size', type=int, default=28)  # on cluster setup, 30 each x 4
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--lr_decay', type=int, default=20, help='epoch at which to start lr decay')
    parser.add_argument('--learning_rate_decay_every', type=int, default=50,
                        help='decay learning rate at every this number')

    args = parser.parse_args()

    print('------------------------Model and Training Details--------------------------')
    print(args)

    # Start training
    main(args)
