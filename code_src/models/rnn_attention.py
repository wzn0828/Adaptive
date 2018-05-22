import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
from code_src.models import base_adaptive

# ========================================Recurrent Attention========================================
# Encoder, doing this for extracting cnn features. Temporarily, this is the same as base_adaptive.py.

# Attention Block for C_hat and attention value calculation
class Atten(nn.Module):
    def __init__(self, hidden_size, cf):
        super(Atten, self).__init__()

        self.affine_v = nn.Linear(hidden_size, 49, bias=False)  # W_v
        self.affine_g = nn.Linear(hidden_size, 49, bias=False)  # W_g
        self.affine_h = nn.Linear(49, 1, bias=False)  # w_h

        self.dropout = nn.Dropout(0)
        self.init_weights()

        self.rnn_attention_hiddensize = cf.rnn_attention_hiddensize//2 if cf.rnn_attention_bidirectional==True else cf.rnn_attention_hiddensize
        self.lstm = nn.LSTM(hidden_size, self.rnn_attention_hiddensize, cf.rnn_attention_numlayers, batch_first=True, bidirectional=cf.rnn_attention_bidirectional)

    def init_weights(self):
        """Initialize the weights."""
        init.xavier_uniform(self.affine_v.weight)
        init.xavier_uniform(self.affine_g.weight)
        init.xavier_uniform(self.affine_h.weight)


    def forward(self, V, h_t):
        '''
        :param V: V=[v_1, v_2, ... v_k], size of [cf.train_batch_size, 49, cf.lstm_hidden_size]
        :param h_t: h_t from LSTM, size of h_t is [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]
        :return: F_T, size of [cf.train_batch_size, maxlength(captions), cf.rnn_attention_hiddensize]; Attention, size of [cf.train_batch_size, maxlength(captions),49]
        '''

        # W_v * V + W_g * h_t * 1^T
        content_v = self.affine_v(self.dropout(V)).unsqueeze(1) \
                    + self.affine_g(self.dropout(h_t)).unsqueeze(2)     # size of [cf.train_batch_size, maxlength(captions), 49, 49]

        # z_t = W_h * tanh( content_v )
        z_t = self.affine_h(self.dropout(F.tanh(content_v))).squeeze(3)      # size of [cf.train_batch_size, maxlength(captions), 49]

        # alpha_t = F.softmax(z_t.view(-1, z_t.size(2))).view(z_t.size(0), z_t.size(1), -1)     # size of [cf.train_batch_size, maxlength(captions),49]
        # use sigmoid instead of softmax.
        alpha_t = F.sigmoid(z_t)    # size of [cf.train_batch_size, maxlength(captions),49]

        # calculated V_weighted
        V = V.unsqueeze(1)      # size of [cf.train_batch_size, 1, 49, cf.lstm_hidden_size]
        alpha_t = alpha_t.unsqueeze(3)  # size of [cf.train_batch_size, maxlength(captions), 49, 1]
        V_weighted = alpha_t*V       # size of [cf.train_batch_size, maxlength(captions), 49, cf.lstm_hidden_size]

        # lstm initialize, temporarily zero
        # use lstm to integrate weighted feature V_weighted
        # self.lstm.flatten_parameters()
        output_h_t, h_c = self.lstm(V_weighted.view(-1, V_weighted.size(2), V_weighted.size(3)), None)    # size of output_h_t is [-1,V_weighted.size(2),cf.rnn_attention_hiddensize]
        # extract the last hidden output
        h_T = h_c[0]    # size of h_T is [num_layers * num_directions, -1, self.rnn_attention_hiddensize]
        h_T = torch.cat((h_T[-1, :, :], h_T[-2, :, :]), 1)  # size of h_T is [-1, 2*self.rnn_attention_hiddensize]

        if self.lstm.bidirectional:
            F_T = h_T.view(V_weighted.size(0), V_weighted.size(1), -1)  # size of [cf.train_batch_size, maxlength(captions), cf.rnn_attention_hiddensize]
        else:
            F_T = output_h_t[:, -1, :].view(V_weighted.size(0), V_weighted.size(1), output_h_t.size(2))       # size of [cf.train_batch_size, maxlength(captions), cf.rnn_attention_hiddensize]

        return F_T, alpha_t.squeeze(3)


# Adaptive Attention Block: C_t, Spatial Attention Weights, Sentinel embedding
class AdaptiveBlock(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, cf):
        super(AdaptiveBlock, self).__init__()

        # # Sentinel block
        # self.sentinel = Sentinel(embed_size * 2, hidden_size)

        # Image Spatial Attention Block
        self.atten = Atten(hidden_size, cf)

        # Final Caption generator
        self.mlp = nn.Linear(hidden_size, vocab_size)

        # Dropout layer inside Affine Transformation
        self.dropout = nn.Dropout(0)

        self.hidden_size = hidden_size
        self.init_weights()

    def init_weights(self):
        '''
        Initialize final classifier weights
        '''
        init.kaiming_normal(self.mlp.weight, mode='fan_in')
        self.mlp.bias.data.fill_(0)

    def forward(self, x, hiddens, cells, V):

        F_T, atten_weights = self.atten(V, hiddens)   # size of F_T is [cf.train_batch_size, maxlength(captions), cf.rnn_attention_hiddensize]
                                                      # size of atten_weights is [cf.train_batch_size, maxlength(captions),49]

        # Final score along vocabulary
        scores = self.mlp(self.dropout(F_T + hiddens))    # size of scores is [cf.train_batch_size, maxlength(captions), 10141(vocab_size)]

        return scores, atten_weights

    def init_hidden(self, bsz):
        '''
        Hidden_0 & Cell_0 initialization
        '''
        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            return (Variable(weight.new(1, bsz, self.hidden_size).zero_().cuda()),
                    Variable(weight.new(1, bsz, self.hidden_size).zero_().cuda()))
        else:
            return (Variable(weight.new(1, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(1, bsz, self.hidden_size).zero_()))



# Caption Decoder
class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, cf):
        super(Decoder, self).__init__()

        # word embedding
        self.embed = nn.Embedding(vocab_size, embed_size)

        # LSTM decoder: input = [ w_t; v_g ] => 2 x word_embed_size;
        self.LSTM = nn.LSTM(embed_size * 2, hidden_size, 1, batch_first=True)

        # Save hidden_size for hidden and cell variable
        self.hidden_size = hidden_size

        # Adaptive Attention Block: Sentinel + C_hat + Final scores for caption sampling
        self.adaptive = AdaptiveBlock(embed_size, hidden_size, vocab_size, cf)

    def forward(self, V, v_g, captions, states=None):

        # Word Embedding
        embeddings = self.embed(captions)   # size of [cf.train_batch_size, maxlength(captions), cf.lstm_embed_size]

        # x_t = [w_t;v_g]
        x = torch.cat((embeddings, v_g.unsqueeze(1).expand_as(embeddings)), dim=2)  # size of [cf.train_batch_size, maxlength(captions), 2*cf.lstm_embed_size]

        # Hiddens: Batch x seq_len x hidden_size
        # Cells: seq_len x Batch x hidden_size, default setup by Pytorch
        if torch.cuda.is_available():
            hiddens = Variable(torch.zeros(x.size(0), x.size(1), self.hidden_size).cuda())      # size of [cf.train_batch_size, maxlength(captions),  cf.lstm_hidden_size]
            cells = Variable(torch.zeros(x.size(1), x.size(0), self.hidden_size).cuda())        # size of [maxlength(captions),  cf.train_batch_size, cf.lstm_hidden_size]
        else:
            hiddens = Variable(torch.zeros(x.size(0), x.size(1), self.hidden_size))
            cells = Variable(torch.zeros(x.size(1), x.size(0), self.hidden_size))

        # Recurrent Block
        # Retrieve hidden & cell for Sentinel simulation
        for time_step in range(x.size(1)):
            # Feed in x_t one at a time
            x_t = x[:, time_step, :]    # size of [cf.train_batch_size, 2*cf.lstm_embed_size]
            x_t = x_t.unsqueeze(1)      # size of [cf.train_batch_size, 1, 2*cf.lstm_embed_size]

            h_t, states = self.LSTM(x_t, states)    # size of ht is [cf.train_batch_size, 1, cf.lstm_hidden_size]
                                                    # states[0] is h_n with the size of [1, cf.train_batch_size, cf.lstm_hidden_size]
                                                    # states[1] is c_n with the size of [1, cf.train_batch_size, cf.lstm_hidden_size]

            # Save hidden and cell
            hiddens[:, time_step, :] = h_t.squeeze(1)  # Batch_first
            cells[time_step, :, :] = states[1]

        # cell: Batch x seq_len x hidden_size
        cells = cells.transpose(0, 1)       # size of [cf.train_batch_size, maxlength(captions),  cf.lstm_hidden_size]

        # Data parallelism for adaptive attention block
        if torch.cuda.device_count() > 1:
            device_ids = range(torch.cuda.device_count())
            adaptive_block_parallel = nn.DataParallel(self.adaptive, device_ids=device_ids)
            scores, atten_weights = adaptive_block_parallel(x, hiddens, cells, V)
        else:
            scores, atten_weights = self.adaptive(x, hiddens, cells, V)

        # Return states for Caption Sampling purpose
        return scores, states, atten_weights


# Whole Architecture with Image Encoder and Caption decoder
class Encoder2Decoder(base_adaptive.Encoder2Decoder):
    def __init__(self, cf):    # size of vocab_size is 10141
        nn.Module.__init__(self)

        # Image CNN encoder and Adaptive Attention Decoder
        self.encoder = base_adaptive.AttentiveCNN(cf.rnn_attention_embed_size, cf.rnn_attention_hiddensize)
        self.decoder = Decoder(cf.rnn_attention_embed_size, cf.vocab_length, cf.rnn_attention_hiddensize, cf)

    # Caption generator
    def sampler(self, images, max_len=30):
        '''
        Samples captions for given image features (Greedy search).
        :param images: size of [cf.eval_batch_size, 3, 224, 224]
        :param max_len: the max length of output caption
        :return:
        '''

        # Data parallelism if multiple GPUs
        if torch.cuda.device_count() > 1:
            device_ids = range(torch.cuda.device_count())
            encoder_parallel = torch.nn.DataParallel(self.encoder, device_ids=device_ids)
            V, v_g = encoder_parallel(images)
        else:
            V, v_g = self.encoder(images)       # size of V is [cf.eval_batch_size, 49, cf.lstm_hidden_size]
                                                # size of v_g is [cf.eval_batch_size, cf.lstm_embed_size]

        # Build the starting token Variable <start> (index 1): B x 1
        if torch.cuda.is_available():
            captions = Variable(torch.LongTensor(images.size(0), 1).fill_(1).cuda())
        else:
            captions = Variable(torch.LongTensor(images.size(0), 1).fill_(1))   # size of captions is temporally [cf.eval_batch_size, 1]

        # Get generated caption idx list, attention weights and sentinel score
        sampled_ids = []
        attention = []

        # Initial hidden states
        states = None

        for i in range(max_len):
            scores, states, atten_weights = self.decoder(V, v_g, captions, states)    # size of scores is [cf.eval_batch_size, 1(maxlength(captions)), 10141(vocab_size)]
                                                                                      # size of atten_weights [cf.eval_batch_size, 1, 49]
                                                                                      # size of beta [cf.eval_batch_size, 1, 1]
            predicted = scores.max(2)[1]
            captions = predicted    # size of captions is [cf.eval_batch_size, 1], captions is the index of current output word

            # Save sampled word, attention map and sentinel at each timestep
            sampled_ids.append(captions)
            attention.append(atten_weights)


        # caption: cf.eval_batch_size x max_len
        # attention: cf.eval_batch_size x max_len x 49
        sampled_ids = torch.cat(sampled_ids, dim=1)
        attention = torch.cat(attention, dim=1)

        return sampled_ids, attention