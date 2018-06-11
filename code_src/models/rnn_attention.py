import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
from code_src.models import baseline_attention
from code_src.models import model_utils

# ========================================Recurrent Attention========================================
# Encoder, doing this for extracting cnn features. Temporarily, this is the same as adaptive_attention.py.

# Attention Block for C_hat and attention value calculation
class Atten(nn.Module):
    def __init__(self, hidden_size, cf):
        super(Atten, self).__init__()

        self.affine_v = nn.Linear(hidden_size, 49, bias=False)  # W_v
        self.affine_g = nn.Linear(hidden_size, 49, bias=False)  # W_g
        self.affine_h = nn.Linear(49, 1, bias=False)  # w_h

        self.dropout = nn.Dropout(0)

        self.rnn_attention_hiddensize = cf.rnn_attention_hiddensize//2 if cf.rnn_attention_bidirectional==True else cf.rnn_attention_hiddensize
        self.lstm = nn.LSTM(hidden_size, self.rnn_attention_hiddensize, cf.rnn_attention_numlayers, batch_first=True, bidirectional=cf.rnn_attention_bidirectional)

        # initialization
        model_utils.xavier_uniform('tanh', self.affine_v, self.affine_g)
        model_utils.xavier_uniform('sigmoid', self.affine_h)
        model_utils.lstm_init(self.lstm)

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
class AdaptiveBlock(baseline_attention.AdaptiveBlock):
    def __init__(self, hidden_size, vocab_size, cf):
        super(AdaptiveBlock, self).__init__(hidden_size, vocab_size)
        # Image Spatial Attention Block
        self.atten = Atten(hidden_size, cf)


# Caption Decoder
class Decoder(baseline_attention.Decoder):
    def __init__(self, embed_size, vocab_size, hidden_size, cf):
        super(Decoder, self).__init__(embed_size, vocab_size, hidden_size)
        # Adaptive Attention Block: Sentinel + C_hat + Final scores for caption sampling
        self.adaptive = AdaptiveBlock(hidden_size, vocab_size, cf)


# Whole Architecture with Image Encoder and Caption decoder
class Encoder2Decoder(baseline_attention.Encoder2Decoder):
    def __init__(self, cf):    # size of vocab_size is 10141
        nn.Module.__init__(self)

        # Image CNN encoder and Adaptive Attention Decoder
        self.encoder = baseline_attention.AttentiveCNN(cf.rnn_attention_embed_size, cf.rnn_attention_hiddensize)
        self.decoder = Decoder(cf.rnn_attention_embed_size, cf.vocab_length, cf.rnn_attention_hiddensize, cf)
