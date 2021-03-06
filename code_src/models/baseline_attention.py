import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from code_src.models import model_utils

# ========================================spatial attention========================================#
# Encoder, doing this for extracting cnn features.
class AttentiveCNN(nn.Module):
    def __init__(self, embed_size, hidden_size, cf):
        super(AttentiveCNN, self).__init__()

        # ResNet-152 backend
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential(*modules)  # last conv feature

        self.resnet_conv = resnet_conv
        self.avgpool = nn.AvgPool2d(7)
        self.affine_a = nn.Linear(2048, hidden_size)  # v_i = W_a * A
        self.affine_b = nn.Linear(2048, embed_size)  # v_g = W_b * a^g

        # Dropout before affine transformation
        self.dropout = nn.Dropout(0)

        # initialization
        model_utils.kaiming_uniform('relu', 0, self.affine_a, self.affine_b)

        # To generate h0 & c0 in LSTM-Decoder
        self.affine_h0 = nn.Linear(2048, hidden_size)
        self.affine_c0 = nn.Linear(2048, hidden_size)
        model_utils.xavier_uniform('tanh', self.affine_h0, self.affine_c0)

    def forward(self, images):
        '''
        Input: images
        Output: V=[v_1, ..., v_n], v_g
        '''

        # Last conv layer feature map, size of [cf.train_batch_size, 2048, 7, 7]
        A = self.resnet_conv(images)

        # a^g, average pooling feature map
        a_g = self.avgpool(A)       # size of [cf.train_batch_size, 2048, 1, 1]
        a_g = a_g.view(a_g.size(0), -1)       # size of [cf.train_batch_size, 2048]

        # V = [ v_1, v_2, ..., v_49 ]
        V = A.view(A.size(0), A.size(1), -1).transpose(1, 2)     # size of [cf.train_batch_size, 49, 2048]
        V = F.relu(self.affine_a(self.dropout(V)))      # size of [cf.train_batch_size, 49, cf.lstm_hidden_size]

        v_g = F.relu(self.affine_b(self.dropout(a_g)))      # size of [cf.train_batch_size, cf.lstm_embed_size]

        # states=(h0,c0)
        h0 = F.tanh(self.affine_h0(self.dropout(a_g)))
        h0 = h0.unsqueeze(1)                                # size of [cf.train_batch_size, 1, cf.lstm_hidden_size]
        c0 = F.tanh(self.affine_c0(self.dropout(a_g)))
        c0 = c0.unsqueeze(1)                                # size of [cf.train_batch_size, 1, cf.lstm_hidden_size]
        states = (h0, c0)

        return V, v_g, states


# Attention Block for C_hat calculation
class Atten(nn.Module):
    def __init__(self, hidden_size):
        super(Atten, self).__init__()

        self.affine_v = nn.Linear(hidden_size, 49, bias=False)  # W_v
        self.affine_g = nn.Linear(hidden_size, 49, bias=False)  # W_g
        self.affine_h = nn.Linear(49, 1, bias=False)  # w_h
        self.dropout = nn.Dropout(0)
        # initialization
        model_utils.xavier_normal('tanh', self.affine_v, self.affine_g)
        model_utils.kaiming_normal('relu', 0, self.affine_h)

    def forward(self, V, h_t):
        '''
        Input: V=[v_1, v_2, ... v_k], h_t, s_t from LSTM, size of h_t & s_t is [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]
                size of V is [cf.train_batch_size, 49, cf.lstm_hidden_size]
        Output: c_hat_t, attention feature map
        '''

        # W_v * V + W_g * h_t * 1^T
        content_v = self.affine_v(self.dropout(V)).unsqueeze(1) \
                    + self.affine_g(self.dropout(h_t)).unsqueeze(2)     # size of [cf.train_batch_size, maxlength(captions), 49,49]

        # z_t = W_h * tanh( content_v )
        z_t = self.affine_h(self.dropout(F.tanh(content_v))).squeeze(3)     # size of [cf.train_batch_size, maxlength(captions), 49]
        alpha_t = F.softmax(z_t.view(-1, z_t.size(2)), dim=1).view(z_t.size(0), z_t.size(1), -1)   # size of [cf.train_batch_size, maxlength(captions),49]

        # Construct c_t: B x seq x hidden_size
        c_t = torch.bmm(alpha_t, V).squeeze(2)      # size of [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]

        return c_t, alpha_t


# Adaptive Attention Block: C_t, Spatial Attention Weights, Sentinel embedding
class AdaptiveBlock(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(AdaptiveBlock, self).__init__()

        # Image Spatial Attention Block
        self.atten = Atten(hidden_size)

        # Final Caption generator
        self.mlp = nn.Linear(hidden_size, vocab_size)

        # Dropout layer inside Affine Transformation
        self.dropout = nn.Dropout(0)

        # initialization
        model_utils.kaiming_normal('relu', 0, self.mlp)

    def forward(self, x, hiddens, cells, V):
        # x's size is [cf.train_batch_size, maxlength(captions), 2*cf.lstm_embed_size]
        # hiddens' size is [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]
        # cells' size is [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]

        # Get C_t, Spatial attention, sentinel score
        c_hat, atten_weights = self.atten(V, hiddens)   # size of c_hat is [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]
                                                                        # size of atten_weights [cf.train_batch_size, maxlength(captions), 49]
                                                                        # size of beta [cf.train_batch_size, maxlength(captions), 1]
        # Final score along vocabulary
        scores = self.mlp(self.dropout(c_hat + hiddens))    # size of scores is [cf.train_batch_size, maxlength(captions), 10141(vocab_size)]

        return scores, atten_weights


# Caption Decoder
class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size):
        super(Decoder, self).__init__()

        # word embedding
        self.embed = nn.Embedding(vocab_size, embed_size)

        # LSTM decoder: input = [ w_t; v_g ] => 2 x word_embed_size;
        self.LSTM = nn.LSTM(embed_size * 2, hidden_size, 1, batch_first=True)

        # Adaptive Attention Block: Sentinel + C_hat + Final scores for caption sampling
        self.adaptive = AdaptiveBlock(hidden_size, vocab_size)

        # initialize the lstm
        model_utils.lstm_init(self.LSTM)

    def forward(self, V, v_g, captions, states=None):

        # Word Embedding
        embeddings = self.embed(captions)   # size of [cf.train_batch_size, maxlength(captions), cf.lstm_embed_size]

        # x_t = [w_t;v_g]
        x = torch.cat((embeddings, v_g.unsqueeze(1).expand_as(embeddings)), dim=2)  # size of [cf.train_batch_size, maxlength(captions), 2*cf.lstm_embed_size]

        # Hiddens: Batch x seq_len x hidden_size
        # Cells: seq_len x Batch x hidden_size, default setup by Pytorch
        if torch.cuda.is_available():
            hiddens = Variable(torch.zeros(x.size(0), x.size(1), self.LSTM.hidden_size).cuda())      # size of [cf.train_batch_size, maxlength(captions),  cf.lstm_hidden_size]
            cells = Variable(torch.zeros(x.size(1), x.size(0), self.LSTM.hidden_size).cuda())        # size of [maxlength(captions),  cf.train_batch_size, cf.lstm_hidden_size]
        else:
            hiddens = Variable(torch.zeros(x.size(0), x.size(1), self.LSTM.hidden_size))
            cells = Variable(torch.zeros(x.size(1), x.size(0), self.LSTM.hidden_size))

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
            scores_attens = adaptive_block_parallel(x, hiddens, cells, V)
        else:
            scores_attens = self.adaptive(x, hiddens, cells, V)   # size of scores is [cf.train_batch_size, maxlength(captions), 10141(vocab_size)]
                                                                                # size of atten_weights is [cf.train_batch_size, maxlength(captions), 49]
                                                                                # size of beta is [cf.train_batch_size, maxlength(captions), 1]

        # Return states for Caption Sampling purpose
        return scores_attens + (states,)


# Whole Architecture with Image Encoder and Caption decoder        
class Encoder2Decoder(nn.Module):
    def __init__(self, cf):    # size of vocab_size is 10141
        super(Encoder2Decoder, self).__init__()

        # Image CNN encoder and Adaptive Attention Decoder
        self.encoder = AttentiveCNN(cf.base_word_embed_size, cf.base_lstm_hidden_size, cf)
        self.decoder = Decoder(cf.base_word_embed_size, cf.vocab_length, cf.base_lstm_hidden_size)

    def forward(self, images, captions, lengths):
        '''
        :param images: size of [cf.train_batch_size, 3, 224, 224]
        :param captions: size of [cf.train_batch_size, maxlength of current batch]
        :param lengths: size of cf.train_batch_size, each element has removed the first word (<start> token)
        :return: packed_scores
        '''
        # Data parallelism for V v_g encoder if multiple GPUs are available
        # V=[ v_1, ..., v_k ], v_g in the original paper
        if torch.cuda.device_count() > 1:
            device_ids = range(torch.cuda.device_count())
            encoder_parallel = torch.nn.DataParallel(self.encoder, device_ids=device_ids)
            V, v_g, states = encoder_parallel(images)
        else:
            V, v_g, states = self.encoder(images)   # size of V is [cf.train_batch_size, 49, 512], v_g's is [cf.train_batch_size, 256]

        states[0].transpose_(0, 1)
        states[1].transpose_(0, 1)
        # Language Modeling on word prediction
        decoder_outputs = self.decoder(V, v_g, captions, states)    # size of scores is [cf.train_batch_size, 18, 10141(vocab_size)]

        # Pack it to make criterion calculation more efficient
        packed_scores = pack_padded_sequence(decoder_outputs[0], lengths, batch_first=True) # size of packed_scores.data is [sum(lengths), 10141]

        return packed_scores

    # Caption generator
    def sampler(self, images, max_len=30):
        '''
        Samples captions for given image features (Greedy search).
        :param images: size of [cf.eval_batch_size, 3, 224, 224]
        :param lengths: size of cf.eval_batch_size, each element has removed the first word (<start> token)
        :param max_len: the max length of output caption
        :return:
        '''

        # Data parallelism if multiple GPUs
        if torch.cuda.device_count() > 1:
            device_ids = range(torch.cuda.device_count())
            encoder_parallel = torch.nn.DataParallel(self.encoder, device_ids=device_ids)
            V, v_g, states = encoder_parallel(images)
        else:
            V, v_g, states = self.encoder(images)   # size of V is [cf.eval_batch_size, 49, cf.lstm_hidden_size]
                                            # size of v_g is [cf.eval_batch_size, cf.lstm_embed_size]

        states[0].transpose_(0, 1)
        states[1].transpose_(0, 1)
        # Build the starting token Variable <start> (index 1): B x 1
        if torch.cuda.is_available():
            captions = Variable(torch.LongTensor(images.size(0), 1).fill_(1).cuda())
        else:
            captions = Variable(torch.LongTensor(images.size(0), 1).fill_(1))  # size of captions is temporally [cf.eval_batch_size, 1]

        # Get generated caption idx list, attention weights and sentinel score
        sampled_ids = []
        attention = []

        for i in range(max_len):
            scores, atten_weights, states = self.decoder(V, v_g, captions, states)    # size of scores is [cf.eval_batch_size, 1(maxlength(captions)), 10141(vocab_size)]
                                                                                            # size of atten_weights [cf.eval_batch_size, 1, 49]
                                                                                            # size of beta [cf.eval_batch_size, 1, 1]
            predicted = scores.max(2)[1]
            captions = predicted    # size of captions is [cf.eval_batch_size, 1], captions is the index of current output word

            # Save sampled word, attention map and sentinel at each timestep
            sampled_ids.append(captions)
            attention.append(atten_weights)

        # caption: B x max_len
        # attention: B x max_len x 49
        # sentinel: B x max_len
        sampled_ids = torch.cat(sampled_ids, dim=1)
        attention = torch.cat(attention, dim=1)

        return sampled_ids, attention
