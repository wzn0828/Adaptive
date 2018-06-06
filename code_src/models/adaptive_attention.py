import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
from code_src.models import baseline_attention

# ========================================Knowing When to Look========================================

# Attention Block for C_hat calculation
class Atten(nn.Module):
    def __init__(self, hidden_size):
        super(Atten, self).__init__()

        self.affine_v = nn.Linear(hidden_size, 49, bias=False)  # W_v
        self.affine_g = nn.Linear(hidden_size, 49, bias=False)  # W_g
        self.affine_s = nn.Linear(hidden_size, 49, bias=False)  # W_s
        self.affine_h = nn.Linear(49, 1, bias=False)  # w_h

        self.dropout = nn.Dropout(0)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        init.xavier_uniform(self.affine_v.weight)
        init.xavier_uniform(self.affine_g.weight)
        init.xavier_uniform(self.affine_h.weight)
        init.xavier_uniform(self.affine_s.weight)

    def forward(self, V, h_t, s_t):
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

        # W_s * s_t + W_g * h_t
        content_s = self.affine_s(self.dropout(s_t)) + self.affine_g(self.dropout(h_t))     # size of [cf.train_batch_size, maxlength(captions), 49]
        # w_t * tanh( content_s )
        z_t_extended = self.affine_h(self.dropout(F.tanh(content_s)))   # size of [cf.train_batch_size, maxlength(captions), 1]

        # Attention score between sentinel and image content
        extended = torch.cat((z_t, z_t_extended), dim=2)    # size of [cf.train_batch_size, maxlength(captions), 50]
        alpha_hat_t = F.softmax(extended.view(-1, extended.size(2)),dim=1).view(extended.size(0), extended.size(1), -1)   # size of [cf.train_batch_size, maxlength(captions), 50]
        beta_t = alpha_hat_t[:, :, -1]  # size of [cf.train_batch_size, maxlength(captions)]

        # c_hat_t = beta * s_t + ( 1 - beta ) * c_t
        beta_t = beta_t.unsqueeze(2)    # size of [cf.train_batch_size, maxlength(captions), 1]
        c_hat_t = beta_t * s_t + (1 - beta_t) * c_t     # size of [cf.train_batch_size, maxlength(captions), 512]

        return c_hat_t, alpha_t, beta_t


# Sentinel BLock
class Sentinel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Sentinel, self).__init__()

        self.affine_x = nn.Linear(input_size, hidden_size, bias=False)
        self.affine_h = nn.Linear(hidden_size, hidden_size, bias=False)

        # Dropout applied before affine transformation
        self.dropout = nn.Dropout(0)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform(self.affine_x.weight)
        init.xavier_uniform(self.affine_h.weight)

    def forward(self, x_t, h_t_1, cell_t):
        # size of x_t, h_t_1, cell_t is [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]

        # g_t = sigmoid( W_x * x_t + W_h * h_(t-1) )
        gate_t = self.affine_x(self.dropout(x_t)) + self.affine_h(self.dropout(h_t_1))  # size of [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]
        gate_t = F.sigmoid(gate_t)  # size of [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]

        # Sentinel embedding
        s_t = gate_t * F.tanh(cell_t)   # size of [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]

        return s_t


# Adaptive Attention Block: C_t, Spatial Attention Weights, Sentinel embedding
class AdaptiveBlock(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(AdaptiveBlock, self).__init__()

        # Sentinel block
        self.sentinel = Sentinel(embed_size * 2, hidden_size)

        # Image Spatial Attention Block
        self.atten = Atten(hidden_size)

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
        # x's size is [cf.train_batch_size, maxlength(captions), 2*cf.lstm_embed_size]
        # hiddens' size is [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]
        # cells' size is [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]

        # hidden for sentinel should be h0-ht-1
        h0 = self.init_hidden(x.size(0))[0].transpose(0, 1)     # size of [cf.train_batch_size, 1, cf.lstm_hidden_size]

        # h_(t-1): B x seq x hidden_size ( 0 - t-1 )
        if hiddens.size(1) > 1:
            hiddens_t_1 = torch.cat((h0, hiddens[:, :-1, :]), dim=1)    # size of [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]
        else:
            hiddens_t_1 = h0

        # Get Sentinel embedding, it's calculated blockly    
        sentinel = self.sentinel(x, hiddens_t_1, cells)     # size of [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]

        # Get C_t, Spatial attention, sentinel score
        c_hat, atten_weights, beta = self.atten(V, hiddens, sentinel)   # size of c_hat is [cf.train_batch_size, maxlength(captions), cf.lstm_hidden_size]
                                                                        # size of atten_weights [cf.train_batch_size, maxlength(captions), 49]
                                                                        # size of beta [cf.train_batch_size, maxlength(captions), 1]
        # Final score along vocabulary
        scores = self.mlp(self.dropout(c_hat + hiddens))    # size of scores is [cf.train_batch_size, maxlength(captions), 10141(vocab_size)]

        return scores, atten_weights, beta

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
class Decoder(baseline_attention.Decoder):
    def __init__(self, embed_size, vocab_size, hidden_size):
        nn.Module.__init__(self)

        # word embedding
        self.embed = nn.Embedding(vocab_size, embed_size)

        # LSTM decoder: input = [ w_t; v_g ] => 2 x word_embed_size;
        self.LSTM = nn.LSTM(embed_size * 2, hidden_size, 1, batch_first=True)

        # Save hidden_size for hidden and cell variable 
        self.hidden_size = hidden_size

        # Adaptive Attention Block: Sentinel + C_hat + Final scores for caption sampling
        self.adaptive = AdaptiveBlock(embed_size, hidden_size, vocab_size)


# Whole Architecture with Image Encoder and Caption decoder        
class Encoder2Decoder(baseline_attention.Encoder2Decoder):
    def __init__(self, cf):    # size of vocab_size is 10141
        nn.Module.__init__(self)

        # Image CNN encoder and Adaptive Attention Decoder
        self.encoder = baseline_attention.AttentiveCNN(cf.adaptive_word_embed_size, cf.adaptive_lstm_hidden_size)
        self.decoder = Decoder(cf.adaptive_word_embed_size, cf.vocab_length, cf.adaptive_lstm_hidden_size)

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
            V, v_g = encoder_parallel(images)
        else:
            V, v_g = self.encoder(images)   # size of V is [cf.eval_batch_size, 49, cf.lstm_hidden_size]
                                            # size of v_g is [cf.eval_batch_size, cf.lstm_embed_size]

        # Build the starting token Variable <start> (index 1): B x 1
        if torch.cuda.is_available():
            captions = Variable(torch.LongTensor(images.size(0), 1).fill_(1).cuda())
        else:
            captions = Variable(torch.LongTensor(images.size(0), 1).fill_(1))  # size of captions is temporally [cf.eval_batch_size, 1]

        # Get generated caption idx list, attention weights and sentinel score
        sampled_ids = []
        attention = []
        Beta = []

        # Initial hidden states
        states = None

        for i in range(max_len):
            scores, atten_weights, beta, states = self.decoder(V, v_g, captions, states)    # size of scores is [cf.eval_batch_size, 1(maxlength(captions)), 10141(vocab_size)]
                                                                                            # size of atten_weights [cf.eval_batch_size, 1, 49]
                                                                                            # size of beta [cf.eval_batch_size, 1, 1]
            predicted = scores.max(2)[1]
            captions = predicted    # size of captions is [cf.eval_batch_size, 1], captions is the index of current output word

            # Save sampled word, attention map and sentinel at each timestep
            sampled_ids.append(captions)
            attention.append(atten_weights)
            Beta.append(beta)

        # caption: B x max_len
        # attention: B x max_len x 49
        # sentinel: B x max_len
        sampled_ids = torch.cat(sampled_ids, dim=1)
        attention = torch.cat(attention, dim=1)
        Beta = torch.cat(Beta, dim=1)

        return sampled_ids, attention, Beta
