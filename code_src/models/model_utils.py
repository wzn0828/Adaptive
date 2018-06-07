from torch.nn import init

#-----model initialization-----#
def xavier_normal(nonlinearity, *modules):
    '''
    xavier normalization, and fill the bias to zero
    :param nonlinearity: string,the non-linear function (nn.functional name), one of ['linear', 'conv1d', 'conv2d',
    'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d', 'sigmoid', 'tanh', 'relu', 'leaky_relu']
    :param modules: modules which need to be initialized
    :return: no return
    '''
    gain = init.calculate_gain(nonlinearity)
    for module in modules:
        init.xavier_normal_(module.weight, gain)
        if module.bias is not None:
            module.bias.data.fill_(0)


def lstm_init(lstm_Module):
    '''
    orthogonalize the weights in lstm, and zeros the bias in lstm, and the bias of forget gate is set to 1.
    :param lstm_Module: the lstm model to be initialized
    :return: no return
    '''
    hidden_size = lstm_Module.hidden_size
    for name, param in lstm_Module.named_parameters():
        if 'bias' in name:
            init.constant_(param, 0.0)
            param.data[hidden_size:2 * hidden_size] = 0.5
        elif 'weight' in name:
            init.orthogonal_(param[:hidden_size, :])
            init.orthogonal_(param[hidden_size:2 * hidden_size, :])
            init.orthogonal_(param[2 * hidden_size:3 * hidden_size, :])
            init.orthogonal_(param[3 * hidden_size:, :])
