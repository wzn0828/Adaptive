from code_src import models
import torch


def get_model(cf):
    # build model
    if cf.atten_model_name == 'adaptive':
        model = models.base_adaptive.Encoder2Decoder(cf.lstm_embed_size, cf.vocab_length, cf.lstm_hidden_size)
    elif cf.atten_model_name == 'rnn_attention':
        model = models.rnn_attention.Encoder2Decoder(cf)

    # load pretrained model or not, and get start_epoch
    if cf.train_pretrained:
        model.load_state_dict(torch.load(cf.train_pretrained_model))
        # Get starting epoch #, note that model is named as '...your path to model/algoname-epoch#.pkl'
        # A little messy here.
        start_epoch = int(cf.train_pretrained_model.split('/')[-1].split('-')[1].split('.')[0]) + 1
    else:
        start_epoch = 1

    return model, start_epoch


def get_encoder_parameters(cf, model):
    """
    Constructing CNN parameters for optimization, only fine-tuning higher layers
    :param model: the encoder2decoder model
    :param cf: config file
    :return: parameters of cnn needed to be optimized
    """

    cnn_subs = list(model.encoder.resnet_conv.children())[cf.opt_fine_tune_cnn_start_layer:]
    cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
    cnn_params = [item for sublist in cnn_params for item in sublist]

    return cnn_params


def get_encoder_optimizer_param(cf, params):
    """
    Constructing CNN finutuning optimizer
    :param params: parameters of cnn needed to be optimized
    :param cf: config file
    :return: optimizer for cnn finetuning
    """

    if cf.opt_cnn_optimization == 'adam':
        encoder_optimizer = torch.optim.Adam(params, lr=cf.opt_cnn_adam_learning_rate, betas=(cf.opt_cnn_adam_alpha, cf.opt_cnn_adam_beta))
    elif cf.opt_cnn_optimization == 'sgd':
        encoder_optimizer = torch.optim.SGD(params, lr=cf.opt_cnn_sgd_learning_rate, momentum=cf.opt_cnn_sgd_momentum, nesterov=True)
    elif cf.opt_cnn_optimization == 'lbfgs':
        encoder_optimizer = torch.optim.LBFGS(params, lr=cf.opt_cnn_lbfgs_lr, max_iter=cf.opt_cnn_lbfgs_max_iter,
                                      history_size=cf.opt_cnn_lbfgs_history)

    return encoder_optimizer


def get_decoder_parameters(model):
    # Other parameter optimization
    decoder_params = list(model.encoder.affine_a.parameters()) + list(model.encoder.affine_b.parameters()) \
             + list(model.decoder.parameters())

    return decoder_params


def get_decoder_optimizer_param(cf, params):
    if cf.opt_rnn_optimization == 'adam':
        decoder_optimizer = torch.optim.Adam(params, lr=cf.opt_rnn_adam_learning_rate, betas=(cf.opt_rnn_adam_alpha, cf.opt_rnn_adam_beta), weight_decay=cf.opt_rnn_adam_weight_decay)
    elif cf.opt_rnn_optimization == 'sgd':
        decoder_optimizer = torch.optim.SGD(params, lr=cf.opt_rnn_sgd_learning_rate, momentum=cf.opt_rnn_sgd_momentum, nesterov=True, weight_decay=cf.opt_rnn_sgd_weight_decay)
    elif cf.opt_rnn_optimization == 'lbfgs':
        decoder_optimizer = torch.optim.LBFGS(params, lr=cf.opt_rnn_lbfgs_lr, max_iter=cf.opt_rnn_lbfgs_max_iter, history_size=cf.opt_rnn_lbfgs_history)

    return decoder_optimizer


def get_encoder_optimizer(cf, model):
    cnn_params = get_encoder_parameters(cf, model)
    encoder_optimizer = get_encoder_optimizer_param(cf, cnn_params)

    return encoder_optimizer


def get_decoder_optimizer(cf, model):
    decoder_params = get_decoder_parameters(model)
    decoder_optimizer = get_decoder_optimizer_param(cf, decoder_params)

    return decoder_optimizer