#--------------------path--------------------#
experiment_path             = 'Experiments'
vocab_path                  = 'code_src/data/vocab.pkl'   # path for vocabulary wrapper
resized_image_dir           = '/home/wangzn/Datasets/ImageCaption/MSCOCO/resized'   # directory for resized training images
image_dir                   = '/home/wangzn/Datasets/ImageCaption/MSCOCO'
captions_val_origin         = '/home/wangzn/Datasets/ImageCaption/MSCOCO/annotations/annotations_trainval2014/captions_val2014.json'
captions_train_origin       = '/home/wangzn/Datasets/ImageCaption/MSCOCO/annotations/annotations_trainval2014/captions_train2014.json'
splited_anno_path_prefix    = 'code_src/data/annotations/karpathy_split_'
train_anno_path             = 'code_src/data/annotations/karpathy_split_train_overfit.json'  # path for train annotation json file
val_anno_path               = 'code_src/data/annotations/karpathy_split_val_hyperparameter.json'    # path for validation annotation json file
test_anno_path              = 'code_src/data/annotations/karpathy_split_test.json'   # path for test annotation json file
train_eval_anno_path        = 'code_src/data/annotations/karpathy_split_train_eval_hyperparameter.json'

#--------------------attention_model--------------------#
atten_model_name            = 'rnn_attention'    # ['adaptive','rnn_attention']

#--------------------train--------------------#
trainOrnot                  = True
train_crop_size             = 224        # size for randomly cropping images
train_log_step              = 10                                                 # step size for printing log info
train_random_seed           = 123                                                # random seed for model reproduction
train_pretrained            = False         # use train_pretrained_model or not
train_pretrained_model      = 'Experiments/Train_adaptive_sgd_sgd_cnn_start_layer_5_cnn_start_epoch_20___2018-05-17-16-37-45/trained_models/adaptive-1.pkl'      # [''|'path'] path of used model'] start from checkpoint or scratch, '' represents start from scratch
train_num_epochs            = 200                                        # the maximum epochs
train_batch_size            = 20                                        # on cluster setup, 60 each x 4 for Huckle server
train_clip                  = 0.1                                       # Gradient clipping for gradient exploding problem in LSTM
train_lr_decay              = 40                                        # epoch at which to start lr decay
train_lr_decay_every        = 50                                        # decay learning rate half at every this number
train_early_stop            = True
train_figure_epoch          = 2
train_evalOrnot             = False


# optimization
# CNN fine-tuning
opt_fine_tune_cnn_start_layer   = 5                                         # CNN fine-tuning layers from: [0-7]
opt_fine_tune_cnn_start_epoch   = 20                                        # start fine-tuning CNN after

# Optimizer parameter of rnn
opt_rnn_optimization                = 'lbfgs'  #['adam','sgd','lbfgs']
opt_rnn_adam_alpha                  = 0.9                                       # alpha in Adam
opt_rnn_adam_beta                   = 0.999                                     # beta in Adam
opt_rnn_adam_learning_rate          = 5e-6                                    # learning rate for the whole model
opt_rnn_adam_weight_decay           = 0

opt_rnn_sgd_learning_rate           = 5e-4
opt_rnn_sgd_momentum                = 0.8
opt_rnn_sgd_weight_decay            = 0

opt_rnn_lbfgs_lr                    = 0.8
opt_rnn_lbfgs_max_iter              = 20
opt_rnn_lbfgs_history               = 50


# Optimizer parameter of cnn
opt_cnn_optimization            = 'lbfgs'  #['adam','sgd','lbfgs']
opt_cnn_adam_alpha              = 0.9                                       # alpha in Adam
opt_cnn_adam_beta               = 0.999
opt_cnn_adam_learning_rate      = 5e-5
opt_cnn_adam_weight_decay       = 0

opt_cnn_sgd_learning_rate       = 1e-5
opt_cnn_sgd_momentum            = 0.99
opt_cnn_sgd_weight_decay        = 0

opt_cnn_lbfgs_lr                = 0.1
opt_cnn_lbfgs_max_iter          = 20
opt_cnn_lbfgs_history           = 50


#--------------------test--------------------#
testOrnot                   = False
test_pretrained_model       = 'Experiments/Train_rnn_attention_lr_5e-06_cnnlr_1e-06_cnn_start_layer_0_cnn_start_epoch_20___2018-04-27-11-00-54/trained_models/adaptive-51.pkl'      # used pretrained model parameters in test

#--------------------hyper parameters--------------------#


# learning rate for fine-tuning CNN

# LSTM hyper parameters
lstm_embed_size             = 256                                       # dimension of word embedding vectors, also dimension of v_g
lstm_hidden_size            = 512                                       # dimension of lstm hidden states

# For eval_size > 30, it will cause cuda OOM error on Huckleberry
eval_batch_size             = 20
# on cluster setup, 30 each x 4
dataloader_num_workers      = 8

#--------------------resize--------------------#
resizeOrnot                 = False                                     # resize images from
resized_image_size          = 256                                       # size for image after processing

#--------------------build_vocabury--------------------#
vacab_build_Ornot           = False
vocab_threshold             = 5                                         # minimum word count threshold

#--------------------KarpathySplit--------------------#
KarpathySplitOrnot          = False
num_val                     = 5000
num_test                    = 5000
num_train_eval              = 5000

num_train_overfit           = 20

num_train_hyperparameter    = 5000
num_train_eval_hyperparameter = 1000
num_val_hyperparameter      = 1000


#--------------------rnn_attention--------------------#
rnn_attention_bidirectional = True                                      # whether use bidirection in lstm structure of rnn_attention
rnn_attention_embed_size    = lstm_embed_size
rnn_attention_hiddensize    = lstm_hidden_size
rnn_attention_numlayers     = 1

#--------------------valid-------------------#
validOrnot                   = False
valid_pretrained_model       = 'Experiments/Train_rnn_attention_lr_5e-05_cnnlr_1e-05_cnn_start_layer_5_cnn_start_epoch_20___2018-04-19-10-51-43/trained_models/adaptive-1.pkl'      # used pretrained model parameters in test
