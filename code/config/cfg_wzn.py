#--------------------path--------------------#
path_experiment             = 'Experiments'
vocab_path                  = 'code/data/vocab.pkl'   # path for vocabulary wrapper
resized_image_dir           = '/media/samsumg_1tb/Image_Caption/Datasets/MSCOCO/resized'   # directory for resized training images
image_dir                   = '/media/samsumg_1tb/Image_Caption/Datasets/MSCOCO'
captions_val_origin         = '/media/samsumg_1tb/Image_Caption/Datasets/MSCOCO/annotations/captions_val2014.json'
captions_train_origin       = '/media/samsumg_1tb/Image_Caption/Datasets/MSCOCO/annotations/captions_train2014.json'
splited_anno_path_prefix    = 'code/data/annotations/karpathy_split_'
train_anno_path             = 'code/data/annotations/karpathy_split_train.json'  # path for train annotation json file
val_anno_path               = 'code/data/annotations/karpathy_split_val.json'    # path for validation annotation json file
test_anno_path              = 'code/data/annotations/karpathy_split_test.json'   # path for test annotation json file

#--------------------train--------------------#
trainOrnot                  = False
train_crop_size             = 224        # size for randomly cropping images
train_log_step              = 10                                                 # step size for printing log info
train_random_seed           = 123                                                # random seed for model reproduction
train_pretrained_model      = 'Experiments/models/adaptive-12.pkl'      # [''|'path'] path of used model'] start from checkpoint or scratch, '' represents start from scratch
train_num_epochs            = 50                                        # the maximum epochs
train_batch_size            = 70                                        # on cluster setup, 60 each x 4 for Huckle server
train_clip                  = 0.1                                       # Gradient clipping for gradient exploding problem in LSTM
train_lr_decay              = 20                                        # epoch at which to start lr decay
train_lr_decay_every        = 50                                        # decay learning rate at every this number

#--------------------hyper parameters--------------------#
# CNN fine-tuning
fine_tune_cnn_start_layer   = 5                                         # CNN fine-tuning layers from: [0-7]
fine_tune_cnn_start_epoch   = 20                                        # start fine-tuning CNN after

# Optimizer Adam parameter
adam_alpha                  = 0.8                                       # alpha in Adam
adam_beta                   = 0.999                                     # beta in Adam
adam_learning_rate          = 4e-4                                      # learning rate for the whole model
adam_learning_rate_cnn      = 1e-4                                      # learning rate for fine-tuning CNN

# LSTM hyper parameters
lstm_embed_size             = 256                                       # dimension of word embedding vectors, also dimension of v_g
lstm_hidden_size            = 512                                       # dimension of lstm hidden states

# For eval_size > 30, it will cause cuda OOM error on Huckleberry
eval_batch_size             = 28                                        # on cluster setup, 30 each x 4
dataloader_num_workers      = 4

#--------------------resize--------------------#
resizeOrnot                 = False                                     # resize images from
resized_image_size          = 256                                       # size for image after processing


#--------------------build_vocabury--------------------#
vacab_build_Ornot           = False
vocab_threshold             = 5                                         # minimum word count threshold


#--------------------KarpathySplit--------------------#
KarpathySplitOrnot          =False
num_val                     = 5000
num_test                    = 5000


