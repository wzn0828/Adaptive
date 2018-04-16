import imp
import os, sys
from datetime import datetime


# Save the printf to a log file
class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")  # , 0)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Configuration():
    def __init__(self, config_path):

        self.config_path = config_path

    def load(self):
        # Load configuration file...
        print(self.config_path)
        cf = imp.load_source('config', self.config_path)

        # experiment description
        cf.model_description = self.get_model_description(cf)
        cf.exp_dir = os.path.join(cf.experiment_path, cf.model_description + '_' * 3 + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))  # Enable log file

        os.mkdir(cf.exp_dir)
        cf.log_file = os.path.join(cf.exp_dir, "logfile.log")
        sys.stdout = Logger(cf.log_file)
        # we print the configuration file here so that the configuration is traceable
        print(help(cf))

        return cf

    def get_model_description(self, cf):
        model_description = ''
        if cf.resizeOrnot:
            model_description += 'resize_images_size_' + str(cf.resized_image_size)
        if cf.vacab_build_Ornot:
            model_description += 'build_vocabulary_vocab_threshold' + str(cf.vocab_threshold)
        if cf.KarpathySplitOrnot:
            model_description += 'Karpathy_Split'
        if cf.trainOrnot:
            model_description += 'Train_' + cf.atten_model_name + '_lr_' + str(cf.adam_learning_rate) + '_cnnlr_' + str(
                cf.adam_learning_rate_cnn) + '_cnn_start_layer_' + str(
                cf.fine_tune_cnn_start_layer) + '_cnn_start_epoch_' + str(cf.fine_tune_cnn_start_epoch)
        if cf.testOrnot:
            model_description += 'Test_' + cf.test_pretrained_model.replace('/', '_').split('.')[0]
        if cf.validOrnot:
            model_description += 'Valid_' + cf.valid_pretrained_model.replace('/', '_').split('.')[0]

        return model_description


