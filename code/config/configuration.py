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

        # # experiment description

        # if cf.A_star_search_3D_multiprocessing_multicost or cf.A_star_search_3D_multiprocessing_rainfall_wind:
        #     if cf.risky:
        #         cf.model_description += '_risky'
        #     elif cf.wind_exp:
        #         cf.model_description += '_wind_exp_mean_' + str(cf.wind_exp_mean) + '_std_' + str(cf.wind_exp_std)


        # experiment directory
        if True:  # This is for submitting test file
            cf.exp_dir = os.path.join(cf.path_experiment, 'Test_' + cf.model_description + '_' * 3 + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        else:
            cf.exp_dir = os.path.join(cf.path_experiment, 'Train_' + cf.model_description + '_' * 3 + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        if True:
            # Enable log file
            os.mkdir(cf.exp_dir)
            cf.log_file = os.path.join(cf.exp_dir, "logfile.log")
            sys.stdout = Logger(cf.log_file)
            # we print the configuration file here so that the configuration is traceable
            print(help(cf))

        return cf


