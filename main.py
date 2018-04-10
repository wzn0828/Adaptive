import argparse
import time
from datetime import datetime
from code_src.tools.utils import HMS, configurationPATH
from code_src.config.configuration import Configuration
from code_src.train import main_train
from code_src.tools.resize import main_resize_images
from code_src.data.build_vocab import main_build_vocab
from code_src.data.KarpathySplit import main_KarpathySplit
from code_src.tools.utils import coco_eval
from code_src.data.build_vocab import Vocabulary
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def process(cf):

    if cf.resizeOrnot:
        print('>---------resize images---------<')
        main_resize_images(cf)

    if cf.vacab_build_Ornot:
        print('>---------vacal build---------<')
        main_build_vocab(cf)

    if cf.KarpathySplitOrnot:
        print('>---------KarpathySplit---------<')
        main_KarpathySplit(cf)

    if cf.trainOrnot:
        print('>---------start train---------<')
        main_train(cf)

    if cf.testOrnot:
        print('>---------start test---------<')
        coco_eval(cf, test_mode=True)

def main():
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str,
                        default='/home/wzn/PycharmProjects/Adaptive/code_src/config/cfg_wzn.py',
                        help='Configuration file')
    arguments = parser.parse_args()
    assert arguments.config_path is not None, 'Please provide a path using -c config/pathname in the command line'
    print('\n > Start Time:')
    print('   ' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    start_time = time.time()

    # Load configuration files
    configuration = Configuration(arguments.config_path)
    cf = configuration.load()
    configurationPATH(cf)

    # Train /test/predict with the network, depending on the configuration
    process(cf)

    # End Time
    end_time = time.time()
    print('\n > End Time:')
    print('   ' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    print('\n   ET: ' + HMS(end_time - start_time))

if __name__ == "__main__":
    main()
