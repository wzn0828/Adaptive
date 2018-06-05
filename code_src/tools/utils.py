import json
import os
import pickle
import string

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms, datasets

from coco.PythonAPI.pycocotools.coco import COCO
from coco.pycocoevalcap.eval import COCOEvalCap
from code_src.models.adaptive_attention import Encoder2Decoder
import code_src.models as atten_models
from code_src.data.data_loader import get_loader
from torch.nn.utils.rnn import pack_padded_sequence


# Variable wrapper
def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile )

# Show multiple images and caption words
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
            
    Adapted from https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    """
    
    assert(( titles is None ) or (len( images ) == len( titles )))
    
    n_images = len( images )
    if titles is None: 
        titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
        
    fig = plt.figure( figsize=( 15, 15 ) )
    for n, (image, title) in enumerate( zip(images, titles) ):
        
        a = fig.add_subplot( np.ceil( n_images/ float( cols ) ), cols, n+1 )
        if image.ndim == 2:
            plt.gray()
            
        plt.imshow( image )
        a.axis('off')
        a.set_title( title, fontsize=200 )
        
    fig.set_size_inches( np.array( fig.get_size_inches() ) * n_images )
    
    plt.tight_layout( pad=0.4, w_pad=0.5, h_pad=1.0 )
    plt.show()

# MS COCO evaluation data loader
class CocoEvalLoader(data.Dataset):

    def __init__(self, root, ann_path, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader):
        '''
        Customized COCO loader to get Image ids and Image Filenames
        root: path for images
        ann_path: path for the annotation file (e.g., caption_val2014.json)
        '''
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.imgs = json.load(open(ann_path, 'r'))['images']

    def __getitem__(self, index):

        filename = self.imgs[index]['file_name']
        img_id = self.imgs[index]['id']

        # Filename for the image
        if 'val' in filename.lower():
            path = os.path.join(self.root, 'val2014', filename)
        else:
            path = os.path.join(self.root, 'train2014', filename)

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, img_id, filename

    def __len__(self):
        return len(self.imgs)


# MSCOCO Evaluation function on validation or test dataset
def coco_eval(cf, model=None, epoch=0, train_mode=False, test_mode=False, valid_mode=False):
    
    '''
    model: trained model to be evaluated
    cf: pre-set parameters
    epoch: epoch #, for disp purpose
    train_mode: flag of evaluate train_subset in training stage
    test_mode: flag of test
    valid_mode: flag of just valid, not validataion in training stage
    '''

    # test_mode and valid_mode can not be true at the same time
    assert not(test_mode and valid_mode and train_mode), "test_mode and valid_mode and train_mode can not be true at the same time"

    # Load the vocabulary
    with open(cf.vocab_path, 'rb') as f:
         vocab = pickle.load(f)

    cf.vocab_length = len(vocab)

    # load parameters dict of model for test or valid
    if test_mode or valid_mode:
        model = get_testOrValid_model(cf, test_mode, valid_mode)

    model.eval()

    # Validation images are required to be resized to 224x224 already
    transform = transforms.Compose([
        transforms.Resize((cf.train_crop_size, cf.train_crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Wrapper the COCO VAL or TEST dataset
    ann_path = cf.val_anno_path
    if test_mode:
        ann_path = cf.test_anno_path
    elif train_mode:
        ann_path = cf.train_eval_anno_path

    data_loader = torch.utils.data.DataLoader(CocoEvalLoader(cf.resized_image_dir, ann_path, transform),
                                              batch_size=cf.eval_batch_size, shuffle=False,
                                              num_workers=cf.dataloader_num_workers, drop_last=False)


    # Generated captions to be compared with GT
    results = []
    print_string = '---------------------Start evaluation on MS-COCO dataset-----------------------'
    if test_mode:
        print_string = '---------------------Start test on MS-COCO dataset-----------------------'
    elif train_mode:
        print_string = '---------------------Start evaluating a subset of training data on MS-COCO dataset-----------------------'

    print(print_string)

    # Language Modeling Loss
    LMcriterion = nn.CrossEntropyLoss()

    valid_batch_losses = []
    with torch.no_grad():
        for i, (images, img_ids, _) in enumerate(data_loader):

            images = to_var(images)
            sampler_output = model.sampler(images)

            generated_caption_idx = sampler_output[0]
            if torch.cuda.is_available():
                captions = generated_caption_idx.cpu().data.numpy()
            else:
                captions = generated_caption_idx.data.numpy()

            # Build caption based on Vocabulary and the '<end>' token
            for image_idx in range(captions.shape[0]):

                sampled_ids = captions[image_idx]
                sampled_caption = []

                for word_id in sampled_ids:
                    word = vocab.idx2word[word_id]
                    if word == '<end>':
                        break
                    else:
                        sampled_caption.append(word)

                sentence = ' '.join(sampled_caption)

                temp = {'image_id': int(img_ids[image_idx]), 'caption': sentence}
                results.append(temp)

            # Disp evaluation process
            if (i+1) % 10 == 0:
                print('[%d/%d]' % ((i + 1), len(data_loader)))

    print('#-----------------------Caption Generated-----------------------#')

    # Evaluate the results based on the COCO API

    # Create result directory
    if test_mode:
        test_pretrained_model_name = cf.test_pretrained_model.replace('/', '_').split('.')[0]
        resFile = os.path.join(cf.exp_dir, test_pretrained_model_name + '.json')
    elif train_mode:
        cf.train_eval_result_path = os.path.join(cf.exp_dir, 'train_eval_results')
        if not os.path.exists(cf.train_eval_result_path):
            os.makedirs(cf.train_eval_result_path)
        filename = 'train_eval-' + str(epoch) + '.json'
        resFile = os.path.join(cf.train_eval_result_path, filename)
    else:
        cf.val_result_path = os.path.join(cf.exp_dir, 'val_results')
        if not os.path.exists(cf.val_result_path):
            os.makedirs(cf.val_result_path)
        filename = 'validation-' + str(epoch) + '.json'
        if valid_mode:
            filename = cf.valid_pretrained_model.replace('/', '_').split('.')[0] + '.json'
        resFile = os.path.join(cf.val_result_path, filename)


    json.dump(results, open(resFile, 'w'))

    coco = COCO(ann_path)
    cocoRes = coco.loadRes(resFile)
    
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds() 
    cocoEval.evaluate()
    
    # Get CIDEr score for validation or test evaluation
    cider = 0.
    print_string = '-----------Evaluation performance on MS-COCO validation dataset for Epoch %d----------' % epoch
    if test_mode:
        print_string = '-----------Evaluation performance on MS-COCO test dataset for pretrained_model: %s----------' % cf.test_pretrained_model
    elif valid_mode:
        print_string = '-----------Evaluation performance on MS-COCO valid dataset for pretrained_model: %s----------' % cf.valid_pretrained_model
    elif train_mode:
        print_string = '-----------Evaluation performance on MS-COCO train_eval dataset for  Epoch %d----------' % epoch
    print(print_string)

    for metric, score in cocoEval.eval.items():
        print('%s: %.4f' % (metric, score))
        if metric == 'CIDEr':
            cider = score
            
    return cider


def get_testOrValid_model(cf, test_mode, valid_mode):
    # build model
    if cf.atten_model_name == 'adaptive':
        adaptive = atten_models.adaptive_attention.Encoder2Decoder(cf.lstm_embed_size, cf.vocab_length, cf.lstm_hidden_size)
    elif cf.atten_model_name == 'rnn_attention':
        adaptive = atten_models.rnn_attention.Encoder2Decoder(cf)

    # load pretrained model
    if test_mode:
        adaptive.load_state_dict(torch.load(cf.test_pretrained_model))
    elif valid_mode:
        adaptive.load_state_dict(torch.load(cf.valid_pretrained_model))
    # Change to GPU mode if available
    if torch.cuda.is_available():
        adaptive.cuda()

    return adaptive


def HMS(sec):
    '''
    :param sec: seconds
    :return: print of H:M:S
    '''

    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)

    return "%dh:%02dm:%02ds" % (h, m, s)


def configurationPATH(cf):
    '''
    :param cf: config file
    :return: Print some paths
    '''
    # Enable log file
    print("\n###########################")
    print(' > Save Path = "%s"' % (cf.exp_dir))
    #print(' > Dataset PATH = "%s"' % (os.path.join(cf.dataroot_dir)))
    print("###########################\n")

