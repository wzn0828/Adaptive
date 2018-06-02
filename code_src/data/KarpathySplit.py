
# coding: utf-8

# # Karpathy Split for MS-COCO Dataset
import json
from random import shuffle, seed, sample

def main_KarpathySplit(cf):
    seed(cf.train_random_seed)  # Make it reproducible

    num_val = cf.num_val
    num_test = cf.num_test

    val = json.load(open(cf.captions_val_origin, 'r'))
    train = json.load(open(cf.captions_train_origin, 'r'))

    # Merge together
    imgs = val['images'] + train['images']
    annots = val['annotations'] + train['annotations']

    shuffle(imgs)

    # Split into val, test, train, based on karpathy split
    dataset = {}
    dataset['val'] = imgs[:num_val]
    dataset['test'] = imgs[num_val: num_val + num_test]
    dataset['train'] = imgs[num_val + num_test:]

    # subset for evaluation of train data
    dataset['train_eval'] = [dataset['train'][i] for i in sorted(sample(range(len(dataset['train'])), cf.num_train_eval))]


    # split subset for hyperparameter, tinyset for overfitting
    dataset['val_hyperparameter'] = [dataset['val'][i] for i in sorted(sample(range(len(dataset['val'])), cf.num_val_hyperparameter))]
    dataset['train_hyperparameter'] = [dataset['train'][i] for i in sorted(sample(range(len(dataset['train'])), cf.num_train_hyperparameter))]
    dataset['train_eval_hyperparameter'] = [dataset['train_hyperparameter'][i] for i in sorted(sample(range(len(dataset['train_hyperparameter'])), cf.num_train_eval_hyperparameter))]

    dataset['train_overfit'] = [dataset['train'][i] for i in sorted(sample(range(len(dataset['train'])), cf.num_train_overfit))]

    # Group by image ids
    itoa = {}
    for a in annots:
        imgid = a['image_id']
        if imgid not in itoa:
            itoa[imgid] = []
        itoa[imgid].append(a)


    json_data = {}
    info = train['info']
    licenses = train['licenses']

    for subset in dataset.keys():

        json_data[subset] = {'type': 'caption', 'info': info, 'licenses': licenses,
                               'images': [], 'annotations': []}

        for img in dataset[subset]:

            img_id = img['id']
            anns = itoa[img_id]

            json_data[subset]['images'].append(img)
            if subset == 'train_overfit':
                json_data[subset]['annotations'].append(anns[0])
            else:
                json_data[subset]['annotations'].extend(anns)

        json.dump(json_data[subset], open(cf.splited_anno_path_prefix + subset + '.json', 'w'))



