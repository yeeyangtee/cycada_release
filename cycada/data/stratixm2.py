import os.path

import numpy as np
from PIL import Image

from .cityscapes import remap_labels_to_train_ids
from .gta5 import GTA5 #, LABEL2TRAIN

from .data_loader import register_data_params, register_dataset_obj


@register_dataset_obj('stratixm2')
class StratixM2(GTA5):

    def collect_ids(self):
        ids = GTA5.collect_ids(self)
        existing_ids = []
        for id in ids:
            filename = '{:05d}.png'.format(id)
            if os.path.exists(os.path.join(self.root, 'images', filename)):
                existing_ids.append(id)
        return existing_ids

    def __getitem__(self, index):
        id = self.ids[index]
        filename = '{:05d}.png'.format(id)
        img_path = os.path.join(self.root, 'images', filename)
        label_path = os.path.join(self.root, 'labels', filename)
        img = Image.open(img_path).convert('RGB')
        target = Image.open(label_path)
        img = img.resize(target.size, resample=Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        if self.remap_labels:
            target = np.asarray(target)
            target = remap_labels_to_train_ids(target)
            #target = self.label2train(target)
            target = Image.fromarray(target, 'L')
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


import random
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF



LOAD_SIZE = 286
CROP_SIZE = 256
INSIZE = 256

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

class RotateNinety:
    """Rotate by one of the given angles."""

    def __init__(self):
        pass

    def __call__(self, x):
        return TF.rotate(x, 90)

def get_params():
    new_h = new_w = LOAD_SIZE

    x = random.randint(0, np.maximum(0, new_w - CROP_SIZE))
    y = random.randint(0, np.maximum(0, new_h - CROP_SIZE))
    flip = random.random() > 0.5
    rotate = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip, 'rotate':rotate}

def get_transform(params=None, method=Image.BICUBIC, grayscale=False):
    transform_list = []
    osize = [LOAD_SIZE, LOAD_SIZE]
    transform_list.append(transforms.Resize(osize, method))
    transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], CROP_SIZE)))
    if params['rotate']: transform_list.append(RotateNinety()) # Use params to rotate image 90 deg with 50% chance after cropping.
    if params['flip']: transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    
    transform_list += [transforms.ToTensor()]
    if grayscale: transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else: 
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class ICDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root

        imgdir = root/'img'
        maskdir = root/'mask'
        filenames = [x.stem for x in sorted(list(imgdir.iterdir()))]

        self.images = [imgdir/f'{i}.jpg' for i in filenames]
        self.masks = [maskdir/f'{i}.png' for i in filenames]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
       
        common_params = get_params()
        img_transform = get_transform(common_params)
        mask_transform = get_transform(common_params, grayscale = True)

        return img_transform(img), mask_transform(mask)

