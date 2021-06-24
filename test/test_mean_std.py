"""REFERENCE
BASE:  https://www.kaggle.com/georgiisirotenko/pytorch-tpu-baseline-flowers-tranlearning-ensemble#2.-Data-preparation

modified sample transform function, from CentorCrop to Resize
"""

import glob
from pprint import pprint
from collections import OrderedDict
from operator import itemgetter

import tensorflow as tf
import torch
from torch.utils.data import DataLoader

from preprocess import convert_tfrecord
from dataset import FlowerDataset
from misc import sample_transforms


if __name__ == "__main__":
    tf.enable_eager_execution()

    train_files = glob.glob('data/tfrecords-*/train/*.tfrec')

    id_list, class_list, image_list = convert_tfrecord(train_files, mode='train')
    transforms = sample_transforms()
    
    ds = FlowerDataset(id_list, class_list, image_list, transforms, mode='train')
    ds_size = len(ds)
    loader = DataLoader(ds, 128, num_workers=4, pin_memory=True)

    channels = 3

    for channel in range(channels):
        #number of pixels in the dataset = number of all pixels in one object * number of all objects in the dataset
        num_pxl = ds_size*224*224

        #we go through the butches and sum up the pixels of the objects,
        #which then divide the sum by the number of all pixels to calculate the average
        total_sum = 0
        for batch in loader:
            layer = list(map(itemgetter(channel), batch[0]))
            layer = torch.stack(layer, dim=0)
            total_sum += layer.sum()
        mean = total_sum / num_pxl

        #we calculate the standard deviation using the formula that I indicated above
        sum_sqrt = 0
        for batch in loader:
            layer = list(map(itemgetter(channel), batch[0]))
            sum_sqrt += ((torch.stack(layer, dim=0) - mean).pow(2)).sum()
        std = torch.sqrt(sum_sqrt / num_pxl)

        print(f'\n|channel:{channel+1}| - mean: {mean}, std: {std}')

"""Result
|channel:1| - mean: 0.45327600836753845, std: 0.27964890003204346

|channel:2| - mean: 0.4157607853412628, std: 0.2421468198299408

|channel:3| - mean: 0.3070197105407715, std: 0.2701781690120697
"""
