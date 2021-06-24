import argparse
import glob

import tensorflow as tf
from torch.utils.data import DataLoader

from preprocess import convert_tfrecord
from dataset import FlowerDataset, train_transforms, val_transforms
from models import PlainNet34, ResNet34, ResNet50, PretrainedResNet50
from train import fit


if __name__ == "__main__":
    tf.enable_eager_execution()
    net_type = 'plain34'

    # load dataset
    print('\nLoad dataset')
    train_files = glob.glob('data/tfrecords-*/train/*.tfrec')
    val_files = glob.glob('data/tfrecords-*/val/*.tfrec')

    train_ids, train_class, train_images = convert_tfrecord(train_files, mode='train')
    val_ids, val_class, val_images = convert_tfrecord(val_files, mode='val')

    train_ids, train_class, train_images = train_ids[:1000], train_class[:1000], train_images[:1000]
    val_ids, val_class, val_images = val_ids[:1000], val_class[:1000], val_images[:1000]

    train_transform, val_transform = train_transforms(net_type), val_transforms(net_type)

    train_ds = FlowerDataset(train_ids, train_class, train_images, train_transform, mode='train')
    val_ds = FlowerDataset(val_ids, val_class, val_images, val_transform, mode='val')

    ds_size = dict(
        train=len(train_ds),
        val=len(val_ds)
    )

    loaders = dict(
        train=DataLoader(train_ds, 256, num_workers=4, pin_memory=True),
        val=DataLoader(val_ds, 256, num_workers=4, pin_memory=True)
    )

    # load net
    print('\nLoad {}'.format(net_type))
    net = PlainNet34()

    # train
    epochs = 10
    fit(net_type, loaders, ds_size, net, epochs, device='cuda:0')
