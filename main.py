import argparse
import glob

import tensorflow as tf
from torch.utils.data import DataLoader

from preprocess import convert_tfrecord
from dataset import FlowerDataset, train_transforms, val_transforms
from models import PlainNet34, ResNet34, ResNet50, PretrainedResNet50
from train import fit


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument(
        '--net_type', 
        choices=['plain34', 'resnet34', 'resnet50', 'pretrained_resnet50'],
        help='net type')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    tf.enable_eager_execution()

    args = parse_args()

    # load dataset
    print('\nLoad dataset')
    train_files = glob.glob('data/tfrecords-*/train/*.tfrec')
    val_files = glob.glob('data/tfrecords-*/val/*.tfrec')

    train_ids, train_class, train_images = convert_tfrecord(train_files, mode='train')
    val_ids, val_class, val_images = convert_tfrecord(val_files, mode='val')
    train_transform, val_transform = train_transforms(args.net_type), val_transforms(args.net_type)

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
    print('\nLoad {}'.format(args.net_type))
    if args.net_type == 'plain34':
        net = PlainNet34()
    elif args.net_type == 'resnet34':
        net = ResNet34()
    elif args.net_type == 'resnet50':
        net = ResNet50()
    elif args.net_type == 'pretrained_resnet50':
        net = PretrainedResNet50()

    # train
    epochs = 70
    fit(args.net_type, loaders, ds_size, net, epochs, device='cuda:0')
