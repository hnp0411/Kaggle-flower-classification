""" ResNet reference 
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

from typing import Type, Any, Callable, Union, List, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models.resnet import conv3x3, _resnet, resnet34, resnet50, ResNet


class PlainBlock(nn.Module):
    """Plain convolution block for 34-layer plain nets

    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(PlainBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('PlainBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in PlainBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)

        return out


def plain34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('plain34', PlainBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def PlainNet34():
    return plain34(pretrained=False, num_classes=104)


def ResNet34():
    return resnet34(pretrained=False, num_classes=104)


def ResNet50():
    return resnet50(pretrained=False, num_classes=104)


def PretrainedResNet50():
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 104)
    return model
