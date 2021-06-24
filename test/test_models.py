from pprint import pprint
from collections import OrderedDict

import torch

from models import PlainNet34, ResNet34, ResNet50, PretrainedResNet50
from misc import count_parameters


if __name__ == "__main__":

    print('\nTest models')
    res = OrderedDict()

    plain34 = PlainNet34()
    resnet34 = ResNet34()
    resnet50 = ResNet50()
    pretrained_resnet50 = PretrainedResNet50()

    n_param_plain34 = count_parameters(plain34)
    n_param_resnet34 = count_parameters(resnet34)
    n_param_resnet50 = count_parameters(resnet50)
    n_param_pretrained_resnet50 = count_parameters(pretrained_resnet50)

    print('\nNum parameters')
    res['34-layer plain nets # parameters'] = n_param_plain34
    res['34-layer residual nets # parameters'] = n_param_resnet34
    res['50-layer residual nets # parameters'] = n_param_resnet50
    res['50-layer pretrained residual nets # parameters'] = n_param_pretrained_resnet50
    pprint(res)


    res = OrderedDict()
    tensor_input = torch.randn(1, 3, 244, 244)
    plain34_out = plain34(tensor_input) 
    resnet34_out = resnet34(tensor_input) 
    resnet50_out = resnet50(tensor_input) 
    pretrained_resnet50_out = pretrained_resnet50(tensor_input) 

    print('\nInput-Output test')
    res['34-layer plain nets'] = dict(
        input_shape=tensor_input.shape,
        output_shape=plain34_out.shape,
    )
    res['34-layer residual nets'] = dict(
        input_shape=tensor_input.shape,
        output_shape=resnet34_out.shape,
    )
    res['50-layer residual nets'] = dict(
        input_shape=tensor_input.shape,
        output_shape=resnet50_out.shape,
    )
    res['Pretrained 50-layer residual nets'] = dict(
        input_shape=tensor_input.shape,
        output_shape=pretrained_resnet50_out.shape,
    )
    
    pprint(res)
