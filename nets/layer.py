# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
from config import cfg

def make_linear_layers(feat_dims, relu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

def make_upsample_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            Interpolate(2, 'bilinear'))
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=3,
                stride=1,
                padding=1
                ))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(ResBlock, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.conv = make_conv_layers([in_feat, out_feat, out_feat], bnrelu_final=False)
        self.bn = nn.BatchNorm2d(out_feat)
        if self.in_feat != self.out_feat:
            self.shortcut_conv = nn.Conv2d(in_feat,out_feat,kernel_size=1,stride=1,padding=0)
            self.shortcut_bn = nn.BatchNorm2d(out_feat)

    def forward(self, input):
        x = self.bn(self.conv(input))
        if self.in_feat != self.out_feat:
            x = F.relu(x + self.shortcut_bn(self.shortcut_conv(input)))
        else:
            x = F.relu(x + input)
        return x

def make_conv3d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv3d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm3d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_deconv3d_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose3d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm3d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, is_activation_last=False):
        super().__init__()
        self.num_layers = num_layers
        self.is_activation_last = is_activation_last
        if not isinstance(hidden_dim, list):
            h = [hidden_dim] * (num_layers - 1)
        else:
            assert isinstance(hidden_dim, list), 'hidden_dim arg should be list or a number'
            assert len(hidden_dim) == num_layers-1, 'len(hidden_dim) != num_layers-1'
            h = hidden_dim

        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        all_dims = [input_dim] + h + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(all_dims[i], all_dims[i+1]) for i in range(num_layers))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))

        if self.is_activation_last:
            x = F.relu(self.layers[-1](x))
        else:
            x = self.layers[-1](x)

        return x