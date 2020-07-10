import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import sys
import os
import numpy as np

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # Modified the way padding was done from the original code base
        self.pad1 = nn.ConstantPad1d((0, padding), 0)

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size=kernel_size,
                               stride=stride, padding=0, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU6()

        self.pad2 = nn.ConstantPad1d((0, padding), 0)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size=kernel_size,
                               stride=stride, padding=0, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU6()

        self.res_path = distiller.modules.EltwiseAdd(inplace=True)

        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # Can use normal weight initialization over xavier if preferred, just remove xavier from train.py
    #     self.init_weights()

    # def init_weights(self):
    #     self.conv1.weight.data.normal_(0, 0.01)
    #     if self.downsample is not None:
    #         self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):

        res = x if self.downsample is None else self.downsample(x)

        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x + res

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, convolution='normal'):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # receptive_field = 1 + (2 * (kernel_size - 1) * (2 ** (i-1)))
            # print('Layer #{} receptive field = {} | kernel = {}'.format(
            #     i, receptive_field, kernel_size))
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                             padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
