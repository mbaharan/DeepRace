import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import sys
import os
import numpy as np
from utility.helpers import init_xavier


"""
Copyright (c) 2020, University of North Carolina at Charlotte All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Authors: Reza Baharani - Transformative Computer Systems Architecture Research (TeCSAR) at UNC Charlotte
         Steven Furgurson - Transformative Computer Systems Architecture Research (TeCSAR) at UNC Charlotte
"""


"""
A Temporal Convolutional Network (TCN).
Inspired by
    https://github.com/locuslab/TCN
    https://arxiv.org/abs/1803.01271

"""


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()

        # kernel_size = (1, kernel_size)
        # dilation = (1, dilation)
        
        self.pad1 = nn.ConstantPad1d((0, padding), 0)

        self.DSConv1 = nn.Sequential(
            nn.Conv1d(n_inputs, n_inputs, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=n_inputs),
            nn.BatchNorm1d(n_inputs),
            nn.ReLU6(),
            nn.Conv1d(n_inputs, n_outputs, kernel_size=1),
            nn.BatchNorm1d(n_outputs),
            nn.ReLU6()
        )

        self.pad2 = nn.ConstantPad1d((0, padding), 0)

        self.DSConv2 = nn.Sequential(
            nn.Conv1d(n_outputs, n_outputs, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=n_outputs),
            nn.BatchNorm1d(n_outputs),
            nn.ReLU6(),
            nn.Conv1d(n_outputs, n_outputs, kernel_size=1),
            nn.BatchNorm1d(n_outputs),
            nn.ReLU6()
        )

        self.relu = nn.ReLU6()

        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):

        res = x if self.downsample is None else self.downsample(x)

        x = self.pad1(x)
        x = self.DSConv1(x)
        x = self.pad2(x)
        x = self.DSConv2(x)
        x = x + res
        out = self.relu(x)

        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
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

        """ Initialize weights with xavier algorithm uniformly
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf 

        """
        self.network.apply(init_xavier)
    def forward(self, x):
        return self.network(x)
