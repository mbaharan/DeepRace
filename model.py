import torch.nn.functional as F
from torch import nn
from tcn import TemporalConvNet

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

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, convolution):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(1, num_channels, kernel_size=kernel_size, dropout=dropout, convolution=convolution)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = inputs.unsqueeze(1)
        x = self.tcn(x) 
        x = self.linear(x[:, :, -1])

        return x

'''
Used to measure the FLOPS of the model, the flop module requires the input to be a certain way while the training requires
it to be in another arrangement. This is the reason for two separate classes
'''
class TCNFlops(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, convolution):
        super(TCNFlops, self).__init__()
        self.tcn = TemporalConvNet(1, num_channels, kernel_size=kernel_size, dropout=dropout, convolution=convolution)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = inputs#.unsqueeze(1)
        x = self.tcn(x)  # input should have dimension (N, C, L)
        x = self.linear(x[:, :, -1])

        return x