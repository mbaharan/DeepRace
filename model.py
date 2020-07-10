import torch.nn.functional as F
from torch import nn
from tcn import TemporalConvNet


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