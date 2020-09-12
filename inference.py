#!/usr/bin/python3

import argparse
from argparse import ArgumentParser
import sys
import time
import math
import os
import warnings
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utility.generate_sample import generate_sample
from utility.dataloaders import NASADataSet, NASARealTime
from utility.helpers import load_checkpoint, load_checkpoint_post, _parse_args
from model import TCN
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import numpy as np
import scipy.io as matloader
import yaml
from sklearn.metrics import mean_squared_error
import time
import os
# Batch norm fusion
from pytorch_bn_fusion.bn_fusion_tcn import fuse_bn_sequential

config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(
    description='Sequence Modeling - Character Level Language Model')
parser.add_argument('--testset', type=int, default=0,
                    help='The data sample to use as the testset (Default: 1')
parser.add_argument('--threshold', type=float, default=0.05,
                    help="Calculate the RUL at given time.")
parser.add_argument('--rul-time', type=int, nargs='*',
                    help="Calculate the RUL at given time.")
parser.add_argument('--anime', type=int, default=0,
                    help='The data sample to use as the testset (Default: 1')
parser.add_argument('-s', '--save', default='', type=str, metavar='PATH',
                    help='Path to save figure')
parser.add_argument('-f', '--filename', default='utility/dR11Devs.mat', type=str, metavar='PATH',
                    help='Path to the dataset')
parser.add_argument('-t', '--tail', default='', type=str,
                    help='Path to save figure')
# Batch-fusion
parser.add_argument('--fuse-bn',default=None, type=str,
                    dest='fuse_bn', help='Fuse BN to Conv')
parser.add_argument('--quantize', default='', type=str, metavar='PATH',
                    help='Resume and quantize pretrained model')
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


def find_index(arr):
    """Find the threshold index, if it exists, and return it.
    arr is a boolean array comparing each element >= threshold value,
    where True if it is >=, and False if not"""
    print('-> Finding threshold index...')
    # Determine if threshold value exists in current array
    # any() returns True if any value of the array is True and False otherwise
    if not arr.any():
        return -1
    
    # Return the FIRST index that is >= the threshold value (True), which is 0.05 as from the NASA paper.
    return np.argmax(arr)


def find_error_at_5_percent(output, target, index):
    """Finds the error between the predicted value and target value at the 
    timestep index when the target value >= 0.05 (5% R_DS(ON))"""

    print("-> Calculating the error at 5%...")
    return (abs((output[index] - target[index])) / target[index]) * 100


def find_log_mse(tar_error, out_error):
    """Calculates the log of the mean squared error (MSE)"""

    print('-> Calculating Log(MSE)...')
    mse = mean_squared_error(tar_error, out_error)
    return np.log(mse)

def report(out, tar, error_index, error_at_5_percent, logMSE, output_dir, device_num, RUL, RUL_hat, RUL_time, RUL_error, RA):
    """Logs the results to the console"""

    print('_' * 80)
    print('===  Current Test: {}'.format(device_num))
    print('-' * 80)
    #print('\nAverage inference time for (1/{}) predictions: {:.4}'.format(batch_count, avg_time))
    print('5% Prediction = {:.8}\n5% Target = {:.8}\n5% Error = {:.4}%'.format(out[error_index], tar[error_index], error_at_5_percent))
    print('Log(MSE) = {}\n'.format(logMSE))

    for i in range(len(RUL_time)):
        print('*' * 80)
        print('Time: {}'.format(RUL_time[i]))
        print('Actual RUL = {:3.1f}'.format(RUL[i]))
        print('Estimated RUL = {:3.1f}'.format(RUL_hat[i]))
        print('Error = {:3.1f}'.format(RUL_error[i]))
        print('RA = {:3.1f}\n'.format(RA[i]))

    print('Average RA: {}'.format(np.average(RA)))
    print('Average RUL Error: {}'.format(np.average(RUL_error)))
    print('Saving statistics to {}'.format(output_dir))     
    print('_' * 80)


def plot(out, tar, device_num, output_dir):
    """Plots the target vs prediction for the current test device"""
    
    plt.figure(figsize=(8.5, 2.85))
    #plt.hlines(y=0.05, xmin=0, xmax=len(out), colors='g', linestyles='-', lw=2)
    plt.plot(range(len(out)), out, 'b',
                label='Testset# {} - Predicted'.format(device_num))
    plt.plot(range(len(tar)), tar, 'r', alpha=0.6,
                label='Testset# {} - Measured'.format(device_num))

    plt.legend(loc='upper left')
    plt.xlabel('Num. of Samples')
    plt.ylabel('ΔR')
    # if args.save:
    #     if error_at5_percent:
    #         plt.savefig('{}/device_{}_tail.png'.format(output_dir, device_num))
    #         plt.savefig('{}/device_{}_tail.png'.format('results', device_num))
        # else:
    print('+ Saving plot to: {}/{}.png'.format(output_dir, device_num))
    plt.savefig('{}/{}.png'.format(output_dir, device_num))
    #plt.savefig('{}/{}_full.png'.format('results', device_num))


def find_degradation(args, model, mean, std, prediction, error_index, mat, actual_device):
    sample_per_min = mat['SamplePerMinuts'][0, args.testset][0]
    deg_at = np.argmax(actual_device >= args.threshold) // sample_per_min
    
    RUL = np.array([])
    RUL_hat = np.array([])
    RUL_time = np.array([])
    RUL_error = np.array([])
    RA = np.array([])

    for rul_time in args.rul_time:
        print("-> Calculating the RUL at {} considering threshold={}.".format(rul_time, args.threshold))
        if rul_time > deg_at:
            raise ValueError("Given time for RUL is bigger than {}.".format(deg_at))

        rul_time_idx = int(rul_time * sample_per_min[0])

        _, _, _, target_rul, _, _ = generate_sample(filename=args.filename,
                                                                batch_size=1,
                                                                samples=args.input_size, 
                                                                predict=args.predict_size,
                                                                start_from=rul_time_idx - (args.input_size + args.predict_size),
                                                                test=True,
                                                                test_set=[args.testset])


        _RUL = 100 * abs(actual_device[rul_time_idx] - actual_device[-1]) / actual_device[-1]
        _RUL_hat = 100 * abs(prediction[-1] - target_rul[0][-1]) / prediction[-1]
        _RUL_error = _RUL - _RUL_hat
        _RA = 100 * (1 - abs(_RUL_error / _RUL))

        RUL = np.append(RUL, _RUL)
        RUL_hat = np.append(RUL_hat, _RUL_hat)
        RUL_time = np.append(RUL_time, rul_time)
        RUL_error = np.append(RUL_error, _RUL_error)
        RA = np.append(RA, _RA)

    
    RA_avg = np.average(RA)
    RUL_error_avg = np.average(RUL_error)
    return RUL, RUL_hat, RUL_time, RUL_error, RA, RA_avg, RUL_error_avg


def main():
    # Parse arguments
    args, output_dir = _parse_args(parser, config_parser)

    # Current test device
    test_idx = [args.testset]
    
    # Load dataset from matlab file
    mat = matloader.loadmat(args.filename)

    # Dataloader for the test device
    testset = NASADataSet(args.data_dir,
                          args.input_size,
                          args.predict_size,
                          test_idx,
                          focus=args.focus,
                          train=False)#, error_at5percent=error_at5_percent)
    testloader = DataLoader(testset, batch_size=1,
                            shuffle=False, drop_last=True)

    std = testloader.dataset.std
    mean = testloader.dataset.mean

    # Current test device number
    device_num = testset.dev_names[0, args.testset][0]
    print('+ Current test device: {} '.format(device_num))

    # Model parameters
    channel_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize
    dropout = args.dropout

    # Load PyTorch model
    model = TCN(args.input_size, args.predict_size,
                     channel_sizes, kernel_size, dropout=dropout)

    parameters = model.parameters()
    optimizer = optim.SGD(parameters, lr=args.lr)
    loss_fn = nn.MSELoss()

    if args.quantize:
        output_dir = args.quantize_path
        # Fuse BN layers (step 1a)
        if args.fuse_bn:
            print("-> Fusing BN to conv...")
            #ds.model_transforms.fold_batch_norms(model, dummy_input)
            model.eval()
            model = fuse_bn_sequential(model)
            model.train()
            print("-> Loading from checkpoint...")
            same_keys, new_state_dict = load_checkpoint_post(model, '{}/model_best.pth'.format(args.quantize_path))
    else:
        output_dir = args.train_path
        load_checkpoint(model, '{}/model_best.pth'.format(args.train_path))

    # Move model to GPU if available
    if torch.cuda.is_available():
        print('+ Using CUDA!')
        model.cuda()

   # The current device being tested
    actual_device = mat['vals'][0, args.testset][0]

    total_dev_len = actual_device.size
    how_many_seg = int(total_dev_len / (args.input_size + args.predict_size))

    # The index (timestep) where the target (actual) value is >= the threshold (0.05)
    error_index = find_index(actual_device >= args.threshold)
    if error_index == -1:
        error_index = total_dev_len - 1
        print('{} Cannot be defined as the threshold value. It has been changed to the highest value found'.format(args.threshold))


    out = np.array([])
    tr = np.array([])

    pred_lst = np.array([])
    out_lst = np.array([])

    # Collect the input, prediction, and target, for the entire dataset.
    for i in range(how_many_seg):
        _, y, next_t, expected_y, _, _ = generate_sample(filename=args.filename,
                                                            batch_size=1, samples=args.input_size, predict=args.predict_size,
                                                            start_from=i * (args.input_size + args.predict_size), test=True, test_set=test_idx)

        test_input = (y - mean)/std
        test_input = torch.from_numpy(test_input).float().cuda()

        prediction = model(test_input)
        prediction = (prediction * std) + mean
        prediction = prediction.cpu().detach().numpy()

        pred_lst = np.hstack((pred_lst, y[0]))  # Input Seq
        pred_lst = np.hstack((pred_lst, prediction[0]))  # Prediction

        out_lst = np.hstack((out_lst, y[0]))
        out_lst = np.hstack((out_lst, expected_y[0]))

        out = np.hstack((out, prediction[0]))
        tr = np.hstack((tr, expected_y[0]))


    # Get the prediction at threshold (0.05)
    _, y_p, _, expected_y_p, _, _ = generate_sample(filename=args.filename,
                                                batch_size=1, samples=args.input_size, predict=args.predict_size,
                                                start_from=error_index - (args.input_size + args.predict_size), test=True, test_set=test_idx)

    y_p = (y_p - mean)/std
    y_p = torch.from_numpy(y_p).float().cuda()
    
    expected_y_p = torch.from_numpy(expected_y_p).float().cuda()
    prediction_5p = model(y_p)
    prediction_5p = (prediction_5p * std) + mean
    prediction_5p = prediction_5p.cpu().detach().numpy()

    # Calculate the error at 5% ( R_DS(ON) = 0.05 )
    error_at_5_percent = find_error_at_5_percent(prediction_5p[0], expected_y_p[0], -1)

    # Calculate the Log(MSE)
    logMSE = find_log_mse(out, tr)

    # Plot the predicted R_DS(ON) for the entire device lifetime
    plot(pred_lst, out_lst, device_num, output_dir)

    # Calculate the device degradation
    RUL, RUL_hat, RUL_time, RUL_error, RA, RA_avg, RUL_error_avg = find_degradation(args, model, mean, std, prediction_5p[0], error_index, mat, actual_device)
    
    # Logs results
    report(
           prediction_5p[0],
           expected_y_p[0],
           -1,
           error_at_5_percent,
           logMSE,
           output_dir,
           device_num,
           RUL, RUL_hat, RUL_time, RUL_error, RA)

    # Stats to dump into YAML file
    stats = [
        {'Device_#' : [str(device_num)]},
        {'Error_at_5_percent' : [float(error_at_5_percent)]},
        {'Log(MSE)' : [float(logMSE)]},
        {'Average RA' : [float(RA_avg)]},
        {'Average RUL Error' : [float(RUL_error_avg)]}]

    with open('{}/inference_stats.yaml'.format(output_dir), 'w') as file:
        yaml.safe_dump(stats, file)
        for (rul, rul_hat, time, error, ra) in zip(RUL, RUL_hat, RUL_time, RUL_error, RA):
            rul_stats = [
                {'RUL_Time' : [float(time)]},
                {'RUL_Actual' : [float(rul)]},
                {'RUL_Estimated' : [float(rul_hat)]},
                {'RUL_Error' : [float(error)]},
                {'RA' : [float(ra)]}]
            yaml.safe_dump(rul_stats, file)

    # Dump prediction array for matlab plotting
    dump_path = '{}/RUL_{}.txt'.format(output_dir, device_num)
    np.savetxt(dump_path, pred_lst, fmt="%f", newline='\r\n')
    
if __name__ == '__main__':
    main()
