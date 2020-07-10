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
from utility.helpers import load_checkpoint
from model import TCN
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy.io as matloader
import yaml
from sklearn.metrics import mean_squared_error
import time
import os

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


def _parse_args():
    """Parse command line argument files (yaml) from previous trainning sessions
    or within this file itself"""

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # Find the output directory and model path 
    output_dir = args_config.config.split('/')
    output_dir.pop(-1)
    output_dir = ('/').join(output_dir)
    model_path = output_dir + '/model_best.pth'
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args, model_path, output_dir


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

def report(batch_count, avg_time, out, tar, error_index, error_at_5_percent, logMSE, output_dir, device_num, RUL, RUL_hat, RUL_time, RUL_error, RA):
    """Logs the results to the console"""

    print('_' * 80)
    print('===  Current Test: {}'.format(device_num))
    print('-' * 80)
    print('\nAverage inference time for (1/{}) predictions: {:.4}'.format(batch_count, avg_time))
    print('5% Prediction = {:.8}\n5% Target = {:.8}\n5% Error = {:.4}%'.format(out[error_index], tar[error_index], error_at_5_percent))
    print('Log(MSE) = {}\n'.format(logMSE))

    for i in range(len(RUL_time)):
        print('*' * 80)
        print('Time: {}'.format(RUL_time[i]))
        print('Actual RUL = {:3.1f}'.format(RUL[i]))
        print('Estimated RUL = {:3.1f}'.format(RUL_hat[i]))
        print('Error = {:3.1f}'.format(RUL_error[i]))
        print('RA = {:3.1f}\n'.format(RA[i]))

    print('Saving statistics to {}'.format(output_dir))     
    print('_' * 80)


def plot(out, tar, device_num, output_dir):
    """Plots the target vs prediction for the current test device"""
    
    plt.figure(figsize=(8.5, 2.85))
    #plt.hlines(y=0.05, xmin=0, xmax=len(out), colors='g', linestyles='-', lw=2)
    plt.plot(range(len(out)), out, 'b',
                label='Testset# {} - Predicted'.format(device_num))
    plt.plot(range(len(out)), tar, 'r', alpha=0.6,
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


def find_degradation(args, error_index, mat, out_error, tar_error, final_gold):
    sample_per_min = mat['SamplePerMinuts'][0, args.testset][0]
    deg_at = np.argmax(final_gold >= args.threshold) // sample_per_min
    
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

        _, y_p_rul, _, expected_y_p_rul, _, _ = generate_sample(filename=args.filename,
                                                                batch_size=1,
                                                                samples=args.input_size, 
                                                                predict=args.predict_size,
                                                                start_from=rul_time_idx - (args.input_size + args.predict_size),
                                                                test=True,
                                                                test_set=[args.testset])

        _RUL = 100 * abs(final_gold[rul_time_idx] - final_gold[-1]) / final_gold[-1]
        _RUL_hat = 100 * abs(out_error[-1] - expected_y_p_rul[0][-1]) / out_error[-1]
        _RUL_error = _RUL - _RUL_hat
        _RA = 100 * (1 - abs(_RUL_error / _RUL))

        RUL = np.append(RUL, _RUL)
        RUL_hat = np.append(RUL_hat, _RUL_hat)
        RUL_time = np.append(RUL_time, rul_time)
        RUL_error = np.append(RUL_error, _RUL_error)
        RA = np.append(RA, _RA)

    return RUL, RUL_hat, RUL_time, RUL_error, RA

def inference(model, loader, args):
    """Test the trained network"""
    print('-> Running inference...')
    # Set Batch Norm and Dropout to eval mode
    # Batch norm will use the running mean and variance calculated during training
    model.eval()

    # Turns off back propagation (learning) and speeds up computation
    with torch.no_grad():
        # The output (prediction) of the model
        out_error = np.array([])
        out_plot = np.array([])

        # The target (actual) outcome
        tar_error = np.array([])
        tar_plot = np.array([])

        # Keep count of the number of batches so we can find the average inference time per batch
        batch_count = 0

        for _, (input, target) in enumerate(loader):
            # Sums up all of the batch inference times
            sum_batch_inference_time = 0

            # Increment the batch count
            batch_count += 1
            
            # Send input and target to GPU for computation, if available
            if torch.cuda.is_available():
                input, target = input.cuda(), target.cuda()

            # Start the inference timer
            tstart = time.time()

            # Inference
            output = model(input)

            # End the inference timer
            tend = time.time()

            # inference time
            inference_time = tend - tstart

            # Running total of inference time
            sum_batch_inference_time += inference_time

            # Use normal (Gaussian) distribution
            if loader.dataset.normal_dis:
                std = loader.dataset.std
                mean = loader.dataset.mean
                # Convert Torch.Tensor to numpy array for calculations
                input_numpy = input[0].cpu().numpy()
                output_numpy = output[0].cpu().numpy()
                target_numpy = target[0].cpu().numpy()
            
            # Use min/max normalization
            else:
                std = (loader.dataset.max_arr - loader.dataset.min_arr)/2
                mean = loader.dataset.min_arr
                # Convert Torch.Tensor to numpy array for calculations
                input_numpy = input[0].cpu().numpy() + 1
                output_numpy = output[0].cpu().numpy() + 1 
                target_numpy = target[0].cpu().numpy() + 1

            # Append the denormalized input/output to the plot array (needs both input and output to plot)
            out_plot = np.append(out_plot, ((input_numpy * std) + mean))
            out_plot = np.append(out_plot, ((output_numpy * std) + mean))

            # Append the denormalized output to the output error array (Faster error calculation without input)
            out_error = np.append(out_error, ((output_numpy * std) + mean))

            # Append the denormalized input/target to the plot array (needs both input and target to plot)
            tar_plot = np.append(tar_plot, ((input_numpy * std) + mean))
            tar_plot = np.append(tar_plot, ((target_numpy * std) + mean))

            # Append the denormalized output to the tar (target) array Faster error calculation without input)
            tar_error = np.append(tar_error, ((target_numpy * std) + mean))

            # Waits for all kernels in all streams on a CUDA device to complete
            torch.cuda.synchronize()

        # Calculate the average inference time
        avg_time = sum_batch_inference_time / batch_count

    return out_plot, out_error, tar_plot, tar_error, avg_time, batch_count


def main():
    # Parse arguments
    args, model_path, output_dir = _parse_args()
    
    # Output DIR for quantization stats
    quant_stats = '{}/qe_stats/quantization_stats.yaml'.format(output_dir)
    print("Quant stats @ {}".format(quant_stats))

    # Current test device
    test_idx = [args.testset]
    
    # Load dataset from matlab file
    mat = matloader.loadmat(args.filename)

    # No longer used?
    if args.tail:
        error_at5_percent = True
    else:
        error_at5_percent = False

    # Dataloader for the test device
    testset = NASADataSet(args.data_dir,
                          args.input_size,
                          args.predict_size,
                          test_idx,
                          inference=True,
                          train=False,
                          normalize=True)#, error_at5percent=error_at5_percent)
    testloader = DataLoader(testset, batch_size=1,
                            shuffle=False, drop_last=True)

    # Current test device number
    device_num = testset.dev_names[0, args.testset][0]
    print('+ Current test device: {} '.format(device_num))

    # Model parameters
    channel_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize
    dropout = args.dropout

    # Load PyTorch model
    model = TCN(args.input_size, args.predict_size,
                     channel_sizes, kernel_size, dropout=dropout, convolution=args.convolution)

    parameters = model.parameters()
    optimizer = optim.SGD(parameters, lr=args.lr)
    loss_fn = nn.MSELoss()

    # Load pretrained model
    load_checkpoint(model, model_path)

    # Move model to GPU if available
    if torch.cuda.is_available():
        print('+ Using CUDA!')
        model.cuda()

    # Run inference
    out_plot, out_error, tar_plot, tar_error, avg_time, batch_count = inference(model, testloader, args)

    chomp = len(out_error) - len(tar_error)
    out_error = out_error[:-chomp]
    out_plot = out_plot[:-chomp]

    # The current device being tested
    final_gold = mat['vals'][0, args.testset][0]

    # The index (timestep) where the target (actual) value is >= the threshold (0.05)
    error_index = find_index(tar_error >= args.threshold)
    if error_index == -1:
        max_value = np.max(tar_error)
        max_index = np.where(tar_error == np.max(tar_error))

        print('Threshold value {} does not exist in device {}. Max value: {} at index {}'.format(args.threshold, args.testset, max_value, max_index))
        sys.exit(-1)
    
    # Calculate the error at 5% ( R_DS(ON) = 0.05 )
    error_at_5_percent = find_error_at_5_percent(out_error, tar_error, error_index)

    # Calculate the Log(MSE)
    logMSE = find_log_mse(out_error, tar_error)

    # Plot the predicted R_DS(ON) for the entire device lifetime
    plot(out_plot, tar_plot, device_num, output_dir)

    # Calculate the device degradation
    RUL, RUL_hat, RUL_time, RUL_error, RA = find_degradation(args, error_index, mat, out_error, tar_error, final_gold)
    
    # Logs results
    report(batch_count,
           avg_time,
           out_error,
           tar_error,
           error_index,
           error_at_5_percent,
           logMSE,
           output_dir,
           device_num,
           RUL, RUL_hat, RUL_time, RUL_error, RA)

    # Stats to dump into YAML file
    stats = [
        {'Device_#' : [str(device_num)]},
        {'Average_inference_time' : [avg_time]},
        {'Error_at_5_percent' : [float(error_at_5_percent)]},
        {'Log(MSE)' : [float(logMSE)]}]

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
    dump_path = '{}/res_{}.txt'.format(output_dir, device_num)
    np.savetxt(dump_path, out_plot, fmt="%f", newline='\r\n')
    
if __name__ == '__main__':
    main()
