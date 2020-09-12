#!/usr/bin/python3
import argparse
import sys
# import time
# from datetime import datetime
import math
import os
import warnings
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from ptflops import get_model_complexity_info
import numpy as np
import scipy.io as matloader
from os import path
from collections import OrderedDict
import yaml

# Utility
from utility.dataloaders import NASADataSet, NASARealTime
from utility.helpers import *
from utility.helpers import _parse_args
from utility.quantize import Quantize

# Batch norm fusion
from pytorch_bn_fusion.bn_fusion_tcn import fuse_bn_sequential

# Distiller Quantization
import distiller as ds
from distiller.data_loggers import collector_context
from distiller import file_config

warnings.filterwarnings("ignore")   # Suppress the RunTimeWarning on unicode

config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(
    description='Sequence Modeling - Character Level Language Model')

parser.add_argument('--data-dir', type=str, default='./utility/dR11Devs.mat',
                    help='Path to mat file containing transistor degradation')

parser.add_argument('--epochs', type=int, default=4000,
                 help='upper epoch limit (default: 100)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', dest="batch_size",
                    help='batch size (default: 256)')
parser.add_argument('--nhid', type=int, default=15,
                    help='number of hidden units per layer (default: 8)')
parser.add_argument('--input-size', type=int, default=21, dest='input_size',
                    help='valid sequence length (default: 320)')
parser.add_argument('--predict-size', type=int, default=104, dest='predict_size',
                    help='valid sequence length (default: 320)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 3)')
parser.add_argument('--levels', type=int, default=3,
                    help='# of levels (default: 4)')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--val-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--decay-epochs', type=float, default=1000, metavar='N', dest="decay_epochs",
                    help='epoch interval to decay LR')
parser.add_argument('--testset', type=int, default=0,
                    help='The data sample to use as the testset (Default: 11')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use (default: SGD)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--total-dev', type=int, default=11, dest='total_dev',
                    help='The number of devices to train against (Default: 11')
parser.add_argument('--focus', dest='focus', action='store_true')
parser.add_argument('--no-focus', dest='focus', action='store_false')
parser.add_argument('--dataset', type=str, default='mosfet',
                    help='Dataset to train on (mosfet, nlp_char, nlp_word, music')

# Online-quantization
parser.add_argument('--quantize_path', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--quantize', dest='quantize', action='store_true')
parser.add_argument('--QAT', dest='QAT', action='store_true')
parser.add_argument('--calibrate', dest='calibrate', action='store_true')
parser.add_argument('--post-training', dest='post_training', action='store_true')

# # Set the bit-width
parser.add_argument('--bit',default=None, type=int,
                     help='Bit width to quantize to')
parser.add_argument('--bit-override',default='', type=str, dest='bit_override',
                     help='Override first layer with 8 bit')
# Batch-fusion
parser.add_argument('--fuse-bn', dest='fuse_bn', action='store_true')

# # Perform range-based linear quantization
parser.add_argument('--qe-config', default=None, type=str,
                    dest='qe_config_file', help='Path to quant_post_linear YAML config file')

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


def main():
    # Parse arguments
    args, output_dir = _parse_args(parser, config_parser)

    # Load in the MatLab file containing the device samples
    filename = args.data_dir

    channel_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize
    dropout = args.dropout

    # Load model for training and inference
    model = TCN(args.input_size, args.predict_size,
                channel_sizes, kernel_size, dropout=dropout)

    # Different model used to calculate the FLOPS, it fits the ptflops input tuple (whereas the model above does not)
    model_flops = TCNFlops(args.input_size, args.predict_size,
                           channel_sizes, kernel_size, dropout=dropout)

    # Calculate the number of trainable parameters within the model
    model_size = sum([m.numel() for m in model.parameters()])
    print('Model {}-nhid {}-layers, {}-ksize, {}-LR, created, param count: {}'.format(
        args.nhid,
        args.levels,
        args.ksize,
        args.lr,
        model_size))

    # Calculate the FLOPS and params (comparing params with the result above)
    flops, params = get_model_complexity_info(
        model_flops, (1, args.input_size), as_strings=False, print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    parameters = model.parameters()
    optimizer = optim.SGD(parameters, lr=args.lr)
    loss_fn = nn.MSELoss()
    if torch.cuda.is_available():
        print('-> Using CUDA!')
        model.cuda()

    # The index of the test sample
    test_idx = [args.testset]

    # RealTime provides more datasamples by creating a sliding window over the entire dataset, incrementing by one each time.
    trainset = NASADataSet(filename,
                             args.input_size,
                             args.predict_size,
                             test_idx,
                             train=True,
                             focus=args.focus,
                             total_dev=args.total_dev)
    trainloader = DataLoader(trainset, num_workers=0,
                             batch_size=args.batch_size, shuffle=False)
    # Fewer datasamples, number of data points / (prediction window + input sequence size)
    testset = NASADataSet(filename,
                            args.input_size,
                            args.predict_size,
                            test_idx,
                            train=False,
                            focus=args.focus,
                            total_dev=args.total_dev)
    testloader = DataLoader(testset, num_workers=0,
                            batch_size=args.batch_size, shuffle=False)

    best_metric = None
    best_epoch = None

  # Optionally resume from checkpoint
    resume_state = {}
    resume_epoch = None

    if args.quantize:
        quantize = Quantize(args, model, optimizer, testloader, loss_fn)
        args, model, compression_scheduler = quantize._args, quantize._model, quantize._compression_scheduler

        if args.calibrate or args.post_training:
            return

    set_inference(args, output_dir)
    # elif resume_state:
    #     resume_state, resume_epoch = resume_checkpoint(model, args.resume)
    #     if 'optimizer' in resume_state:
    #         print('Restoring Optimizer state from checkpoint')
    #         optimizer.load_state_dict(resume_state['optimizer'])
    #     del resume_state

    start_epoch = 0
    if resume_epoch is not None:
        start_epoch = resume_epoch

    if torch.cuda.is_available():
        print('-> Using CUDA!')
        model.cuda()

    # if args.calibrate:
    #     ds.utils.assign_layer_fq_names(model)
    #     print("=> Generating quantization calibration stats based on {0} users".format(args.calibrate))
    #     collector = ds.data_loggers.QuantCalibrationStatsCollector(model)
        
    #     print("=> Validating...")
    #     with collector_context(collector):
    #         validate(model, testloader, loss_fn)


    #     qe_path = os.path.join(qe_dir, 'qe_stats')
    #     if not os.path.isdir(qe_path):
    #         os.mkdir(qe_path)
    #     yaml_path = os.path.join(qe_path, 'quantization_stats.yaml')
    #     collector.save(yaml_path)
    #     print("Quantization statics is saved at {}".format(yaml_path))
    #     return

    # if args.qe_config_file is not None:
    #     print("-> validating post quant...")
    #     validate(model, testloader, loss_fn)
    #     qe_path = os.path.join(qe_dir, 'post_model')
    #     if not os.path.isdir(qe_path):
    #         os.mkdir(qe_path)
    #     args.resume = os.path.join(qe_path, 'model.pth')
    #     torch.save(model.state_dict(), args.resume)
    #     print("Post quantized model is saved at {}".format(args.resume))
    #     return


    print(model)

    saver = CheckpointSaver(checkpoint_dir=output_dir, decreasing=True)

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)
    num_epochs = args.epochs
    check_epoch = None
    counter = 0
    lr = optimizer.param_groups[0]['lr']


    print('Scheduled epochs: {}'.format(num_epochs))
    try:
        for epoch in range(start_epoch, num_epochs):
            
            if args.QAT:
                compression_scheduler.on_epoch_begin(epoch)
            else:
                compression_scheduler = None

            train_epoch(epoch, model, trainloader, optimizer, compression_scheduler,
                        loss_fn, args, saver=saver, output_dir=output_dir)

            print()
            eval_metrics = validate(model, testloader, loss_fn)
        
            if args.QAT:
                compression_scheduler.on_epoch_end(epoch)

            if (saver is not None):
                last_test_loss = save_metric = eval_metrics['loss']
                best_metric, best_epoch = saver.save_checkpoint(
                    model, optimizer, args, epoch=epoch, metric=save_metric)

                # Checks if there has been progress
                if (best_epoch == check_epoch):
                    counter += 1
                else:
                    counter = 0
                check_epoch = best_epoch


                # No progress has been made in 'patience' number of epochs
                if (epoch % args.decay_epochs == 0) and epoch != 0:
                    lr /= 10
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

        print('Saving model size and loss to {}/model_stats.yaml'.format(output_dir))
        stats = [
            {'Model Size': [model_size]},
            {'FLOPS': [flops]},
            {'NHID': [args.nhid]},
            {'levels': [args.levels]},
            {'ksize': [args.ksize]},
            {'testset': [args.testset]},
            {'Loss': [best_metric]}]

        with open('{}/model_stats.yaml'.format(output_dir), 'w') as file:
            yaml.safe_dump(stats, file)

    except KeyboardInterrupt:

        if best_metric is not None:
            print('*** Best metric: {} (epoch {})\n\n'.format(best_metric, best_epoch))

            print('Saving model size and loss to {}/model_stats.yaml'.format(output_dir))
            stats = [
                {'Model Size': [model_size]},
                {'FLOPS': [flops]},
                {'NHID': [args.nhid]},
                {'levels': [args.levels]},
                {'ksize': [args.ksize]},
                {'testset': [args.testset]},
                {'Loss': [best_metric]}]
            with open('{}/model_stats.yaml'.format(output_dir), 'w') as file:
                yaml.safe_dump(stats, file)


def train_epoch(epoch, model, loader, optimizer, compression_scheduler, loss_fn, args, saver=None, output_dir=''):
    """Train the network"""
    model.train()

    losses_m = AverageMeter()

    # Used by the compression_scheduler
    last_idx = len(loader) - 1
    total_samples = len(loader.sampler)
    batch_size = args.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()
        optimizer.zero_grad()
        # Quantization
        if args.QAT:
            compression_scheduler.on_minibatch_begin(epoch, batch_idx, steps_per_epoch, optimizer)

        output = model(input)

        loss = loss_fn(output, target)
        losses_m.update(loss.item(), input.size(0))
        
        if args.QAT:
            compression_scheduler.before_backward_pass(epoch, batch_idx, steps_per_epoch, loss, optimizer=optimizer, return_loss_components=True)
        loss.backward()

        if args.QAT:
            compression_scheduler.before_parameter_optimization(epoch, batch_idx, steps_per_epoch, optimizer)
        optimizer.step()

        if args.QAT:
            compression_scheduler.on_minibatch_end(epoch, batch_idx, steps_per_epoch, optimizer)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            print('Train: {}/{} [({:>3.3f}%)] | Batch: {}/{} [({:>3.3f}%)] | Loss: {loss.val:>9.6f} ({loss.avg:>6.6f})| LR: {lr:.3e}'.format
                  (epoch, args.epochs,
                   (epoch/args.epochs) * 100,
                   batch_idx, last_idx+1,
                   (batch_idx/(last_idx+1)) * 100,
                   loss=losses_m,
                   lr=lr),
                  end='\r')


def validate(model, loader, loss_fn):
    """Test the trained network"""
    model.eval()

    losses_m = AverageMeter()

    batch_idx = -1
    with torch.no_grad():
        for _, (input, target) in enumerate(loader):
            batch_idx = batch_idx + 1

            if torch.cuda.is_available():
                input, target = input.cuda(), target.cuda()

            output = model(input)

            loss = loss_fn(output, target)

            torch.cuda.synchronize()

            losses_m.update(loss.data.item(), input.size(0))

            log_name = 'Test'
            print('{0}: [{1:>4d}] | Loss: {loss.val:>7.9f} (--> {loss.avg:>6.9f} <--)'.format(
                log_name,
                batch_idx,
                loss=losses_m), end='\r')

        metrics = OrderedDict([('loss', losses_m.avg)])
        print()

        return metrics

if __name__ == '__main__':
    main()
