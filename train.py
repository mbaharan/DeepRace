#!/usr/bin/python3
import argparse
import sys
import time
from datetime import datetime
import math
import os
import warnings
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TCN, TCNFlops
from ptflops import get_model_complexity_info
import numpy as np

import scipy.io as matloader
from os import path
from collections import OrderedDict
import yaml

from utility.dataloaders import NASADataSet, NASARealTime
from utility.helpers import resume_checkpoint, load_checkpoint, load_checkpoint_post, AverageMeter, CheckpointSaver, get_outdir, init_xavier

warnings.filterwarnings("ignore")   # Suppress the RunTimeWarning on unicode

config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(
    description='Sequence Modeling - Character Level Language Model')

parser.add_argument('--data-dir', type=str, default='./utility/dR11Devs.mat',
                    help='Path to mat file containing transistor degradation')
parser.add_argument('--epochs', type=int, default=1000,
                 help='upper epoch limit (default: 100)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', dest="batch_size",
                    help='batch size (default: 256)')
parser.add_argument('--nhid', type=int, default=37,
                    help='number of hidden units per layer (default: 8)')
parser.add_argument('--input-size', type=int, default=21, dest='input_size',
                    help='valid sequence length (default: 320)')
parser.add_argument('--predict-size', type=int, default=104, dest='predict_size',
                    help='valid sequence length (default: 320)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 3)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate (default: 1)')
parser.add_argument('--min-lr', type=float, default=1e-4, dest='min_lr',
                    help='The lowest LR that scheduling will set (1e-4)')
parser.add_argument('--patience', type=int, default=20,
                    help='How many epochs without progress before LR drops (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--val-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--decay-epochs', type=float, default=200, metavar='N', dest="decay_epochs",
                    help='epoch interval to decay LR')
parser.add_argument('--testset', type=int, default=1,
                    help='The data sample to use as the testset (Default: 11')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use (default: SGD)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--dump-pdf', type=str, default='', dest="dump_pdf",
                    help='Save pdf of model architecture')
parser.add_argument('--total-dev', type=int, default=11, dest='total_dev',
                    help='The number of devices to train against (Default: 11')


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

# Parse arguments
args, model_path, output_dir = _parse_args()

args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

# Load in the MatLab file containing the device samples
filename = args.data_dir

def main():

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

    # Optionally resume from checkpoint
    resume_state = {}
    resume_epoch = None
    if args.resume:
        resume_state, resume_epoch = resume_checkpoint(model, args.resume)
        if 'optimizer' in resume_state:
            print('Restoring Optimizer state from checkpoint')
            optimizer.load_state_dict(resume_state['optimizer'])
        del resume_state

    start_epoch = 0
    if resume_epoch is not None:
        start_epoch = resume_epoch

    if torch.cuda.is_available():
        print('-> Using CUDA!')
        model.cuda()

    """ Initialize weights with xavier algorithm uniformly
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf 
    """
    model.apply(init_xavier)

    # The index of the test sample
    test_idx = [args.testset]

    # RealTime provides more datasamples by creating a sliding window over the entire dataset, incrementing by one each time.
    trainset = NASARealTime(filename,
                             args.input_size,
                             args.predict_size,
                             test_idx,
                             train=True,
                             normalize=True,
                             normal_dis=True,
                             total_dev=args.total_dev)
    trainloader = DataLoader(trainset, num_workers=8,
                             batch_size=args.batch_size, shuffle=True)
    # Fewer datasamples, number of data points / (prediction window + input sequence size)
    testset = NASADataSet(filename,
                            args.input_size,
                            args.predict_size,
                            test_idx,
                            inference=False,
                            train=False,
                            normalize=True,
                            normal_dis=True,
                            total_dev=args.total_dev)
    testloader = DataLoader(testset, num_workers=8,
                            batch_size=args.batch_size, shuffle=False)

    best_metric = None
    best_epoch = None

    # The directory where the model will be saved
    output_base = './output'
    exp_name = '_'.join([
        datetime.now().strftime("%m_%d_%y__%H%M"),
        "TCN",
        str(args.input_size),
        str(args.predict_size),
        str(args.nhid),
        str(args.levels),
        "Dev",
        str(args.testset)
    ])
    output_dir = get_outdir(output_base, 'train', exp_name)

    saver = CheckpointSaver(checkpoint_dir=output_dir, decreasing=True)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)
    num_epochs = args.epochs
    min_lr = args.min_lr
    patience = args.patience
    check_epoch = None
    counter = 0
    lr = optimizer.param_groups[0]['lr']
    print('Scheduled epochs: {}'.format(num_epochs))
    try:
        for epoch in range(start_epoch, num_epochs):

            train_epoch(epoch, model, trainloader, optimizer,
                        loss_fn, args, saver=saver, output_dir=output_dir)

            if (epoch % args.val_interval == 0):
                print()
                eval_metrics = validate(model, testloader, loss_fn)

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


def train_epoch(epoch, model, loader, optimizer, loss_fn, args, saver=None, output_dir=''):
    """Train the network"""
    model.train()

    losses_m = AverageMeter()

    # Used by the compression_scheduler
    last_idx = len(loader) - 1

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()

        output = model(input)

        loss = loss_fn(output, target)
        losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()

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
