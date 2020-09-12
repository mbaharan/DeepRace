import torch
import torch.nn as nn
import operator
import os
from collections import OrderedDict
import glob
import shutil
import argparse
import time
from datetime import datetime
import yaml
from model import *

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def get_state_dict(model):
    return unwrap_model(model).state_dict()


def resume_checkpoint(model, checkpoint_path):
    other_state = {}
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            if 'optimizer' in checkpoint:
                other_state['optimizer'] = checkpoint['optimizer']
            if 'amp' in checkpoint:
                other_state['amp'] = checkpoint['amp']
            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save
            print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            print("Loaded checkpoint '{}'".format(checkpoint_path))
        return other_state, resume_epoch
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            state_dict_key = 'state_dict'
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)
        print("+ Loaded {} from checkpoint '{}'".format(state_dict_key or 'weights', checkpoint_path))
    else:
        print("!!! No checkpoint found at '{}'!!!".format(checkpoint_path))
        raise FileNotFoundError()

def load_checkpoint_post(model, checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            state_dict_key = 'state_dict'
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
                
            same_keys = (new_state_dict.keys() & model.state_dict().keys())
            #model.load_state_dict(new_state_dict)
            return same_keys, new_state_dict
        else:
            model.load_state_dict(checkpoint)
        logging.info("Loaded {} from checkpoint '{}'".format(state_dict_key or 'weights', checkpoint_path))
    else:
        logging.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
    
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CheckpointSaver:
    def __init__(
            self,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            max_history=2):

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.last_recovery_file = ''

        # config
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = '.pth'
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
        self.max_history = max_history
        assert self.max_history >= 1

    def save_checkpoint(self, model, optimizer, args, epoch, model_ema=None, metric=None, use_amp=False):
        assert epoch >= 0
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (len(self.checkpoint_files) < self.max_history
                or metric is None or self.cmp(metric, worst_file[1])):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            self._save(save_path, model, optimizer, args, epoch, model_ema, metric, use_amp)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1],
                reverse=not self.decreasing)  # sort in descending order if a lower metric is not better

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += ' {}\n'.format(c)
            print(checkpoints_str)

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                shutil.copyfile(save_path, os.path.join(self.checkpoint_dir, 'model_best' + self.extension))

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _save(self, save_path, model, optimizer, args, epoch, model_ema=None, metric=None, use_amp=False):
        save_state = {
            'epoch': epoch,
            'arch': args.nhid,
            'state_dict': get_state_dict(model),
            'optimizer': optimizer.state_dict(),
            'args': args,
            'version': 2,  # version < 2 increments epoch before save
        }
        if use_amp and 'state_dict' in amp.__dict__:
            save_state['amp'] = amp.state_dict()
        if model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(model_ema)
        if metric is not None:
            save_state['metric'] = metric
        torch.save(save_state, save_path)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index <= 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                print("Cleaning checkpoint: {}".format(d))
                os.remove(d[0])
            except Exception as e:
                print("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def save_recovery(self, model, optimizer, args, epoch, model_ema=None, use_amp=False, batch_idx=0):
        assert epoch >= 0
        filename = '-'.join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        self._save(save_path, model, optimizer, args, epoch, model_ema, use_amp=use_amp)
        if os.path.exists(self.last_recovery_file):
            try:
                print("Cleaning recovery: {}".format(self.last_recovery_file))
                os.remove(self.last_recovery_file)
            except Exception as e:
                print("Exception '{}' while removing {}".format(e, self.last_recovery_file))
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path

    def find_recovery(self):
        recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
        files = glob.glob(recovery_path + '*' + self.extension)
        files = sorted(files)
        if len(files):
            return files[0]
        else:
            return ''


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir

def init_xavier(m):
    if type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight.data)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight.data)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)

def set_inference(args, output_dir):
    if args.focus:
        inference_path = 'scripts/inference1.sh'
    else:
        inference_path = 'scripts/inference2.sh'

    print('Setting inference path to: {}'.format(inference_path))

    
    if args.testset == 0:
        with open (inference_path, 'w') as rsh:
            rsh.write("#! /bin/bash\npython3 inference.py --config {} --rul-time 77 83 90 95 101 107 114 117 119 122 123".format(output_dir +'/args.yaml'))

    if args.testset == 1:
        with open (inference_path, 'w') as rsh:
            rsh.write("#! /bin/bash\npython3 inference.py --config {} --rul-time 119 128 138 147 156 164 175 180 184 188 193 194".format(output_dir +'/args.yaml'))

    if args.testset == 4:
        with open (inference_path, 'w') as rsh:
            rsh.write("#! /bin/bash\npython3 inference.py --config {} --rul-time 89 95 100 106 113 116 119 121 124 126 133".format(output_dir +'/args.yaml'))

    if args.testset == 9:
        with open (inference_path, 'w') as rsh:
            rsh.write("#! /bin/bash\npython3 inference.py --config {} --rul-time 130 139 151 161 170 175 180 185 189".format(output_dir +'/args.yaml'))


def _parse_args(parser, config_parser):
    """Parse command line argument files (yaml) from previous trainning sessions
    or within this file itself"""

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()

    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)

    output_base = './output'
    exp_name = '_'.join([
    datetime.now().strftime("%m_%d_%y__%H%M%S"),
    "TCN",
    str(args.input_size),
    str(args.predict_size),
    str(args.nhid),
    str(args.levels),
    "Dev",
    str(args.testset),
    str(args.focus)])

    if not args_config.config:
        output_dir = get_outdir(output_base, 'train', exp_name)
        args.train_path = output_dir

    if args.quantize:
        if args.quantize_path:
            output_dir = os.path.join(output_base, 'train', exp_name)
        else:
            output_dir = get_outdir(output_base, 'train', exp_name)
            args.quantize_path = '{}/{}'.format(os.getcwd(), output_dir)
    else:
        output_dir = os.path.join(output_base, 'train', exp_name)

    return args, output_dir
