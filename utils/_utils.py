#encoding: utf-8
from __future__ import print_function
import os
import random
import torch
import numpy as np
import logging
import warnings
import sys
import shutil
import csv

def init_env(gpu_id='0', seed=42):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('exist path: ', path)


def set_logger(log_path, level=logging.INFO):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
        logging.info("Starting training...")

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    if not logger.handlers:
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        # formatter = logging.Formatter('%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        logger.addHandler(stream_handler)

        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode == 'sec':
        #t = int(t)
        min = t // 60
        sec = t % 60
        return '%2f min %02f sec' % (min, sec)
    else:
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value"""

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

def save_checkpoint(state, is_best, file_name):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        path: (string) folder where parameters are to be saved
    """
    save_dir = os.path.split(file_name)[0]
    mkdir(save_dir)
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(save_dir, 'best.pth.tar'))

def load_checkpoint(model, checkpoint, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
        optimizer assuming it is present in checkpoint.

        Args:
            checkpoint: (string) filename which needs to be loaded
            model: (torch.nn.Module) model for which the parameters are loaded
            optimizer: (torch.optim) optional: resume optimizer from checkpoint
        """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    lr = lr[0]

    return lr

def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def dict_to_csv(dct, csv_path):
    with open(csv_path, 'wt') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, dct.keys())
        w.writeheader()
        w.writerow(dct)
def list_to_csv(dct, csv_path):
    with open(csv_path, 'wt') as f:  # Just use 'w' mode in 3.x
        w = csv.writer(f)
        w.writerow(dct)