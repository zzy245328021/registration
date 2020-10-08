#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/3/27 10:18
# @Author  : Eric Ching
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def dice_coef(input, target, threshold=0.5):
    smooth = 1.
    iflat = (input.view(-1) > threshold).float()
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

def dice_coef_np(input, target, eps=1e-7):
    input = np.ravel(input)
    target = np.ravel(target)
    intersection = (input * target).sum()

    return (2. * intersection) / (input.sum() + target.sum() + eps)

def parse_lable_subpopulation(input, target, label_dict):
    sub_dice = {}
    for label_id, label_name in label_dict.items():
        lid = int(label_id)
        if lid == 0 or lid ==181 or lid ==182:
            continue
        sub_input = input == lid
        sub_target = target == lid
        dsc = dice_coef_np(sub_input, sub_target)
        sub_dice[label_id] = dsc

    return sub_dice


