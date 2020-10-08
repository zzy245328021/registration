#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/3/27 10:17
# @Author  : Eric Ching
import torch
from math import exp


from torch.nn import functional as F
import numpy as np
from model import dct
from torch.autograd import Variable
def dice_loss(input, target):
    """soft dice loss"""
    eps = 1e-7
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - (2. * intersection / ((iflat**2).sum() + (tflat**2).sum() + eps))




def patch_ncc_loss(I, J, win=(9, 9, 9), eps=1e-5):
    # compute CC squares

    ndims = len(I.size()) - 2
    I2 = I * I
    J2 = J * J
    IJ = I * J
    conv_fn = getattr(F, 'conv%dd' % ndims)
    sum_filt = torch.ones([1, 1, *win], dtype=torch.float).cuda()
    strides = [1] * ndims
    I_sum = conv_fn(I, sum_filt, stride=strides, padding=win[0]//2)
    J_sum = conv_fn(J, sum_filt, stride=strides, padding=win[0] // 2)
    I2_sum = conv_fn(I2, sum_filt, stride=strides, padding=win[0] // 2)
    J2_sum = conv_fn(J2, sum_filt, stride=strides, padding=win[0] // 2)
    IJ_sum = conv_fn(IJ, sum_filt, stride=strides, padding=win[0] // 2)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    cc = cross * cross / (I_var * J_var + eps)

    return -torch.mean(cc)






def ncc_loss(I, J):
    mean_I = I.mean([1, 2, 3, 4], keepdim=True)
    mean_J = J.mean([1, 2, 3, 4], keepdim=True)
    I2 = I * I
    J2 = J * J
    mean_I2 = I2.mean([1, 2, 3, 4], keepdim=True)
    mean_J2 = J2.mean([1, 2, 3, 4], keepdim=True)
    stddev_I = torch.sqrt(mean_I2 - mean_I * mean_I).sum([1, 2, 3, 4], keepdim=True)
    stddev_J = torch.sqrt(mean_J2 - mean_J * mean_J).sum([1, 2, 3, 4], keepdim=True)

    return -torch.mean((I - mean_I) * (J - mean_J) / (stddev_I * stddev_J))

def l1_smooth3d(flow):
    """computes TV loss over entire composed image since gradient will
     not be passed backward to input
    计算图像梯度平均值
    Args:
        flow: 5d tensor, [batch, height, width, depth, channel(translation)]
    """
    loss = torch.mean(torch.mean(torch.abs(flow[:, 1:, :,  :, :] - flow[:, :-1, :,   :, :])) +
                      torch.mean(torch.abs(flow[:, :, 1:,  :, :] - flow[:, :,   :-1, :, :])) +
                      torch.mean(torch.abs(flow[:, :,  :, 1:, :] - flow[:, :,   :,   :-1, :])))
    return loss


def l2_smooth3d(flow):
    """computes TV loss over entire composed image since gradient will
     not be passed backward to input
    计算图像梯度平均值
    Args:
        flow: 5d tensor, [batch, height, width, depth, channel(translation)]
    """
    loss = torch.mean(torch.mean(torch.pow(flow[:, 1:, :, :, :] - flow[:, :-1, :, :, :], 2)) +
                      torch.mean(torch.pow(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :], 2)) +
                      torch.mean(torch.pow(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :], 2)))

    return loss

def get_losses():
    losses = {}
    losses['vae'] = vae_loss
    losses['dice'] = dice_loss
    losses['ncc'] = ncc_loss
    losses['patch_ncc'] = patch_ncc_loss
    losses['rc'] = residual_complexity_loss
    losses['l1_smooth'] = l1_smooth3d
    losses['l2_smooth'] = l2_smooth3d

    return losses
