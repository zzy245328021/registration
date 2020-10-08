#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/4/2 19:40
# @Author  : Eric Ching
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .blocks import affine_grid3d

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x

class DilatedBlock(nn.Module):
    """ASPP block"""
    def __init__(self, in_channels, n_groups=8, mode='cascade'):
        super(DilatedBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.GroupNorm(n_groups, in_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=1))
        self.conv2 = nn.Sequential(nn.GroupNorm(n_groups, in_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=2, dilation=2))

        self.conv3 = nn.Sequential(nn.GroupNorm(n_groups, in_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=4, dilation=4))

        self.conv4 = nn.Sequential(nn.GroupNorm(n_groups, in_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=8,
                                             dilation=8))
        self.mode = mode

    def forward(self, x):
        if self.mode == 'parallel':
            c1 = self.conv1(x)
            c2 = self.conv2(x)
            c3 = self.conv3(x)
            c4 = self.conv4(x)
        else:   # cascade

            c1 = self.conv1(x)

            c2 = self.conv2(c1)

            c3 = self.conv3(c2)

            c4 = self.conv4(c3)

        c = x + c1 + c2 + c3 + c4

        return c

class UNetCore3D(nn.Module):
    """3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    """
    def __init__(self, out_channels=3, init_channels=32, use_dilate=False):
        super(UNetCore3D, self).__init__()
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.use_dilate = use_dilate
        self.make_encoder()
        self.make_decoder()

    def make_encoder(self):
        init_channels = self.init_channels
        self.conv1a = nn.Conv3d(2, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.ds1 = nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2),
                             padding=(1, 1, 1))  # down sampling

        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.ds2 = nn.Conv3d(init_channels * 2, init_channels * 2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv3a = BasicBlock(init_channels * 2, init_channels * 2)
        self.ds3 = nn.Conv3d(init_channels * 2, init_channels * 2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        if self.use_dilate:
            self.conv4a = DilatedBlock(init_channels * 2)
        else:
            self.conv4a = BasicBlock(init_channels * 2, init_channels * 2)

    def make_decoder(self):
        init_channels = self.init_channels
        self.up4conva = nn.Conv3d(init_channels * 2, init_channels * 2, (1, 1, 1))
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up3conva = nn.Conv3d(init_channels * 2, init_channels * 2, (1, 1, 1))
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, (1, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2)

        self.up1conv = nn.Conv3d(init_channels, self.out_channels, (3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        c1 = self.conv1a(x)
        c1 = F.relu(c1, inplace=True)

        c1d = self.ds1(c1)
        c2 = self.conv2a(c1d)

        c2d = self.ds2(c2)

        c3 = self.conv3a(c2d)
        c3d = self.ds3(c3)

        c4d = self.conv4a(c3d)
        u4 = self.up4conva(c4d)

        u4 = self.up4(u4)


        u4 = u4 + c3

        u4 = self.up4convb(u4)


        u3 = self.up3conva(u4)


        u3 = self.up3(u3)

        u3 = u3 + c2


        u3 = self.up3convb(u3)


        u2 = self.up2conva(u3)
        u2 = self.up2(u2)

        u2 = u2 + c1

        output = self.up1conv(u2)



        return output
    


class WNet3D(nn.Module):
    def __init__(self, init_channels=16, use_dialte=False):
        self.init_channels = init_channels
        super(WNet3D, self).__init__()
        self.unet = UNetCore3D(init_channels=self.init_channels, out_channels=16, use_dilate=use_dialte)
        self.glob_pool = nn.AdaptiveAvgPool3d(1)
        self.flow_conv = nn.Sequential(
            nn.Conv3d(16, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(8, 3, kernel_size=(1, 1, 1)),


                                       )
        self.affine_dense = nn.Sequential(nn.Linear(32, 12),)
        self.init_weights()

    def creat_regular_grid(self, batch_shape):
        grid = torch.meshgrid(torch.linspace(-1, 1, batch_shape[2]).cuda(),
                              torch.linspace(-1, 1, batch_shape[3]).cuda(),
                              torch.linspace(-1, 1, batch_shape[4]).cuda()) # depth, height, width
        grid = torch.stack([grid[2], grid[1], grid[0]], dim=-1)
        batch_grid = grid.unsqueeze(0).repeat([batch_shape[0], 1, 1, 1, 1])
        
        return batch_grid
    
    def init_weights(self):
        nn.init.zeros_(self.flow_conv[2].weight)
        nn.init.zeros_(self.flow_conv[2].bias)
        nn.init.zeros_(self.affine_dense[0].weight)
        self.affine_dense[0].bias.data.copy_(torch.eye(3, 4, dtype=torch.float).view(-1))

    def forward_unet_down(self, x):
        c1 = self.unet.conv1a(x)
        c1 = F.relu(c1, inplace=True)
        c1d = self.unet.ds1(c1)

        c2 = self.unet.conv2a(c1d)
        c2d = self.unet.ds2(c2)

        c3 = self.unet.conv3a(c2d)
        c3d = self.unet.ds3(c3)

        c4d = self.unet.conv4a(c3d)

        affine_coef = self.glob_pool(c4d)
        affine_coef = affine_coef.view([-1, 32])
        affine_coef = self.affine_dense(affine_coef)
        affine_coef = affine_coef.view([-1, 3, 4])

        return affine_coef
    
    def forward(self, fix, move):
        x = torch.cat((fix, move), dim=1)
        affine_coef = self.forward_unet_down(x)
        affine_grid = affine_grid3d(affine_coef, fix.size())
        affine = F.grid_sample(move, affine_grid)
        x2 = torch.cat((fix, affine), dim=1)
        flow = self.unet(x2)

        flow = self.flow_conv(flow)


        flow = flow.permute(0, 2, 3, 4, 1)

        if not hasattr(self, 'batch_regular_grid'):
            self.batch_regular_grid = self.creat_regular_grid(x.size())

        flow_grid = flow + self.batch_regular_grid

        #flow_grid = F.hardtanh(flow_grid,-1, 1 )
        wrap = F.grid_sample(affine, flow_grid)  #[1, 1, 192, 192, 192])


        return wrap, flow_grid, flow, affine,affine_grid