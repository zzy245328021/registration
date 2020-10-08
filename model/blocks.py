#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/3/28 15:46
# @Author  : Eric Ching
from torch import nn
import torch
from torch.autograd.function import once_differentiable
from torch.autograd import Function

class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func

    def forward(self, x): return self.func(x)

def Flatten():
    "Flattens `x` to a single dimension, often used at the end of a model."
    return Lambda(lambda x: x.view((x.size(0), -1)))


class AdaptiveConcatPool3d(nn.Module):
    "Layer that concats `AdaptiveAvgPool3d` and `AdaptiveMaxPool3d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or (1, 1, 1)
        self.ap, self.mp = nn.AdaptiveAvgPool3d(sz), nn.AdaptiveMaxPool3d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

def affine_grid3d(theta, size):
    """gennerate affine transform grid
    Args:
        theta: tensor, [batch, 3, 4]
        size: torch.Size, the target output image size [N, C, D, H, W]
    """
    return AffineGridGenerator.apply(theta, size)


class AffineGridGenerator(Function):
    @staticmethod
    def forward(ctx, theta, size):
        assert type(size) == torch.Size

        ctx.size = size
        ctx.is_cuda = theta.is_cuda

        if len(size) == 5:
            N, C, D, H, W = size
            base_grid = theta.new(N, D, H, W, 4)
            base_grid[:, :, :, :, 0] = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
            base_grid[:, :, :, :, 1] = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1]))\
                .unsqueeze(-1)
            base_grid[:, :, :, :, 2] = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1]))\
                .unsqueeze(-1).unsqueeze(-1)
            base_grid[:, :, :, :, 3] = 1
            grid = torch.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2))
            grid = grid.view(N, D, H, W, 3)
        else:
            raise RuntimeError("AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.")

        ctx.base_grid = base_grid

        return grid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grid):
        assert ctx.is_cuda == grad_grid.is_cuda
        base_grid = ctx.base_grid

        if len(ctx.size) == 5:
            N, C, D, H, W = ctx.size
            assert grad_grid.size() == torch.Size([N, D, H, W, 3])
            grad_theta = torch.bmm(
                base_grid.view(N, D * H * W, 4).transpose(1, 2),
                grad_grid.view(N, D * H * W, 3))
        else:
            assert False

        grad_theta = grad_theta.transpose(1, 2)
        return grad_theta, None