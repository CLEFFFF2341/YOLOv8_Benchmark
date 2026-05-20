# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Coordinate attention and lightweight BiFPN fusion modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class CA(nn.Module):
    """Coordinate Attention block.

    This module keeps positional information by pooling separately along height
    and width, then uses the resulting direction-aware attention maps to
    recalibrate the input feature.
    """

    def __init__(self, c1, c2=None, reduction=32):
        super().__init__()
        c2 = c1 if c2 is None else c2
        mip = max(8, c2 // reduction)

        self.align = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.reduce = nn.Sequential(
            nn.Conv2d(c2, mip, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mip),
            nn.SiLU(inplace=True),
        )
        self.conv_h = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.align(x)
        _, _, h, w = x.shape

        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = self.reduce(torch.cat((x_h, x_w), dim=2))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


class BiFPN_Add2(nn.Module):
    """Weighted two-input BiFPN fusion with channel alignment."""

    def __init__(self, c1, c2=None, eps=1e-4):
        super().__init__()
        if not isinstance(c1, (list, tuple)) or len(c1) != 2:
            raise ValueError("BiFPN_Add2 expects exactly two input channel values.")

        c2 = c1[0] if c2 is None else c2
        self.eps = eps
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.align0 = Conv(c1[0], c2, 1, 1) if c1[0] != c2 else nn.Identity()
        self.align1 = Conv(c1[1], c2, 1, 1) if c1[1] != c2 else nn.Identity()
        self.out = Conv(c2, c2, 3, 1)

    def forward(self, x):
        x0, x1 = x
        x0 = self.align0(x0)
        x1 = self.align1(x1)
        if x1.shape[-2:] != x0.shape[-2:]:
            x1 = F.interpolate(x1, size=x0.shape[-2:], mode="nearest")

        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        return self.out(w[0] * x0 + w[1] * x1)
