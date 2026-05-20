# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Coordinate Attention module."""

import torch
import torch.nn as nn

from .conv import Conv


class CA(nn.Module):
    """Coordinate Attention block.

    This module keeps positional information by pooling separately along height
    and width, then uses direction-aware attention maps to recalibrate features.
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
