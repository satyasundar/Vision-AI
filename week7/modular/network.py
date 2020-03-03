#!/usr/bin/env python
"""
network.py: This contains the model definition.
It needs to be further abstracted out, to be used with
more user-args (TBD)
"""
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import sys
import cfg

sys.path.append('./')

args = cfg.parser.parse_args()
dropout_value = args.dropout


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels,
                      bias=bias),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias),
        )

    def forward(self, x):
        x = self.convblock(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # C1 Block
        self.convblock1 = nn.Sequential(
            SeparableConv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:32x32x3, output:32x32x64, RF:3x3
        self.convblock2 = nn.Sequential(
            SeparableConv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:32x32x64, output:32x32x128, RF:7x7

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)  # input:32x32x128, output:16x16x128, RF:8x8
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:16x16x128, output:16x16x64, RF:8x8

        # C2 Block
        self.convblock4 = nn.Sequential(
            SeparableConv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:16x16x64, output:16x16x128, RF:10x10

        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2)  # input:16x16x128, output:8x8x128, RF:14x14
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:8x8x128, output:8x8x64, RF:14x14

        # C3 Block
        self.convblock6 = nn.Sequential(
            SeparableConv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:8x8x64, output:8x8x128, RF:22x22

        # TRANSITION BLOCK 3
        self.pool3 = nn.MaxPool2d(2, 2)  # input:8x8x128, output:4x4x128, RF:26x26
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:4x4x128, output:4x4x64, RF:26x26

        # C4 Block
        self.convblock8 = nn.Sequential(
            SeparableConv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:4x4x64, output:4x4x64, RF:42x42

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )  # input:4x4x64, output:1x1x64, RF:66x66

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )  # input:1x1x64, output:1x1x10,

    def forward(self, x):
        # C1 Block
        x = self.convblock1(x)
        x = self.convblock2(x)
        # TRANSITION BLOCK 1
        x = self.pool1(x)
        x = self.convblock3(x)
        # C2 Block
        x = self.convblock4(x)
        # TRANSITION BLOCK 2
        x = self.pool2(x)
        x = self.convblock5(x)
        # C3 Block
        x = self.convblock6(x)
        # TRANSITION BLOCK 3
        x = self.pool3(x)
        x = self.convblock7(x)
        # C4 Block
        x = self.convblock8(x)
        # OUTPUT BLOCK
        x = self.gap(x)
        x = self.convblock9(x)
        # Reshape
        x = x.view(-1, 10)
        # the torch.nn.CrossEntropyLoss, criterion combines
        # nn.LogSoftmax() and nn.NLLLoss() in one single class.
        return F.log_softmax(x, dim=-1)