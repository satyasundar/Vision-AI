#!/usr/bin/env python
"""
network.py: This contains the ResNet18 model definition.
as extracted from: 
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""
from __future__ import print_function

import sys

import torch.nn as nn
import torch.nn.functional as F

from week8.modular import cfg

sys.path.append('./')

args = cfg.parser.parse_args(args=[])
dropout_value = args.dropout

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out) 
        # the torch.nn.CrossEntropyLoss, criterion combines
        # nn.LogSoftmax() and nn.NLLLoss() in one single class.
        return F.log_softmax(out)


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())   

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
