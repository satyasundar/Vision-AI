#!/usr/bin/env python
"""
cfg.py: This contains the hooks, for providing different default or user-supplied parameters.
And also the global variables used across different packages.
"""
import sys
import argparse
import torch
import matplotlib.pyplot as plt

sys.path.append('./')
IPYNB_ENV = True #By default ipynb notebook env
# The AGG backend(for matplotlib) is for writing to "file", not for rendering in a "window".
if not IPYNB_ENV:
  plt.switch_backend('agg')
parser = argparse.ArgumentParser(description='Training and Validation on CIFAR10 Dataset')
parser.add_argument('--cmd', default='test', choices=['train', 'test'])
parser.add_argument('--SEED', '-S', default=1, type=int, help='Random Seed')
parser.add_argument('--batch_size', '-b', default=64, type=int, help='batch size')
parser.add_argument('--epochs', '-e', default=40, type=int, help='training epochs')
parser.add_argument('--lr', default=0.0006, type=float, help='learning rate')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='use gpu or not')
parser.add_argument('--dropout', '-d', default=0.05, type=float, help='dropout percentage for all layers')
parser.add_argument('--l1_weight', default=0.000025, type=float, help='L1-penalty value')
parser.add_argument('--l2_weight_decay', default=0.0002125, type=float, help='L2-penalty/weight_decay value')
parser.add_argument('--L1', default=True, type=bool, help='L1-penalty to be used or not?')
parser.add_argument('--L2', default=False, type=bool, help='L2-penalty/weight_decay to be used or not?')
parser.add_argument('--data', '-s', default='./data/', help='path to save train/test data')
parser.add_argument('--best_model_path', default='./saved_models/', help='best model saved path')
parser.add_argument('--prefix', '-p', default='data', type=str, help='folder prefix')
parser.add_argument('--best_model', '-m', default='CIFAR10_model_epoch-39_L1-1_L2-0_val_acc-81.83.h5',
                    type=str, help='name of best-accuracy model saved')

# Following are the set of global variable being used across.
current_best_acc = 0
last_best_acc = 0
train_losses = []
test_losses = []
train_acc = []
test_acc = []
