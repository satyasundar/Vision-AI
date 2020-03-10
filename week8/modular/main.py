#!/usr/bin/env python
"""
main.py: This is the main script to be run to either train or make inference.

example usage is as below:
python main.py train --SEED 2 --batch_size 64  --epochs 10 --lr 0.01 \
                     --dropout 0.05 --l1_weight 0.00002  --l2_weight_decay 0.000125 \
                     --L1 True --L2 False --data data --best_model_path saved_models \
                     --prefix data

python main.py test  --batch_size 64  --data data --best_model_path saved_models \
                     --best_model 'CIFAR10_model_epoch-39_L1-1_L2-0_val_acc-81.83.h5' \
                     --prefix data
"""
from __future__ import print_function

import os
import sys
import warnings

import numpy as np
import torch
import torch.optim as optim
from torchsummary import summary

from week8.modular import cfg
from week8.modular import network
from week8.modular import preprocess
from week8.modular import test
from week8.modular import train
from week8.modular import utils

sys.path.append('./')
args = cfg.parser.parse_args(args=[])
if args.cmd == None:
    args.cmd = 'test'


def main():
    print("The config used for this run are being saved @ {}".format(os.path.join(args.prefix, 'config_params.txt')))
    utils.write(vars(args), os.path.join(args.prefix, 'config_params.txt'))
    mean, std = preprocess.get_dataset_mean_std()
    train_cifar10, test_cifar10, train_loader, test_loader = preprocess.preprocess_data((mean[0], mean[1], mean[2]), (std[0], std[1], std[2]))
    preprocess.get_data_stats(train_cifar10, test_cifar10, train_loader)
    utils.plot_train_samples(train_loader)
    L1 = args.L1   
    L2 = args.L2   
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)
    model = network.ResNet18().to(device)
    summary(model, input_size=(3, 32, 32))
    if args.cmd == 'train':
        print("Model training starts on CIFAR10 dataset")
        # Enable L2-regularization with supplied value of weight decay, or keep it default-0
        if L2:
            weight_decay = args.l2_weight_decay
        else:
            weight_decay = 0

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)

        EPOCHS = args.epochs
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch + 1)
            train.train(model, device, train_loader, optimizer, epoch)
            test.test(model, device, test_loader, optimizer, epoch)
        utils.plot_acc_loss()
    elif args.cmd == 'test':
        print("Model inference starts on CIFAR10 dataset")
        model_name = args.best_model
        print("Loaded the best model: {} from last training session".format(model_name))
        model = utils.load_model(network.ResNet18(), device, model_name=model_name)
        y_test = np.array(test_cifar10.targets)
        print("The confusion-matrix and classification-report for this model are:")
        y_pred = utils.model_pred(model, device, y_test, test_cifar10)
        x_test = test_cifar10.data
        utils.display_mislabelled(model, device, x_test, y_test.reshape(-1, 1), y_pred, test_cifar10,
                            title_str='Predicted Vs Actual With L1')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    main()
