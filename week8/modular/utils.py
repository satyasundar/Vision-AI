#!/usr/bin/env python
"""
utils.py: This contains misc-utility code used across.
"""
from __future__ import print_function

import json
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

from week8.modular import cfg

sys.path.append('./')
args = cfg.parser.parse_args(args=[])
file_path = args.data
IPYNB_ENV = True #By default ipynb notebook env


def plot_train_samples(train_loader):
    """
    Plot dataset class samples
    """
    num_classes = len(np.unique(train_loader.dataset.targets))
    save_dir = os.path.join(os.getcwd(), args.data)
    file_name = 'plot_class_samples'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, '{}.png'.format(file_name))
    if not IPYNB_ENV:
        print("Saving plot {} class samples to {}".format(num_classes, filepath))
    fig = plt.figure(figsize=(8, 3))
    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(np.array(train_loader.dataset.targets)[:] == i)[0]
        features_idx = train_loader.dataset.data[idx, ::]
        img_num = np.random.randint(features_idx.shape[0])
        im = features_idx[img_num]
        ax.set_title(train_loader.dataset.classes[i])
        plt.imshow(im)
        if not IPYNB_ENV:
            plt.savefig(filepath)
    if IPYNB_ENV:
        plt.show()


def l1_penalty(x):
    """
    L1 regularization adds an L1 penalty equal
    to the absolute value of the magnitude of coefficients
    """

    return torch.abs(x).sum()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the model to the path
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def model_pred(model, device, y_test, test_cifar10, batchsize=100):
    """
    Make inference on the test-data &
    print classification-report
    """
    start = 0
    stop = batchsize
    model.eval()
    dataldr_args = dict(shuffle=False, batch_size=batchsize, num_workers=4, pin_memory=True) if args.cuda else dict(
        shuffle=False, batch_size=batchsize)
    test_ldr = torch.utils.data.DataLoader(test_cifar10, **dataldr_args)
    y_pred = np.zeros((y_test.shape[0], 1))
    with torch.no_grad():
        for data, target in test_ldr:
            batch_nums = np.arange(start, stop)
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_pred[batch_nums] = output.argmax(dim=1, keepdim=True).cpu().numpy()
            start += batchsize
            stop += batchsize
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=test_cifar10.classes))
    return y_pred


def display_mislabelled(model, device, x_test, y_test, y_pred, test_cifar10, title_str):
    """
    Plot 3 groups of 10 mislabelled data class-samples.
    """
    save_dir = os.path.join(os.getcwd(), args.data)
    file_name = 'plot_mislabelled'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, '{}.png'.format(file_name))
    if not IPYNB_ENV:
        print("Saving plot for the mislabelled images to {}".format(filepath))
    fig = plt.figure(figsize=(55, 10))
    fig.suptitle(title_str, fontsize=24)
    idx1 = np.where(y_test[:] != y_pred)[0]
    for j in range(3):
        for i in range(len(test_cifar10.classes)):
            ax = fig.add_subplot(3, 10, j * 10 + i + 1, xticks=[], yticks=[])
            idx = np.where(y_test[:] == i)[0]
            intsct = np.intersect1d(idx1, idx)
            features_idx = x_test[intsct, ::]
            img_num = np.random.randint(features_idx.shape[0])
            im = features_idx[img_num]
            ax.set_title('Act:{} '.format(test_cifar10.classes[int(i)]) + ' Pred:{} '.format(
                test_cifar10.classes[int(y_pred[intsct[img_num]][0])]), fontsize=24)
            plt.imshow(im)
            if not IPYNB_ENV:
                plt.savefig(filepath)
    if IPYNB_ENV:
        plt.show()


def load_model(describe_model_nn, device, model_name):
    """
    load the best-accuracy model from the given name
    """
    save_dir = os.path.join(os.getcwd(), args.best_model_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    model = describe_model_nn  # describe_model_nn is for example: Net1()
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model


def save_acc_loss(test_losses, test_acc, test_loss_file_name, test_acc_file_name):
    """
    Save test-accuracies and test-losses during training.
    """
    import os
    import numpy as np
    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), file_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath_test_loss = os.path.join(save_dir, test_loss_file_name)
    filepath_test_acc = os.path.join(save_dir, test_acc_file_name)
    np.save(filepath_test_loss, test_losses)
    np.save(filepath_test_acc, test_acc)


def load_acc_loss(test_loss_file_name, test_acc_file_name):
    """
    Load the accuracy and loss data from files.
    """
    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), file_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath_test_loss = os.path.join(save_dir, test_loss_file_name)
    filepath_test_acc = os.path.join(save_dir, test_acc_file_name)
    loaded_test_losses = np.load(filepath_test_loss).tolist()
    loaded_test_acc = np.load(filepath_test_acc).tolist()
    return loaded_test_losses, loaded_test_acc


def plot_acc_loss():
    """
    Plot both accuracy and loss plots.
    """
    save_dir = os.path.join(os.getcwd(), args.data)
    file_name = 'plot_acc_loss'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, '{}.png'.format(file_name))
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(cfg.train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(cfg.train_acc[4000:])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(cfg.test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(cfg.test_acc)
    axs[1, 1].set_title("Test Accuracy")
    if not IPYNB_ENV:
        fig.savefig(filepath)
    else:
        fig.show()


def write(dic, path):
    with open(path, 'w+') as f:
        # write params to txt file
        f.write(json.dumps(dic))
