# #!/usr/bin/env python
"""
preprocess.py: This contains the data-pre-processing routines.
"""
from __future__ import print_function

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from week8.modular import cfg

sys.path.append('./')
args = cfg.parser.parse_args(args=[])
file_path = args.data


def get_dataset_mean_std():
    """
    Get the CIFAR10 dataset mean and std to be used as tuples
    @ transforms.Normalize
    """
    # load the training data
    cifar10_train = datasets.CIFAR10('./data', train=True, download=True)
    # use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
    x = np.concatenate([np.asarray(cifar10_train[i][0]) for i in range(len(cifar10_train))])
    # print(x)
    print(x.shape)
    # calculate the mean and std
    train_mean = np.mean(x, axis=(0, 1)) / 255
    train_std = np.std(x, axis=(0, 1)) / 255
    # the the mean and std
    print(train_mean, train_std)
    return train_mean, train_std


def preprocess_data(mean_tuple, std_tuple):
    """
    Used for pre-processing the data
    """
    # Train Phase transformations
    train_transforms = transforms.Compose([
        #  transforms.Resize((28, 28)),
        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        # transforms.RandomRotation((-10.0, 10.0), fill=(1,)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_tuple, std_tuple),
        # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
        # Note the difference between (0.1307) and (0.1307,)
        
    ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
        #  transforms.Resize((28, 28)),
        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean_tuple, std_tuple)
    ])
    train_cifar10 = datasets.CIFAR10(file_path, train=True, download=True, transform=train_transforms)
    test_cifar10 = datasets.CIFAR10(file_path, train=False, download=True, transform=test_transforms)
    print("CUDA Available?", args.cuda)

    # For reproducibility
    torch.manual_seed(args.SEED)

    if args.cuda:
        torch.cuda.manual_seed(args.SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=args.batch_size, num_workers=4,
                           pin_memory=True) if args.cuda else dict(shuffle=True,
                                                                   batch_size=args.batch_size)
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_cifar10, **dataloader_args)
    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_cifar10, **dataloader_args)
    return train_cifar10, test_cifar10, train_loader, test_loader


def get_data_stats(train_cifar10, test_cifar10, train_loader):
    """
    Get the data-statistics
    """
    # We'd need to convert it into Numpy! Remember above we have converted it into tensors already
    train_data = torch.from_numpy(train_cifar10.data)
    print('[Stats from Train Data]')
    print(' - Numpy Shape:', torch.from_numpy(train_cifar10.data).cpu().numpy().shape)
    print(' - Tensor Shape:', torch.from_numpy(train_cifar10.data).size())
    print(' - min:', torch.min(torch.from_numpy(train_cifar10.data)))
    print(' - max:', torch.max(torch.from_numpy(train_cifar10.data)))
    test_data = torch.from_numpy(test_cifar10.data)
    print('[Stats from Test Data]')
    print(' - Numpy Shape:', torch.from_numpy(test_cifar10.data).cpu().numpy().shape)
    print(' - Tensor Shape:', torch.from_numpy(test_cifar10.data).size())
    print(' - min:', torch.min(torch.from_numpy(test_cifar10.data)))
    print(' - max:', torch.max(torch.from_numpy(test_cifar10.data)))

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    print(images.shape)
    print(labels.shape)
    save_dir = os.path.join(os.getcwd(), args.data)
    file_name = 'data_stats'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, '{}.png'.format(file_name))
    print("Saving plot for a sample to ascertain RF required for edges & gradient {}".format(filepath))
    img_number = np.random.randint(images.shape[0])
    plt.figure().suptitle('{} '.format(train_loader.dataset.classes[labels[img_number]]), fontsize=20)
    plt.imshow(images.numpy().squeeze()[img_number, ::].transpose((1, 2, 0)), interpolation='nearest')
    plt.savefig(filepath)
    # plt.show()
