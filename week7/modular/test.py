#!/usr/bin/env python
"""
test.py: This contains the model-inference code.
"""
from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import sys
import cfg
from utils import save_checkpoint

sys.path.append('./')
args = cfg.parser.parse_args()


def test(model, device, test_loader, optimizer, epoch):
    """
    main test code
    """
    # global current_best_acc, last_best_acc
    model.eval()
    test_loss = 0
    correct = 0
    acc1 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    cfg.test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc1 = 100. * correct / len(test_loader.dataset)
    is_best = acc1 > cfg.current_best_acc
    cfg.last_best_acc = cfg.current_best_acc
    cfg.current_best_acc = max(acc1, cfg.current_best_acc)
    # Prepare model model saving directory.
    if is_best:
        save_dir = os.path.join(os.getcwd(), args.best_model_path)
        model_name = 'CIFAR10_model_epoch-{}_L1-{}_L2-{}_val_acc-{}.h5'.format(epoch + 1, int(args.L1), int(args.L2),
                                                                               acc1)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        print(
            "validation-accuracy improved from {} to {}, saving model to {}".format(cfg.last_best_acc,
                                                                                    cfg.current_best_acc, filepath))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': cfg.current_best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=filepath)
    cfg.test_acc.append(100. * correct / len(test_loader.dataset))
