from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
from torchvision import transforms

import datasets
from test import test_multimnist
from train import load_checkpoint
from utils import charlist_tensor


if __name__ == "__main__":
    import os
    import argparse
    from glob import glob

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('models_dir', type=str, help='path to output directory of weak.py')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    x, y1, y2 = [], [], []

    for dir_path in glob(os.path.join(args.models_dir, '*')):
        weak_perc = float(os.path.basename(dir_path).split('_')[-1])
        loader = torch.utils.data.DataLoader(
            datasets.MultiMNIST('./data', train=False, download=True,
                                transform=transforms.ToTensor(),
                                target_transform=charlist_tensor),
            batch_size=128, shuffle=True)
        vae = load_checkpoint(os.path.join(dir_path, 'model_best.pth.tar'), use_cuda=args.cuda)
        vae.eval()
        weak_char_acc, weak_len_acc = test_multimnist(vae, loader, 
                                                      use_cuda=args.cuda, verbose=False)

        x.append(weak_perc)
        y1.append(weak_char_acc)
        y2.append(weak_len_acc)
        print('Got accuracies for %s.' % dir_path)

    x, y1, y2 = np.array(x), np.array(y1), np.array(y2)
    ix = np.argsort(x)
    x, y1, y2 = x[ix], y1[ix], y2[ix]

    plt.figure()
    plt.plot(x, y1, '-o', label='character accuracy')
    plt.plot(x, y2, '-o', label='length accuracy')
    plt.xlabel('% Supervision', fontsize=18)
    plt.ylabel('MultiMNIST Accuracy', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./weak_supervision.png')
