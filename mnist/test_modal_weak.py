from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import operator
import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms

from test import test_mnist
from train import load_checkpoint


if __name__ == "__main__":
    import os
    import argparse
    from glob import glob

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    parser = argparse.ArgumentParser()
    parser.add_argument('models_dir', type=str, help='path to output directory of weak.py')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    x1, x2, y = [], [], []

    for dir_path in glob(os.path.join(args.models_dir, '*')):
        weak_perc_m1 = float(os.path.basename(dir_path).split('_')[-3])
        weak_perc_m2 = float(os.path.basename(dir_path).split('_')[-1])
        loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, download=True,
                           transform=transforms.ToTensor()),
            batch_size=128, shuffle=True)
        vae = load_checkpoint(os.path.join(dir_path, 'model_best.pth.tar'), use_cuda=args.cuda)
        vae.eval()
        weak_acc = test_mnist(vae, loader, use_cuda=args.cuda, verbose=False)

        x1.append(weak_perc_m1)
        x2.append(weak_perc_m2)
        y.append(weak_acc)
        print('Got accuracies for %s.' % dir_path)

    assert len(set(x1)) == len(set(x2))
    percs = sorted(list(set(x1)))
    n_perc = len(percs)

    x1, x2 = np.array(x1), np.array(x2) 
    y = np.array(y)
    data = np.vstack((x1, x2, y)).T.tolist()
    data = np.array(sorted(data, key=operator.itemgetter(0, 1)))
    data = data[:, -1].reshape(n_perc, n_perc)

    def save_plot(data, savepath):
        column_labels = [str(i) for i in percs]
        row_labels = [str(i) for i in percs]
    
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(data)

        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                plt.text(x + 0.5, y + 0.5, '%.2f' % data[y, x],
                         horizontalalignment='center',
                         verticalalignment='center')

        # put the major ticks at the middle of each cell, notice "reverse" use of dimension
        ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
        ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
        ax.set_xticklabels(row_labels, minor=False)
        ax.set_yticklabels(column_labels, minor=False)
        
        plt.colorbar(heatmap)
        plt.xlabel('% Text Examples', fontsize=18)
        plt.ylabel('% Image Examples', fontsize=18)

        plt.tight_layout()
        plt.savefig(savepath)

    save_plot(data, './weak_modal_supervision.png')
