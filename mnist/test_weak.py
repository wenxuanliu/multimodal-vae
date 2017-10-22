from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

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

    parser = argparse.ArgumentParser()
    parser.add_argument('models_dir', type=str, help='path to output directory of weak.py')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    x, y = [], []

    for dir_path in glob(os.path.join(args.models_dir, '*')):
        weak_perc = float(os.path.basename(dir_path).split('_')[-1])
        loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, download=True,
                           transform=transforms.ToTensor()),
            batch_size=128, shuffle=True)
        vae = load_checkpoint(os.path.join(dir_path, 'model_best.pth.tar'), use_cuda=args.cuda)
        vae.eval()
        weak_acc = test_mnist(vae, loader, use_cuda=args.cuda, verbose=False)

        x.append(weak_perc)
        y.append(weak_acc)
        print('Got accuracies for %s.' % dir_path)


    plt.figure()
    plt.plot(x, y, '-o')
    plt.xlabel('% Supervision', fontsize=18)
    plt.ylabel('MNIST Accuracy', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.tight_layout()
    plt.savefig('./weak_supervision.png')
