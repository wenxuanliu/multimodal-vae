"""Call datasets.MNIST once to download the processed PyTorch objects"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch.utils.data as data
from torchvision import datasets


if __name__ == "__main__":
    if not os.path.isdir('./data'):
        os.makedirs('./data')
        print('Created ./data.')

    train_loader = data.DataLoader(
        datasets.MNIST('./data', train=True, download=True),
        batch_size=64, shuffle=True)
    test_loader = data.DataLoader(
        datasets.MNIST('./data', train=False, download=True),
        batch_size=64, shuffle=True)

    print('Downloaded MNIST to ./data')
