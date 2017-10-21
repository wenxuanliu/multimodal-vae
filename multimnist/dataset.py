"""
This script generates a dataset similar to the Multi-MNIST dataset
described in [1].

[1] Eslami, SM Ali, et al. "Attend, infer, repeat: Fast scene
understanding with generative models." Advances in Neural Information
Processing Systems. 2016.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import torch
import torchvision.datasets as dset
from scipy.misc import imresize


def sample_one(canvas_size, mnist):
    i = np.random.randint(mnist['digits'].shape[0])
    digit = mnist['digits'][i]
    label = mnist['labels'][i]
    scale = 0.1 * np.random.randn() + 1.3
    resized = imresize(digit, 1. / scale)
    w = resized.shape[0]
    assert w == resized.shape[1]
    padding = canvas_size - w
    pad_l = np.random.randint(0, padding)
    pad_r = np.random.randint(0, padding)
    pad_width = ((pad_l, padding - pad_l), (pad_r, padding - pad_r))
    positioned = np.pad(resized, pad_width, 'constant', constant_values=0)
    return positioned, label


def sample_multi(num_digits, canvas_size, mnist):
    canvas = np.zeros((canvas_size, canvas_size))
    labels = []
    for _ in range(num_digits):
        positioned_digit, label = sample_one(canvas_size, mnist)
        canvas += positioned_digit
        labels.append(label)
    
    # Crude check for overlapping digits.
    if np.max(canvas) > 255:
        return sample_multi(num_digits, canvas_size, mnist)
    else:
        return canvas, labels


def mk_dataset(n, mnist, min_digits, max_digits, canvas_size):
    x = []
    y = []
    for _ in range(n):
        num_digits = np.random.randint(min_digits, max_digits + 1)
        canvas, labels = sample_multi(num_digits, canvas_size, mnist)
        x.append(canvas)
        y.append(labels)
    return np.array(x, dtype=np.uint8), y


def load_mnist():
    train_loader = torch.utils.data.DataLoader(
        dset.MNIST(root='./data', train=True, download=True))

    test_loader = torch.utils.data.DataLoader(
        dset.MNIST(root='./data', train=False, download=True))
    
    train_data = {
        'digits': train_loader.dataset.train_data.numpy(),
        'labels': train_loader.dataset.train_labels
    }

    test_data = {
        'digits': test_loader.dataset.test_data.numpy(),
        'labels': test_loader.dataset.test_labels
    }

    return train_data, test_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_digits', type=int, default=0,
                        help='minimum number of digits on a single example')
    parser.add_argument('--max_digits', type=int, default=2,
                        help='maximum number of digits on a single example')
    args = parser.parse_args()
    # Generate the training set and dump it to disk. (Note, this will
    # always generate the same data, else error out.)
    train_outfile = './data/multi_mnist_train_uint8.npz'
    test_outfile = './data/multi_mnist_test_uint8.npz'

    np.random.seed(681307)
    train_mnist, test_mnist = load_mnist()
    train_x, train_y = mk_dataset(60000, train_mnist, args.min_digits, 
                                  args.max_digits, 50)
    test_x, test_y = mk_dataset(60000, test_mnist, args.min_digits, 
                                args.max_digits, 50)
    with open(train_outfile, 'wb') as f:
        np.savez_compressed(f, x=train_x, y=train_y)

    with open(test_outfile, 'wb') as f:
        np.savez_compressed(f, x=test_x, y=test_y)
