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
import random
import numpy as np
from PIL import Image
from random import shuffle
from scipy.misc import imresize

import torch
import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset


class MultiMNIST(Dataset):
    """images with 0 to 4 digits of non-overlapping MNIST numbers."""
    processed_folder = 'multimnist'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        if self._check_exists():
            return
        make_dataset(self.root, self.processed_folder, 
                     self.training_file, self.test_file)


def scramble(word):
    word = list(word)
    shuffle(word)
    return ''.join(word)


def sample_one(canvas_size, mnist, resize=True, translate=True):
    i = np.random.randint(mnist['digits'].shape[0])
    digit = mnist['digits'][i]
    label = mnist['labels'][i]
    if resize:  # resize only if user specified
        scale = 0.1 * np.random.randn() + 1.3
        resized = imresize(digit, 1. / scale)
    else:
        resized = digit
    w = resized.shape[0]
    assert w == resized.shape[1]
    padding = canvas_size - w
    if translate:  # translate only if user specified
        pad_l = np.random.randint(0, padding)
        pad_r = np.random.randint(0, padding)
        pad_width = ((pad_l, padding - pad_l), (pad_r, padding - pad_r))
        positioned = np.pad(resized, pad_width, 'constant', constant_values=0)
    else:
        pad_l = padding // 2
        pad_r = padding // 2
        pad_width = ((pad_l, padding - pad_l), (pad_r, padding - pad_r))
        positioned = np.pad(resized, pad_width, 'constant', constant_values=0)
    return positioned, label


def sample_multi(num_digits, canvas_size, mnist, resize=True, translate=True):
    canvas = np.zeros((canvas_size, canvas_size))
    labels = []
    for _ in range(num_digits):
        positioned_digit, label = sample_one(canvas_size, mnist, resize=resize,
                                             translate=translate)
        canvas += positioned_digit
        labels.append(label)
    
    # Crude check for overlapping digits.
    if np.max(canvas) > 255:
        return sample_multi(num_digits, canvas_size, mnist, 
                            resize=resize, translate=translate)
    else:
        return canvas, labels


def mk_dataset(n, mnist, min_digits, max_digits, canvas_size, 
               resize=True, translate=True):
    x = []
    y = []
    for _ in range(n):
        num_digits = np.random.randint(min_digits, max_digits + 1)
        canvas, labels = sample_multi(num_digits, canvas_size, mnist,
                                      resize=resize, translate=translate)
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


def make_dataset(root, folder, training_file, test_file, min_digits=0, max_digits=2,
                 resize=True, translate=True):
    if not os.path.isdir(os.path.join(root, folder)):
        os.makedirs(os.path.join(root, folder))

    np.random.seed(681307)
    train_mnist, test_mnist = load_mnist()
    train_x, train_y = mk_dataset(60000, train_mnist, min_digits, max_digits, 50,
                                  resize=resize, translate=translate)
    test_x, test_y = mk_dataset(10000, test_mnist, min_digits, max_digits, 50,
                                resize=resize, translate=translate)
    
    train_x = torch.from_numpy(train_x).byte()
    test_x = torch.from_numpy(test_x).byte()

    training_set = (train_x, train_y)
    test_set = (test_x, test_y)

    with open(os.path.join(root, folder, training_file), 'wb') as f:
        torch.save(training_set, f)

    with open(os.path.join(root, folder, test_file), 'wb') as f:
        torch.save(test_set, f)


def sample_one_fixed(canvas_size, mnist, pad_l, pad_r, scale=1.3):
    i = np.random.randint(mnist['digits'].shape[0])
    digit = mnist['digits'][i]
    label = mnist['labels'][i]
    resized = imresize(digit, 1. / scale)
    w = resized.shape[0]
    assert w == resized.shape[1]
    padding = canvas_size - w
    pad_width = ((pad_l, padding - pad_l), (pad_r, padding - pad_r))
    positioned = np.pad(resized, pad_width, 'constant', constant_values=0)
    return positioned, label


def sample_multi_fixed(num_digits, canvas_size, mnist, scramble=False):
    canvas = np.zeros((canvas_size, canvas_size))
    labels = []
    pads = [(4, 4), (4, 23), (23, 4), (23, 23)]
    for i in range(num_digits):
        positioned_digit, label = sample_one_fixed(canvas_size, mnist, pads[i][0], pads[i][1])
        canvas += positioned_digit
        labels.append(label)
    
    if scramble:
        random.shuffle(labels)

    # Crude check for overlapping digits.
    if np.max(canvas) > 255:
        return sample_multi_fixed(num_digits, canvas_size, mnist, scramble=scramble)
    else:
        return canvas, labels


def mk_dataset_fixed(n, mnist, min_digits, max_digits, canvas_size, scramble=False):
    x = []
    y = []
    for _ in range(n):
        num_digits = np.random.randint(min_digits, max_digits + 1)
        canvas, labels = sample_multi_fixed(num_digits, canvas_size, mnist, scramble=scramble)
        x.append(canvas)
        y.append(labels)
    return np.array(x, dtype=np.uint8), y


def make_dataset_fixed(root, folder, training_file, test_file, 
                       min_digits=0, max_digits=3, scramble=False):
    if not os.path.isdir(os.path.join(root, folder)):
        os.makedirs(os.path.join(root, folder))

    np.random.seed(681307)
    train_mnist, test_mnist = load_mnist()
    train_x, train_y = mk_dataset_fixed(60000, train_mnist, min_digits, max_digits, 
                                        50, scramble=scramble)
    test_x, test_y = mk_dataset_fixed(10000, test_mnist, min_digits, max_digits, 
                                      50, scramble=scramble)
    
    train_x = torch.from_numpy(train_x).byte()
    test_x = torch.from_numpy(test_x).byte()

    training_set = (train_x, train_y)
    test_set = (test_x, test_y)

    with open(os.path.join(root, folder, training_file), 'wb') as f:
        torch.save(training_set, f)

    with open(os.path.join(root, folder, test_file), 'wb') as f:
        torch.save(test_set, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_digits', type=int, default=0, 
                        help='minimum number of digits to add to an image')
    parser.add_argument('--max_digits', type=int, default=2,
                        help='maximum number of digits to add to an image')
    parser.add_argument('--no_resize', action='store_true', default=False,
                        help='if True, fix the image to be MNIST size')
    parser.add_argument('--no_translate', action='store_true', default=False,
                        help='if True, fix the image to be in the center')
    parser.add_argument('--fixed', action='store_true', default=False,
                        help='If True, ignore resize/translate options and generate')
    parser.add_argument('--scramble', action='store_true', default=False,
                        help='If True, scramble labels and generate. Only does something if fixed is True.')
    args = parser.parse_args()
    args.resize = not args.no_resize
    args.translate = not args.no_translate
    if args.scramble and not args.fixed:
        raise Exception('Must have --fixed if --scramble is supplied.')

    # Generate the training set and dump it to disk. (Note, this will
    # always generate the same data, else error out.)
    if args.fixed:
        make_dataset_fixed('./data', 'multimnist', 'training.pt', 'test.pt',
                           min_digits=args.min_digits, max_digits=args.max_digits,
                           scramble=args.scramble)
    else:  # if not fixed, then make classific MultiMNIST dataset
        make_dataset('./data', 'multimnist', 'training.pt', 'test.pt',
                     min_digits=args.min_digits, max_digits=args.max_digits,
                     resize=args.resize, translate=args.translate)
