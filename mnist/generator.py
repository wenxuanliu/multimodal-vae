from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from PIL import Image

import torch
from torch.utils.data.dataset import Dataset


class ShuffleMNIST(Dataset):
    """Dataset wrapping around MNIST but includes an equal 
    amount of random negatives. This requires using datasets.MNIST
    to download the directory beforehand.

    :param pt_path: path to PyTorch file downloaded 
                    by datasets.MNIST
    :param transform: any transformations to data (default: None)
    """
    def __init__(self, pt_path, transform=None, target_transform=None):
        images, targets = torch.load(pt_path)
        labels = torch.ones(len(images))
        # load it again and use this copy to randomly shuffle
        shuffle = torch.randperm(len(images))
        _images, _targets = torch.load(pt_path)
        _targets = _targets[shuffle]
        _labels = torch.zeros(len(_images))
        # concatenate positive and negative examples
        self.images = torch.cat((images, _images))
        self.targets = torch.cat((targets, _targets))
        self.labels = torch.cat((labels, _labels))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        target = self.targets[index]

        image = Image.fromarray(image.numpy(), mode='L')
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, label

    def __len__(self):
        return len(self.images)
