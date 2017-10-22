from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
from torch.utils.data.dataset import Dataset

import numpy as np
from PIL import Image


class ShuffleMNIST(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        data_path = os.path.join(data_path, 'processed',
                                 'training.pt' if train else 'test.pt')
        data, targets = torch.load(data_path)
        shuffled = targets.numpy()
        np.random.shuffle(shuffled)
        shuffled = torch.from_numpy(shuffled)
        self.data = data
        self.targets = targets
        self.shuffled = shuffled
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]
        shuffled = self.shuffled[index]

        if self.transform is not None:
            data = Image.fromarray(data.numpy(), mode='L')
            data = self.tranform(data)

        return data, targets, shuffled

    def __len__(self):
        return len(self.data)
