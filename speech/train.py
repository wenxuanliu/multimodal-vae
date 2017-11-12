from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json

import torch
from torch.autograd import Variable

from datasets import SpectrogramDataset, BucketingSampler, AudioDataLoader


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    audio_conf = dict(sample_rate=16000,
                      window_size=.02,
                      window_stride=.01,
                      window='hamming')    

    with open('./data/labels.json') as label_file:
        labels = str(''.join(json.load(label_file)))

    train_dataset = SpectrogramDataset(manifest_filepath='./data/libri_train_manifest.csv',
                                       audio_conf=audio_conf, labels=labels, 
                                       normalize=True, augment=True)
    test_dataset = SpectrogramDataset(manifest_filepath='./data/libri_test_manifest.csv',
                                      audio_conf=audio_conf, labels=labels, 
                                      normalize=True, augment=False)
    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    train_loader = AudioDataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size)
    train_sampler.shuffle()


    def train(epoch):
        for i, data in enumerate(train_loader):
            inputs, targets, input_percentages, target_sizes = data
            import pdb; pdb.set_trace()

    train(0)

