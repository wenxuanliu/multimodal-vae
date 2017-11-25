from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shutil
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import GatedPixelCNN
from model import cross_entropy_by_dim, log_softmax_by_dim
from train import AverageMeter


def quantisize(images, levels):
    """Convert images to N levels from 0 to N.

    :param images: numpy array
    :return: numpy array
    """
    return (np.digitize(images, np.arange(levels) / levels) - 1).astype('i')


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path,
                                map_location=lambda storage, location: storage)
    model = GatedPixelCNN(n_groups=checkpoint['n_groups'],
                          data_channels=3, out_dims=checkpoint['out_dims'])
    model.load_state_dict(checkpoint['state_dict'])
    if use_cuda:
        model.cuda()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_blocks', type=int, default=15, metavar='N',
                        help='number of blocks ResNet')
    parser.add_argument('--out_dims', type=int, default=256, metavar='N',
                        help='2|4|8|16|...|256 (default: 256)')
    parser.add_argument('--image_size', type=int, default=32, metavar='N',
                        help='size to reshape image to and generate (default: 32)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--cifar', action='store_true', default=False,
                        help='train on CIFAR if set (default: False)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training (default: False)')
    args = parser.parse_args()
    args.folder_name = 'pixel_cifar' if args.cifar else 'pixel_cnn'
    args.cuda = args.cuda and torch.cuda.is_available()
    assert args.out_dims <= 256 and args.out_dims > 1

    if not os.path.isdir('./trained_models'):
        os.makedirs('./trained_models')

    if not os.path.isdir('./trained_models/%s' % args.folder_name):
        os.makedirs('./trained_models/%s' % args.folder_name)

    if not os.path.isdir('./results'):
        os.makedirs('./results')

    if not os.path.isdir('./results/%s' % args.folder_name):
        os.makedirs('./results/%s' % args.folder_name)

    def preprocess(x):
        transform = transforms.Compose([transforms.Scale(args.image_size),
                                        transforms.CenterCrop(args.image_size),
                                        transforms.ToTensor()])
        x = transform(x)
        if args.out_dims < 256:
            x = quantisize(x.numpy(), args.out_dims).astype('f')
            x = torch.from_numpy(x) / (args.out_dims - 1)
        return x

    if args.cifar:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data/cifar', train=True,
                             download=True, transform=preprocess),
            batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data/cifar', train=False,
                             download=True, transform=preprocess),
            batch_size=args.batch_size, shuffle=True)
    else:
        # create loaders for COCO
        train_loader = torch.utils.data.DataLoader(
            datasets.CocoCaptions('./data/coco/train2014', 
                                  './data/coco/annotations/captions_train2014.json',
                                  transform=preprocess),
            batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CocoCaptions('./data/coco/val2014', 
                                  './data/coco/annotations/captions_val2014.json',
                                  transform=preprocess),
            batch_size=args.batch_size, shuffle=True)

    # load multimodal VAE
    model = GatedPixelCNN(n_blocks=args.n_blocks, data_channels=3, 
                          out_dims=args.out_dims)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)


    def train(epoch):
        model.train()
        loss_meter = AverageMeter()

        for batch_idx, (data, _) in enumerate(train_loader):
            data = Variable(data)
            target = Variable((data.data * (args.out_dims - 1)).long())

            if args.cuda:
                data = data.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = cross_entropy_by_dim(output, target)
            loss_meter.update(loss.data[0], len(data))
            
            loss.backward()
            # clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg))

        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))


    def test():
        model.eval()
        loss_meter = AverageMeter()

        for batch_idx, (data, _) in enumerate(test_loader):
            data = Variable(data)
            target = Variable((data.data * (args.out_dims - 1)).long())
            
            if args.cuda:
                data = data.cuda()
                target = target.cuda()

            output = model(data)
            loss = cross_entropy_by_dim(output, target)
            loss_meter.update(loss.data[0], len(data))
        
        print('====> Test Epoch\tLoss: {:.4f}'.format(loss_meter.avg))
        return loss_meter.avg


    def generate(epoch):
        sample = torch.zeros(64, 3, args.image_size, args.image_size)
        if args.cuda:
            sample = sample.cuda()
        model.eval() 

        for i in xrange(args.image_size):
            for j in xrange(args.image_size):
                for k in xrange(3):
                    output = model(Variable(sample, volatile=True))
                    output = torch.exp(log_softmax_by_dim(output, dim=1))
                    probs = output[:, :, k, i, j].data
                    sample[:, k, i, j] = torch.multinomial(probs, 1).float() / (args.out_dims - 1)

        save_image(sample, './results/{}/sample_{}.png'.format(args.folder_name, epoch)) 


    best_loss = sys.maxint

    for epoch in xrange(args.epochs):
        train(epoch)
        loss = test()
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
            'out_dims': args.out_dims,
            'n_blocks': args.n_blocks,
        }, is_best, folder='./trained_models/%s' % args.folder_name)     

        if is_best:
            generate(epoch)
