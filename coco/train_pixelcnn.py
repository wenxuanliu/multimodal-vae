from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import PixelCNN
from train import AverageMeter


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
    model = PixelCNN()
    model.load_state_dict(checkpoint['state_dict'])
    
    if use_cuda:
        model.cuda()

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if not os.path.isdir('./trained_models'):
        os.makedirs('./trained_models')

    if not os.path.isdir('./trained_models/pixel_cnn'):
        os.makedirs('./trained_models/pixel_cnn')

    if not os.path.isdir('./results'):
        os.makedirs('./results')

    if not os.path.isdir('./results/pixel_cnn'):
        os.makedirs('./results/pixel_cnn')

    # create transformers for preprocessing images
    transform_train = transforms.Compose([transforms.Scale(64),
                                          transforms.CenterCrop(64),
                                          transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.Scale(64),
                                         transforms.CenterCrop(64),
                                         transforms.ToTensor()])
    # create loaders for COCO
    train_loader = torch.utils.data.DataLoader(
        datasets.CocoCaptions('./data/coco/train2014', 
                              './data/coco/annotations/captions_train2014.json',
                              transform=transform_train),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CocoCaptions('./data/coco/val2014', 
                              './data/coco/annotations/captions_val2014.json',
                              transform=transform_test),
        batch_size=args.batch_size, shuffle=True)

    # load multimodal VAE
    model = PixelCNN()
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    def train(epoch):
        model.train()
        loss_meter = AverageMeter()

        for batch_idx, (data, _) in enumerate(train_loader):
            if args.cuda:
                data = data.cuda()
            n_data = len(data)

            data = Variable(data)
            target = Variable((data.data * 255).long())

            optimizer.zero_grad()
            output = model(data)
            
            loss = 0
            for i in xrange(3):  # loop through each RGB dimension
                loss += F.cross_entropy(output[:, :, i, :, :], target[:, i, :, :])
            loss_meter.update(loss.data[0], len(data))
            
            loss.backward()
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
            if args.cuda:
                data = data.cuda()
            n_data = len(data)
            data = Variable(data)
            target = Variable((data.data * 255).long())
            output = model(data)
            
            loss = 0
            for i in xrange(3):  # loop through dimensions for each RGB channel
                loss += F.cross_entropy(output[:, :, i, :, :], target[:, i, :, :]) 
            loss_meter.update(loss.data[0], len(data))
        
        print('====> Test Epoch\tLoss: {:.4f}'.format(loss_meter.avg))
        return loss_meter.avg


    def generate(epoch):
        sample = torch.zeros(64, 3, 64, 64)
        if args.cuda:
            sample = sample.cuda()
        model.eval() 

        for i in xrange(64):
            for j in xrange(64):
                sample = Variable(sample, volatile=True)
                output = model(sample)
                
                for k in xrange(3):
                    probs = F.softmax(output[:, :, k, i, j]).data
                    sample[:, k, i, j] = torch.multinomial(probs, 1).float() / 255.

        save_image(sample, './results/pixel_cnn/sample_{}.png'.format(epoch)) 


    best_loss = sys.maxint

    for epoch in xrange(args.epochs):
        train(epoch)
        loss = test()
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint({
            'state_dict': vae.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder='./trained_models/pixel_cnn')     

        if is_best:
            generate(epoch)
