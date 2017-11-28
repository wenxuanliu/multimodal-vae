from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import ImageVAE
from train import loss_function, AverageMeter


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    """Return EmbedNet instance"""
    if use_cuda:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path,
                                map_location=lambda storage, location: storage)

    vae = ImageVAE(n_latents=checkpoint['n_latents'])
    vae.load_state_dict(checkpoint['state_dict'])
    
    return vae


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_latents', type=int, default=20,
                        help='size of the latent embedding')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--anneal_kl', action='store_true', default=False, 
                        help='if True, use a fixed interval of doubling the KL term')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if not os.path.isdir('./trained_models'):
        os.makedirs('./trained_models')

    if not os.path.isdir('./trained_models/image_only'):
        os.makedirs('./trained_models/image_only')

    if not os.path.isdir('./results'):
        os.makedirs('./results')

    if not os.path.isdir('./results/image_only'):
        os.makedirs('./results/image_only')

    transform_train = transforms.Compose([transforms.Scale(32),
                                          transforms.CenterCrop(32),
                                          transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.Scale(32),
                                         transforms.CenterCrop(32),
                                         transforms.ToTensor()])
 
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

    vae = ImageVAE(n_latents=args.n_latents)
    if args.cuda:
        vae.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=args.lr)


    def train(epoch):
        print('Using KL Lambda: {}'.format(kl_lambda))
        vae.train()
        loss_meter = AverageMeter()

        for batch_idx, (data, _) in enumerate(train_loader):
            data = Variable(data)
            if args.cuda:
                data = data.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            # watch out for logvar -- could explode if learning rate is too high.  
            loss = loss_function(mu, logvar, recon_image=recon_batch, image=data, 
                                 kl_lambda=kl_lambda, lambda_xy=1.)
            loss_meter.update(loss.data[0], len(data))            
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg))

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_meter.avg))


    def test():
        vae.eval()
        test_loss = 0
        for i, (data, _) in enumerate(test_loader):
            if args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = vae(data)
            test_loss += loss_function(mu, logvar, recon_image=recon_batch, image=data, 
                                       kl_lambda=kl_lambda, lambda_xy=1.).data[0]

        test_loss /= len(test_loader)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss


    kl_lambda = 5e-4
    schedule = iter([1e-5, 1e-4, 1e-3])  # , 1e-2, 1e-1, 1.0])
    best_loss = sys.maxint
    for epoch in range(1, args.epochs + 1):
        if (epoch - 1) % 5 == 0 and args.anneal_kl:
            try:
                kl_lambda = next(schedule)
            except:
                pass

        train(epoch)
        loss = test()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint({
            'state_dict': vae.state_dict(),
            'best_loss': best_loss,
            'n_latents': args.n_latents,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder='./trained_models/image_only')

        sample = Variable(torch.randn(64, args.n_latents))
        if args.cuda:
           sample = sample.cuda()

        sample = vae.decode(sample).cpu()
        save_image(sample.data.view(64, 3, 32, 32),
                   './results/image_only/sample_image_epoch%d.png' % epoch)
