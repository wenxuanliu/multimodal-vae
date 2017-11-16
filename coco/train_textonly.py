from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, datasets

from model import TextVAE
from train import loss_function, AverageMeter
from utils import text_transformer


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

    vae = TextVAE(checkpoint['n_latents'], use_cuda=use_cuda)
    vae.load_state_dict(checkpoint['state_dict'])
    
    return vae


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_latents', type=int, default=200,
                        help='size of the latent embedding')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
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

    if not os.path.isdir('./trained_models/text_only'):
        os.makedirs('./trained_models/text_only')

    if not os.path.isdir('./results'):
        os.makedirs('./results')

    if not os.path.isdir('./results/text_only'):
        os.makedirs('./results/text_only')

    # even though we don't use images, we need the transformations to be 
    # able to use DataLoader.
    transform_train = transforms.Compose([transforms.RandomSizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.Scale(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()])
    # transformer for text.
    transform_text = text_transformer(deterministic=False)

    train_loader = torch.utils.data.DataLoader(
        datasets.CocoCaptions('./data/coco/train2014', 
                              './data/coco/annotations/captions_train2014.json',
                              transform=transform_train,
                              target_transform=transform_text),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CocoCaptions('./data/coco/val2014', 
                              './data/coco/annotations/captions_val2014.json',
                              transform=transform_test,
                              target_transform=transform_text),
        batch_size=args.batch_size, shuffle=True) 

    vae = TextVAE(args.n_latents, use_cuda=args.cuda)
    if args.cuda:
        vae.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=args.lr)


    def train(epoch):
        vae.train()
        loss_meter = AverageMeter()

        for batch_idx, (_, data) in enumerate(train_loader):
            data = Variable(data)
            if args.cuda:
                data = data.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = loss_function(mu, logvar, recon_text=recon_batch, text=data, 
                                 kl_lambda=kl_lambda, lambda_yx=1.)
            loss.backward()
            loss_meter.update(loss.data[0], len(data))
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg))

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_meter.avg))


    def test():
        vae.eval()
        test_loss = 0
        for i, (_, data) in enumerate(test_loader):
            if args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = vae(data)
            test_loss += loss_function(mu, logvar, recon_text=recon_batch, text=data, 
                                       kl_lambda=kl_lambda, lambda_yx=1.).data[0]

        test_loss /= len(test_loader)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss


    kl_lambda = 1e-3
    schedule = iter([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
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
        }, is_best, folder='./trained_models/text_only')

        sample = Variable(torch.randn(64, args.n_latents))
        if args.cuda:
           sample = sample.cuda()

        sample_texts = vae.decoder.generate(sample)
        with open('./results/text_only/sample_text_epoch%d.txt' % epoch, 'w') as fp:
            fp.writelines(sample_texts)
