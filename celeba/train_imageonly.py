from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

import datasets
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


def loss_function(recon_x, x, mu, logvar, kl_lambda=1e-3):
    batch_size = mu.size(0)    
    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 64 * 64), 
                                 x.view(-1, 3 * 64 * 64))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / batch_size * kl_lambda
    return BCE + KLD


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_latents', type=int, default=100,
                        help='size of the latent embedding')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
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

    if not os.path.isdir('./trained_models/image_only'):
        os.makedirs('./trained_models/image_only')

    if not os.path.isdir('./results'):
        os.makedirs('./results')

    if not os.path.isdir('./results/image_only'):
        os.makedirs('./results/image_only')

    preprocess_data = transforms.Compose([transforms.Scale(64),
                                          transforms.CenterCrop(64),
                                          transforms.ToTensor()])

    train_loader = torch.utils.data.DataLoader(
        datasets.CelebAttributes(partition='train', 
                                 image_transform=preprocess_data),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CelebAttributes(partition='val',
                                 image_transform=preprocess_data),
        batch_size=args.batch_size, shuffle=True)

    vae = ImageVAE(n_latents=args.n_latents)
    if args.cuda:
        vae.cuda()
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)


    def train(epoch):
        vae.train()
        loss_meter = AverageMeter()
        for batch_idx, (data, _) in enumerate(train_loader):
            data = Variable(data)
            if args.cuda:
                data = data.cuda()
            optimizer.zero_grad()
            recon_data, mu, logvar = vae(data)
            loss = loss_function(recon_data, data, mu, logvar)
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
        for i, (data, _) in enumerate(test_loader):
            if args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_data, mu, logvar = vae(data)
            test_loss += loss_function(recon_data, data, mu, logvar).data[0]

        test_loss /= len(test_loader)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss


    best_loss = sys.maxint
    for epoch in range(1, args.epochs + 1):
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
        save_image(sample.data.view(64, 3, 64, 64),
                   './results/image_only/sample_image_epoch%d.png' % epoch)
