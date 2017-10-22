"""Weak supervision test. This will loop through and train
a number of MMVAE models where we progressively cut the 
number of examples we show it in half. The point is to test 
how well the model is able to learn on sparse relation data.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shutil
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, datasets

from model import MultimodalVAE
from train import AverageMeter
from train import save_checkpoint, load_checkpoint
from train import loss_function


def train_pipeline(out_dir, weak_perc, n_latents=20, batch_size=128, epochs=20, lr=1e-3, 
                   log_interval=10, cuda=False):
    """Pipeline to train and test MultimodalVAE on MNIST dataset. This is 
    identical to the code in train.py.

    :param out_dir: directory to store trained models
    :param weak_perc: percent of time to show a relation pair (vs no relation pair)
    :param n_latents: size of latent variable (default: 20)
    :param batch_size: number of examples to show at once (default: 128)
    :param epochs: number of loops over dataset (default: 20)
    :param lr: learning rate (default: 1e-3)
    :param log_interval: interval of printing (default: 10)
    :param cuda: whether to use cuda or not (default: False)
    """
    # create loaders for MNIST
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    # load multimodal VAE
    vae = MultimodalVAE(n_latents=n_latents)
    if cuda:
        vae.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=lr)


    def loss_function(recon_image, image, recon_text, text, mu, logvar):
        image_BCE = F.binary_cross_entropy(recon_image, image.view(-1, 784))
        text_BCE = F.nll_loss(recon_text, text)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= args.batch_size * 784
        return image_BCE + text_BCE + KLD
    

    def train(epoch):
        random.seed(42)
        np.random.seed(42)  # important to have the same seed
                            # in order to make the same choices for weak supervision
                            # otherwise, we end up showing different examples over epochs
        vae.train()
        loss_meter = AverageMeter()

        for batch_idx, (image, text) in enumerate(train_loader):
            if cuda:
                image, text = image.cuda(), text.cuda()
            image, text = Variable(image), Variable(text)
            image = image.view(-1, 784)  # flatten image
            optimizer.zero_grad()
            
            # for each batch, use 3 types of examples (joint, image-only, and text-only)
            # this way, we can hope to reconstruct both modalities from one
            recon_image_2, recon_text_2, mu_2, logvar_2 = vae(image=image)
            recon_image_3, recon_text_3, mu_3, logvar_3 = vae(text=text)
            # combine all of the batches (no need to reorder; we show the model all at once)
            recon_image = torch.cat((recon_image_2, recon_image_3))
            recon_text = torch.cat((recon_text_2, recon_text_3))
            mu = torch.cat((mu_2, mu_3))
            logvar = torch.cat((logvar_2, logvar_3))
            # combine all of the input image/texts
            image_nx = torch.cat((image, image))
            text_nx = torch.cat((text, text))
            n = 2

            # if we flip(weak_perc), then we show a paired relation example.
            flip = np.random.random()
            if flip < weak_perc:
                recon_image_1, recon_text_1, mu_1, logvar_1 = vae(image, text)
                recon_image = torch.cat((recon_image, recon_image_1))
                recon_text = torch.cat((recon_text, recon_text_1))
                mu = torch.cat((mu, mu_1))
                logvar = torch.cat((logvar, logvar_1))
                image_nx = torch.cat((image_nx, image))
                text_nx = torch.cat((text_nx, text))
                n = 3
                import pdb; pdb.set_trace()

            loss = loss_function(recon_image, image_nx, recon_text, text_nx, mu, logvar)
            loss.backward()
            loss_meter.update(loss.data[0], len(image) * n)
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('[Weak {:.0f}%] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    100. * weak_perc, epoch, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg))

        print('====> [Weak {:.0f}%] Epoch: {} Average loss: {:.4f}'.format(
            100. * weak_perc, epoch, loss_meter.avg))


    def test():
        vae.eval()
        test_loss = 0

        for batch_idx, (image, text) in enumerate(test_loader):
            if cuda:
                image, text = image.cuda(), text.cuda()
            image, text = Variable(image), Variable(text)
            image = image.view(-1, 784)  # flatten image
                
            recon_image_1, recon_text_1, mu_1, logvar_1 = vae(image, text)
            recon_image_2, recon_text_2, mu_2, logvar_2 = vae(image=image)
            recon_image_3, recon_text_3, mu_3, logvar_3 = vae(text=text)
            recon_image = torch.cat((recon_image_1, recon_image_2, recon_image_3))
            recon_text = torch.cat((recon_text_1, recon_text_2, recon_text_3))
            mu = torch.cat((mu_1, mu_2, mu_3))
            logvar = torch.cat((logvar_1, logvar_2, logvar_3))
            image_3x = torch.cat((image, image, image))
            text_3x = torch.cat((text, text, text))
            
            loss = loss_function(recon_image, image_3x, recon_text, text_3x, mu, logvar)
            test_loss += loss.data[0]

        test_loss /= len(test_loader.dataset)
        print('====> [Weak {:.0f}%] Test set loss: {:.4f}'.format(100. * weak_perc, test_loss))
        return test_loss


    best_loss = sys.maxint
    for epoch in range(1, epochs + 1):
        train(epoch)
        loss = test()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint({
            'state_dict': vae.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder=out_dir)     


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_latents', type=int, default=20,
                        help='size of the latent embedding')
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

    supervision_dir = './trained_models/weak_supervision'
    if os.path.isdir(supervision_dir):
        shutil.rmtree(supervision_dir)
    os.makedirs(supervision_dir)
    print('Created directory: %s' % supervision_dir)

    for weak_perc in [0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.]:
        perc_dir = os.path.join(supervision_dir, 'weak_perc_{}'.format(weak_perc))
        if os.path.isdir(perc_dir):
            shutil.rmtree(perc_dir)
        os.makedirs(perc_dir)
        print('Created directory: %s' % perc_dir)
        
        train_pipeline(perc_dir, weak_perc, n_latents=args.n_latents, 
                       batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, 
                       log_interval=args.log_interval, cuda=args.cuda)
    
