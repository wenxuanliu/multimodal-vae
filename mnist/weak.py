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
from train import joint_loss_function, image_loss_function, text_loss_function


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


    def train(epoch):
        random.seed(42)
        np.random.seed(42)  # important to have the same seed
                            # in order to make the same choices for weak supervision
                            # otherwise, we end up showing different examples over epochs
        vae.train()

        joint_loss_meter = AverageMeter()
        image_loss_meter = AverageMeter()
        text_loss_meter = AverageMeter()

        for batch_idx, (image, text) in enumerate(train_loader):
            if cuda:
                image, text = image.cuda(), text.cuda()
            image, text = Variable(image), Variable(text)
            image = image.view(-1, 784)  # flatten image
            optimizer.zero_grad()
            
            # for each batch, use 3 types of examples (joint, image-only, and text-only)
            # this way, we can hope to reconstruct both modalities from one
            recon_image_2, _, mu_2, logvar_2 = vae(image=image)
            _, recon_text_3, mu_3, logvar_3 = vae(text=text)

            loss_2 = image_loss_function(recon_image_2, image, mu_2, logvar_2,
                                         batch_size=batch_size)
            loss_3 = text_loss_function(recon_text_3, text, mu_3, logvar_3,
                                        batch_size=batch_size)  
            loss = loss_2 + loss_3          

            # if we flip(weak_perc), then we show a paired relation example.
            flip = np.random.random()
            if flip < weak_perc:
                recon_image_1, recon_text_1, mu_1, logvar_1 = vae(image, text)
                loss_1 = joint_loss_function(recon_image_1, image, recon_text_1, text, mu_1, logvar_1,
                                             batch_size=args.batch_size)
                loss = loss + loss_1

            loss.backward()
            joint_loss_meter.update(loss_1.data[0], len(image))
            image_loss_meter.update(loss_2.data[0], len(image))
            text_loss_meter.update(loss_3.data[0], len(text))
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('[Weak {:.0f}%] Train Epoch: {} [{}/{} ({:.0f}%)]\tJoint Loss: {:.6f}\tImage Loss: {:.6f}\tText Loss: {:.6f}'.format(
                    100. * weak_perc, epoch, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), joint_loss_meter.avg,
                    image_loss_meter.avg, text_loss_meter.avg))

        print('====> [Weak {:.0f}%] Epoch: {} Joint loss: {:.4f}\tImage loss: {:.4f}\tText loss: {:.4f}'.format(
            100. * weak_perc, epoch, joint_loss_meter.avg, image_loss_meter.avg, text_loss_meter.avg))


    def test():
        vae.eval()
        test_joint_loss = 0
        test_image_loss = 0
        test_text_loss = 0

        for batch_idx, (image, text) in enumerate(test_loader):
            if cuda:
                image, text = image.cuda(), text.cuda()
            image, text = Variable(image), Variable(text)
            image = image.view(-1, 784)  # flatten image
                
            recon_image_1, recon_text_1, mu_1, logvar_1 = vae(image, text)
            recon_image_2, _, mu_2, logvar_2 = vae(image=image)
            _, recon_text_3, mu_3, logvar_3 = vae(text=text)

            loss_1 = joint_loss_function(recon_image_1, image, recon_text_1, text, mu_1, logvar_1,
                                         batch_size=args.batch_size)
            loss_2 = image_loss_function(recon_image_2, image, mu_2, logvar_2,
                                         batch_size=args.batch_size)
            loss_3 = text_loss_function(recon_text_3, text, mu_3, logvar_3,
                                        batch_size=args.batch_size)
            
            test_joint_loss += loss_1.data[0]
            test_image_loss += loss_2.data[0]
            test_text_loss += loss_3.data[0]

        test_loss = test_joint_loss + test_image_loss + test_text_loss
        test_joint_loss /= len(test_loader.dataset)
        test_image_loss /= len(test_loader.dataset)
        test_text_loss /= len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)

        print('====> [Weak {:.0f}%] Test joint loss: {:.4f}\timage loss: {:.4f}\ttext loss:{:.4f}'.format(
            100. * weak_perc, test_joint_loss, test_image_loss, test_text_loss))

        return test_loss, (test_joint_loss, test_image_loss, test_text_loss)


    best_loss = sys.maxint
    for epoch in range(1, epochs + 1):
        train(epoch)
        loss, (joint_loss, image_loss, text_loss) = test()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint({
            'state_dict': vae.state_dict(),
            'best_loss': best_loss,
            'joint_loss': joint_loss,
            'image_loss': image_loss,
            'text_loss': text_loss,
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
    
