"""Weak supervision test. This will loop through and train
a number of MMVAE models where we progressively cut the 
number of examples we show it in half. The point is to test 
how well the model is able to learn on sparse relation data.

This is different than weak.py because we cut 1 or both modalities
instead of paired data.
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
from torchvision import transforms

import datasets
from model import MultimodalVAE
from train import AverageMeter
from train import save_checkpoint, load_checkpoint
from utils import n_characters, max_length
from utils import tensor_to_string, charlist_tensor
from train import joint_loss_function, image_loss_function, text_loss_function


def train_pipeline(out_dir, weak_perc_m1, weak_perc_m2, n_latents=20, batch_size=128, 
                   epochs=20, lr=1e-3, log_interval=10, cuda=False):
    """Pipeline to train and test MultimodalVAE on MNIST dataset. This is 
    identical to the code in train.py.

    :param out_dir: directory to store trained models
    :param weak_perc_m1: percent of time to show first modality
    :param weak_perc_m2: percent of time to show second modality
    :param n_latents: size of latent variable (default: 20)
    :param batch_size: number of examples to show at once (default: 128)
    :param epochs: number of loops over dataset (default: 20)
    :param lr: learning rate (default: 1e-3)
    :param log_interval: interval of printing (default: 10)
    :param cuda: whether to use cuda or not (default: False)
    """
        # create loaders for MNIST
    train_loader = torch.utils.data.DataLoader(
        datasets.MultiMNIST('./data', train=True, download=True,
                            transform=transforms.ToTensor(),
                            target_transform=charlist_tensor),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MultiMNIST('./data', train=False, download=True,
                            transform=transforms.ToTensor(),
                            target_transform=charlist_tensor),
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
            optimizer.zero_grad()
            
            recon_image_1, recon_text_1, mu_1, logvar_1 = vae(image, text)
            loss = joint_loss_function( recon_image_1, image, recon_text_1, text, mu_1, logvar_1,
                                        batch_size=batch_size, kl_lambda=kl_lambda,
                                        lambda_xy=1., lambda_yx=1.)
            joint_loss_meter.update(loss.data[0], len(image))

            flip = np.random.random()
            if flip < weak_perc_m1:
                recon_image_2, _, mu_2, logvar_2 = vae(image=image)
                loss_2 = image_loss_function(recon_image_2, image, mu_2, logvar_2,
                                             batch_size=batch_size, kl_lambda=kl_lambda,
                                             lambda_x=1.)
                image_loss_meter.update(loss_2.data[0], len(image))
                loss += loss_2

            flip = np.random.ranodm()
            if flip > weak_perc_m2:
                _, recon_text_3, mu_3, logvar_3 = vae(text=text)
                loss_3 = text_loss_function(recon_text_3, text, mu_3, logvar_3,
                                            batch_size=batch_size, kl_lambda=kl_lambda,
                                            lambda_y=100.)
                text_loss_meter.update(loss_3.data[0], len(text))
                loss += loss_3 

            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('[Weak (Image) {:.0f}% | Weak (Text) {:.0f}%] Train Epoch: {} [{}/{} ({:.0f}%)]\tJoint Loss: {:.6f}\tImage Loss: {:.6f}\tText Loss: {:.6f}'.format(
                    100. * weak_perc_m1, 100. * weak_perc_m2, epoch, batch_idx * len(image), 
                    len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                    joint_loss_meter.avg, image_loss_meter.avg, text_loss_meter.avg))

        print('====> [Weak (Image) {:.0f}% | Weak (Text) {:.0f}%] Epoch: {} Joint loss: {:.4f}\tImage loss: {:.4f}\tText loss: {:.4f}'.format(
            100. * weak_perc_m1, 100. * weak_perc_m2, epoch, joint_loss_meter.avg, 
            image_loss_meter.avg, text_loss_meter.avg))


    def test():
        vae.eval()
        test_joint_loss = 0
        test_image_loss = 0
        test_text_loss = 0

        for batch_idx, (image, text) in enumerate(test_loader):
            if cuda:
                image, text = image.cuda(), text.cuda()
            image, text = Variable(image), Variable(text)
                
            recon_image_1, recon_text_1, mu_1, logvar_1 = vae(image, text)
            recon_image_2, _, mu_2, logvar_2 = vae(image=image)
            _, recon_text_3, mu_3, logvar_3 = vae(text=text)

            loss_1 = joint_loss_function(recon_image_1, image, recon_text_1, text, mu_1, logvar_1,
                                         batch_size=batch_size, kl_lambda=kl_lambda,
                                         lambda_xy=1., lambda_yx=1.)
            loss_2 = image_loss_function(recon_image_2, image, mu_2, logvar_2,
                                         batch_size=batch_size, kl_lambda=kl_lambda,
                                         lambda_x=1.)
            loss_3 = text_loss_function(recon_text_3, text, mu_3, logvar_3,
                                        batch_size=batch_size, kl_lambda=kl_lambda,
                                        lambda_y=100.)
            
            test_joint_loss += loss_1.data[0]
            test_image_loss += loss_2.data[0]
            test_text_loss += loss_3.data[0]

        test_loss = test_joint_loss + test_image_loss + test_text_loss
        test_joint_loss /= len(test_loader.dataset)
        test_image_loss /= len(test_loader.dataset)
        test_text_loss /= len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)

        print('====> [Weak (Image) {:.0f}% | Weak (Text) {:.0f}%] Test joint loss: {:.4f}\timage loss: {:.4f}\ttext loss:{:.4f}'.format(
            100. * weak_perc_m1, 100. * weak_perc_m2, test_joint_loss, test_image_loss, test_text_loss))

        return test_loss, (test_joint_loss, test_image_loss, test_text_loss)


    best_loss = sys.maxint
    schedule = iter([5e-5, 1e-4, 5e-4, 1e-3])

    for epoch in range(1, epochs + 1):
        if (epoch - 1) % 10 == 0:
            try:
                kl_lambda = next(schedule)
            except:
                pass

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
    parser.add_argument('--n_latents', type=int, default=100,
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

    supervision_dir = './trained_models/weak_modal_supervision'
    if os.path.isdir(supervision_dir):
        shutil.rmtree(supervision_dir)
    os.makedirs(supervision_dir)
    print('Created directory: %s' % supervision_dir)

    for weak_perc_1 in [0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.]:
        for weak_perc_2 in [0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.]:
            perc_dir = os.path.join(supervision_dir, 'weak_perc_m1_{}_m2_{}'.format(
                weak_perc_1, weak_perc_2))
            
            if os.path.isdir(perc_dir):
                shutil.rmtree(perc_dir)
            os.makedirs(perc_dir)
            print('Created directory: %s' % perc_dir)

            # train a modal that shows all paired data but a subset of the data for each 
            # modality. We can then make a heatmap and do analysis.
            train_pipeline(perc_dir, weak_perc_1, weak_perc_2, n_latents=args.n_latents, 
                           batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, 
                           log_interval=args.log_interval, cuda=args.cuda)
