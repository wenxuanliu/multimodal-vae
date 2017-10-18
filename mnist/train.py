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
from torchvision import transforms, datasets

from model import MultimodalVAE


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


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, n_latents=20, use_cuda=False):
    """Return EmbedNet instance"""
    if use_cuda:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path,
                                map_location=lambda storage, location: storage)

    vae = MultimodalVAE(n_latents=n_latents)
    vae.load_state_dict(checkpoint['state_dict'])
    
    return vae


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

    # create loaders for MNIST
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                     transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                     transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)

    # load multimodal VAE
    vae = MultimodalVAE(n_latents=args.n_latents)
    if args.cuda:
        vae.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=args.lr)


    def train(epoch):
        vae.train()
        loss_meter = AverageMeter()

        for batch_idx, (image, text) in enumerate(train_loader):
            if args.cuda:
                image, text = image.cuda(), text.cuda()
            image, text = Variable(image), Variable(text)
            image = image.view(-1, 784)  # flatten image
            optimizer.zero_grad()
            
            # for each batch, use 3 types of examples (joint, image-only, and text-only)
            # this way, we can hope to reconstruct both modalities from one
            recon_image_1, recon_text_1, mu_1, logvar_1 = vae(image, text)
            recon_image_2, recon_text_2, mu_2, logvar_2 = vae(image=image)
            recon_image_3, recon_text_3, mu_3, logvar_3 = vae(text=text)
            # combine all of the batches (no need to reorder; we show the model all at once)
            recon_image = torch.cat((recon_image_1, recon_image_2, recon_image_3))
            recon_text = torch.cat((recon_text_1, recon_text_2, recon_text_3))
            mu = torch.cat((mu_1, mu_2, mu_3))
            logvar = torch.cat((logvar_1, logvar_2, logvar_3))
            # combine all of the input image/texts
            image_3x = torch.cat((image, image, image))
            text_3x = torch.cat((text, text, text))
            
            loss = loss_function(recon_image, image_3x, recon_text, text_3x, mu, logvar)
            loss.backward()
            loss_meter.update(loss.data[0], len(image))
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg))

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_meter.avg))


    def test():
        vae.eval()
        test_loss = 0

        for batch_idx, (image, text) in enumerate(test_loader):
            if args.cuda:
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
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder='./trained_models')     
