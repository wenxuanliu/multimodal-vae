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
from torchvision import transforms
from torchvision.utils import save_image

import datasets
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


def load_checkpoint(file_path, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path,
                                map_location=lambda storage, location: storage)
    vae = MultimodalVAE(n_latents=checkpoint['n_latents'], 
                        use_cuda=use_cuda)
    vae.load_state_dict(checkpoint['state_dict'])
    if use_cuda:
        vae.cuda()
    return vae


def loss_function(mu, logvar, recon_x=None, x=None, recon_y=None, y=None,  
                  kl_lambda=1e-3, lambda_x=1., lambda_y=1.):
    batch_size = mu.size(0)
    x_BCE, y_BCE = 0, 0
    
    if recon_x is not None and x is not None:
        x_BCE = F.binary_cross_entropy(recon_x.view(-1, 1 * 64 * 64), 
                                       x.view(-1, 1 * 64 * 64))

    if recon_y is not None and y is not None:
        y_BCE = 0
        for i in xrange(y.size(1)):
            y_BCE += F.binary_cross_entropy(recon_y, y)
        y_BCE /= y.size(1)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / batch_size * kl_lambda
    return lambda_x * x_BCE + lambda_y * y_BCE + KLD


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_latents', type=int, default=100,
                        help='size of the latent embedding (default: 100)')
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

    # create loaders for CelebA
    train_loader = torch.utils.data.DataLoader(
        datasets.CelebAttributes('./data', partition='train'),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CelebAttributes('./data', partition='val'),
        batch_size=args.batch_size, shuffle=True)

    # load multimodal VAE
    vae = MultimodalVAE(args.n_latents, use_cuda=args.cuda)
    if args.cuda:
        vae.cuda()
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)


    def train(epoch):
        vae.train()
        joint_loss_meter = AverageMeter()
        image_loss_meter = AverageMeter()
        attrs_loss_meter = AverageMeter()

        for batch_idx, (image, attrs) in enumerate(train_loader):
            if args.cuda:
                image, attrs = image.cuda(), attrs.cuda()
            image, attrs = Variable(image), Variable(attrs)
            optimizer.zero_grad()
            
            # for each batch, use 3 types of examples (joint, image-only, and attrs-only)
            # this way, we can hope to reconstruct both modalities from one
            recon_image_1, recon_attrs_1, mu_1, logvar_1 = vae(image=image, attrs=attrs)
            recon_image_2, recon_attrs_2, mu_2, logvar_2 = vae(image=image)
            recon_image_3, recon_attrs_3, mu_3, logvar_3 = vae(attrs=attrs)
            
            loss_1 = loss_function(mu_1, logvar_1, recon_x=recon_image_1, x=image, 
                                   recon_y=recon_attrs_1, y=attrs, kl_lambda=kl_lambda, 
                                   lambda_x=1., lambda_y=1.)
            loss_2 = loss_function(mu_2, logvar_2, recon_x=recon_image_2, x=image, 
                                   recon_y=recon_attrs_2, y=attrs, kl_lambda=kl_lambda, 
                                   lambda_xy=1., lambda_yx=0.5)
            loss_3 = loss_function(mu_3, logvar_3, recon_x=recon_image_3, x=image, 
                                   recon_y=recon_attrs_3, y=attrs, kl_lambda=kl_lambda, 
                                   lambda_xy=0., lambda_yx=1.)
            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            
            joint_loss_meter.update(loss_1.data[0], len(image))
            image_loss_meter.update(loss_2.data[0], len(image))
            attrs_loss_meter.update(loss_3.data[0], len(attrs))
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tJoint Loss: {:.6f}\tImage Loss: {:.6f}\tAttrs Loss: {:.6f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), joint_loss_meter.avg,
                    image_loss_meter.avg, attrs_loss_meter.avg))

        print('====> Epoch: {}\tJoint loss: {:.4f}\tImage loss: {:.4f}\tAttrs loss: {:.4f}'.format(
            epoch, joint_loss_meter.avg, image_loss_meter.avg, attrs_loss_meter.avg))


    def test():
        vae.eval()
        test_joint_loss = 0
        test_image_loss = 0
        test_attrs_loss = 0

        for batch_idx, (image, attrs) in enumerate(test_loader):
            if args.cuda:
                image, attrs = image.cuda(), attrs.cuda()
            image = Variable(image, volatile=True)
            attrs = Variable(attrs, volatile=True)
                
            recon_image_1, recon_attrs_1, mu_1, logvar_1 = vae(image=image, attrs=attrs)
            recon_image_2, recon_attrs_2, mu_2, logvar_2 = vae(image=image)
            recon_image_3, recon_attrs_3, mu_3, logvar_3 = vae(attrs=attrs)
            
            loss_1 = loss_function(mu_1, logvar_1, recon_x=recon_image_1, x=image, 
                                   recon_y=recon_attrs_1, y=attrs, kl_lambda=kl_lambda, 
                                   lambda_x=1., lambda_y=1.)
            loss_2 = loss_function(mu_2, logvar_2, recon_x=recon_image_2, x=image, 
                                   recon_y=recon_attrs_2, y=attrs, kl_lambda=kl_lambda, 
                                   lambda_xy=1., lambda_yx=0.5)
            loss_3 = loss_function(mu_3, logvar_3, recon_x=recon_image_3, x=image, 
                                   recon_y=recon_attrs_3, y=attrs, kl_lambda=kl_lambda, 
                                   lambda_xy=0., lambda_yx=1.)
            test_joint_loss += loss_1.data[0]
            test_image_loss += loss_2.data[0]
            test_attrs_loss += loss_3.data[0]

        test_loss = test_joint_loss + test_image_loss + test_attrs_loss
        test_joint_loss /= len(test_loader)
        test_image_loss /= len(test_loader)
        test_attrs_loss /= len(test_loader)
        test_loss /= len(test_loader)
        
        print('====> Test Epoch\tJoint loss: {:.4f}\tImage loss: {:.4f}\tAttrs loss:{:.4f}'.format(
            test_joint_loss, test_image_loss, test_attrs_loss))
        
        return test_loss, (test_joint_loss, test_image_loss, test_attrs_loss)


    best_loss = sys.maxint
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        loss, (joint_loss, image_loss, attrs_loss) = test()
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint({
            'state_dict': vae.state_dict(),
            'best_loss': best_loss,
            'joint_loss': joint_loss,
            'image_loss': image_loss,
            'attrs_loss': attrs_loss,
            'n_latents': args.n_latents,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder='./trained_models')   

        sample = Variable(torch.randn(64, args.n_latents))
        if args.cuda:
           sample = sample.cuda()

        # reconstruct image
        image_sample = vae.image_decoder(sample).cpu().data
        save_image(image_sample.view(64, 1, 64, 64),
                   './results/sample_image_epoch%d.png' % epoch)
        # reconstruct attributes
        attrs_sample = vae.attrs_decoder.generate(sample).cpu().data.long()
        sample_attrs = []
        for i in xrange(sample.size(0)):
            attrs = datasets.tensor_to_attributes(attrs_sample[i])
            sample_attrs.append(','.join(attrs))

        with open('./results/sample_attrs_epoch%d.txt' % epoch, 'w') as fp:
            for attrs in sample_attrs:
                fp.write('%s\n' % attrs)
