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

from utils import n_characters, max_length
from utils import tensor_to_string, charlist_tensor

DEFAULT_N_LATENTS = 100


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
    """Return EmbedNet instance"""
    if use_cuda:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path,
                                map_location=lambda storage, location: storage)

    n_latents = checkpoint['n_latents'] if 'n_latents' in checkpoint else DEFAULT_N_LATENTS
    vae = MultimodalVAE(n_latents=n_latents, use_cuda=use_cuda)
    vae.load_state_dict(checkpoint['state_dict'])
    
    if use_cuda:
        vae.cuda()

    return vae


def joint_loss_function(recon_image, image, recon_text, text, mu, logvar, 
                        kl_lambda=1e-3, lambda_xy=1, lambda_yx=1, scramble=False):
    batch_size = recon_image.size(0)
    if scramble:  # if we turn scramble on, we should not penalize the model for generating 
        # 1234 when the right answer is 4312. Location no longer matters so we should only 
        # consider the characters themselves. To represent this in the loss, we sort the 
        # true characters and we sort our model predictions by index i.e. 4312 --> 1234 and 
        # then compute the log-loss.
        text = torch.sort(text)[0]
        ix = torch.sort(torch.max(recon_text, dim=2)[1], dim=1)[1]
        recon_text = torch.stack([recon_text[i][ix[i]] for i in xrange(batch_size)])

    image_BCE = lambda_xy * F.binary_cross_entropy(recon_image.view(-1, 1 * 50 * 50), 
                                                   image.view(-1, 1 * 50 * 50))
    text_BCE = lambda_yx * F.nll_loss(recon_text.view(-1, recon_text.size(2)), text.view(-1))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / batch_size * kl_lambda
    return image_BCE + text_BCE + KLD


def image_loss_function(recon_image, image, mu, logvar, kl_lambda=1e-3, lambda_x=1):
    batch_size = recon_image.size(0)
    image_BCE = lambda_x * F.binary_cross_entropy(recon_image.view(-1, 1 * 50 * 50), 
                                                  image.view(-1, 1 * 50 * 50))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / batch_size * kl_lambda
    return image_BCE + KLD


def text_loss_function(recon_text, text, mu, logvar, kl_lambda=1e-3, lambda_y=100, scramble=False):
    batch_size = recon_text.size(0)
    if scramble:  # if we turn scramble on, we should not penalize the model for generating 
        # 1234 when the right answer is 4312. Location no longer matters so we should only 
        # consider the characters themselves. To represent this in the loss, we sort the 
        # true characters and we sort our model predictions by index i.e. 4312 --> 1234 and 
        # then compute the log-loss.
        text = torch.sort(text)[0]
        ix = torch.sort(torch.max(recon_text, dim=2)[1], dim=1)[1]
        recon_text = torch.stack([recon_text[i][ix[i]] for i in xrange(batch_size)])

    text_BCE = lambda_y * F.nll_loss(recon_text.view(-1, recon_text.size(2)), text.view(-1))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / batch_size * kl_lambda
    return text_BCE + KLD


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scramble', action='store_true', default=False, 
                        help='If True, compute text loss by checking character existence (not order)')
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
    parser.add_argument('--anneal_kl', action='store_true', default=False, 
                        help='if True, use a fixed interval of doubling the KL term')
    parser.add_argument('--anneal_lr', action='store_true', default=False,
                        help='If True, half learning rate every 5 epochs')
    parser.add_argument('--lambda_xy', type=float, default=1.)
    parser.add_argument('--lambda_yx', type=float, default=1.)
    parser.add_argument('--lambda_x', type=float, default=1.)
    parser.add_argument('--lambda_y', type=float, default=100.)
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # create loaders for MNIST
    train_loader = torch.utils.data.DataLoader(
        datasets.MultiMNIST('./data', train=True, download=True,
                            transform=transforms.ToTensor(),
                            target_transform=charlist_tensor),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MultiMNIST('./data', train=False, download=True,
                            transform=transforms.ToTensor(),
                            target_transform=charlist_tensor),
        batch_size=args.batch_size, shuffle=True)

    # load multimodal VAE
    vae = MultimodalVAE(args.n_latents, use_cuda=args.cuda)
    if args.cuda:
        vae.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=args.lr)


    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // 5))
        print('learning rate: {:.4f}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(epoch):
        vae.train()
        joint_loss_meter = AverageMeter()
        image_loss_meter = AverageMeter()
        text_loss_meter = AverageMeter()

        for batch_idx, (image, text) in enumerate(train_loader):
            if args.cuda:
                image, text = image.cuda(), text.cuda()
            image, text = Variable(image), Variable(text)
            optimizer.zero_grad()
            
            # for each batch, use 3 types of examples (joint, image-only, and text-only)
            # this way, we can hope to reconstruct both modalities from one
            recon_image_1, recon_text_1, mu_1, logvar_1 = vae(image, text)
            recon_image_2, _, mu_2, logvar_2 = vae(image=image)
            _, recon_text_3, mu_3, logvar_3 = vae(text=text)
            
            loss_1 = joint_loss_function(recon_image_1, image, recon_text_1, text, mu_1, logvar_1,
                                         kl_lambda=kl_lambda, lambda_xy=args.lambda_xy, lambda_yx=args.lambda_yx,
                                         scramble=args.scramble)
            loss_2 = image_loss_function(recon_image_2, image, mu_2, logvar_2,
                                         kl_lambda=kl_lambda, lambda_x=args.lambda_x)
            loss_3 = text_loss_function(recon_text_3, text, mu_3, logvar_3,
                                        kl_lambda=kl_lambda, lambda_y=args.lambda_y, scramble=args.scramble)
            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            joint_loss_meter.update(loss_1.data[0], len(image))
            image_loss_meter.update(loss_2.data[0], len(image))
            text_loss_meter.update(loss_3.data[0], len(text))
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tJoint Loss: {:.6f}\tImage Loss: {:.6f}\tText Loss: {:.6f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), joint_loss_meter.avg,
                    image_loss_meter.avg, text_loss_meter.avg))

        print('====> Epoch: {}\tJoint loss: {:.4f}\tImage loss: {:.4f}\tText loss: {:.4f}'.format(
            epoch, joint_loss_meter.avg, image_loss_meter.avg, text_loss_meter.avg))


    def test():
        vae.eval()
        test_joint_loss = 0
        test_image_loss = 0
        test_text_loss = 0

        for batch_idx, (image, text) in enumerate(test_loader):
            if args.cuda:
                image, text = image.cuda(), text.cuda()
            image, text = Variable(image), Variable(text)
                
            recon_image_1, recon_text_1, mu_1, logvar_1 = vae(image, text)
            recon_image_2, _, mu_2, logvar_2 = vae(image=image)
            _, recon_text_3, mu_3, logvar_3 = vae(text=text)
            
            loss_1 = joint_loss_function(recon_image_1, image, recon_text_1, text, mu_1, logvar_1,
                                         kl_lambda=kl_lambda, lambda_xy=args.lambda_xy, lambda_yx=args.lambda_yx,
                                         scramble=args.scramble)
            loss_2 = image_loss_function(recon_image_2, image, mu_2, logvar_2,
                                         kl_lambda=kl_lambda, lambda_x=args.lambda_x)
            loss_3 = text_loss_function(recon_text_3, text, mu_3, logvar_3,
                                        kl_lambda=kl_lambda, lambda_y=args.lambda_y, scramble=args.scramble)

            test_joint_loss += loss_1.data[0]
            test_image_loss += loss_2.data[0]
            test_text_loss += loss_3.data[0]

        test_loss = test_joint_loss + test_image_loss + test_text_loss
        test_joint_loss /= len(test_loader.dataset)
        test_image_loss /= len(test_loader.dataset)
        test_text_loss /= len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        
        print('====> Test Epoch\tJoint loss: {:.4f}\tImage loss: {:.4f}\tText loss:{:.4f}'.format(
            test_joint_loss, test_image_loss, test_text_loss))
        
        return test_loss, (test_joint_loss, test_image_loss, test_text_loss)


    kl_lambda = 1e-3
    # schedule = iter([5e-5, 1e-4, 5e-4, 1e-3])
    schedule = iter([0, 1e-3, 1e-2, 1e-1, 1.0])
    best_loss = sys.maxint
    for epoch in range(1, args.epochs + 1):
        if (epoch - 1) % 5 == 0 and args.anneal_kl:
            try:
                kl_lambda = next(schedule)
            except:
                pass

        train(epoch)
        loss, (joint_loss, image_loss, text_loss) = test()
        
        if args.anneal_lr:
            adjust_learning_rate(optimizer, epoch)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint({
            'state_dict': vae.state_dict(),
            'best_loss': best_loss,
            'joint_loss': joint_loss,
            'image_loss': image_loss,
            'text_loss': text_loss,
            'n_latents': args.n_latents,
            'scramble': args.scramble,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder='./trained_models')   

        sample = Variable(torch.randn(64, args.n_latents))
        if args.cuda:
           sample = sample.cuda()

        image_sample = vae.image_decoder(sample).cpu().data
        save_image(image_sample.view(64, 1, 50, 50),
                   './results/sample_image_epoch%d.png' % epoch)

        text_sample = vae.text_decoder.generate(sample).cpu().data.long()
        sample_texts = []
        for i in xrange(sample.size(0)):
            text = tensor_to_string(text_sample[i])
            sample_texts.append(text)

        with open('./results/sample_text_epoch%d.txt' % epoch, 'w') as fp:
            for text in sample_texts:
                fp.write('%s\n' % text)
