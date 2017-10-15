from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from model import MultimodalVAE
from generator import ShuffleMNIST


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

    model = MultimodalVAE()
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def loss_function(recon_image, image, recon_text, text,
                  mu, logvar, elbo_lambda=1):
    image_BCE = F.binary_cross_entropy(recon_image, image.view(-1, 784))
    text_BCE = F.nll_loss(recon_text, text)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # TODO: which mu/logvar to use? Do I sample the mixture_model?
    return image_BCE + text_BCE + elbo_lambda * KLD


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--elbo_lambda', type=float, default=0.1,
                        help='BCE + elbo_lambda * KLD')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # create loaders for MNIST
    train_loader = torch.utils.data.DataLoader(
        ShuffleMNIST('./data/processed/training.pt',
                     transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        ShuffleMNIST('./data/processed/test.pt',
                     transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)

    # load multimodal VAE
    model = MultimodalVAE()
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    def train(epoch):
        model.train()
        loss_meter = AverageMeter()

        for batch_idx, (image, text) in enumerate(train_loader):
            if args.cuda:
                image, text = image.cuda(), text.cuda()
            image, text = Variable(image), Variable(text)
            optimizer.zero_grad()

            image = image.view(-1, 784)  # flatten image
            recon_image, recon_text, mu, logvar = model(image, text) 
            loss = loss_function(recon_image, image, recon_text, text, mu, logvar)
            loss.backward()

            loss_meter.update(loss.data[0], len(image))
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg))

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_meter.avg))


    def test():
        model.eval()
        test_loss = 0

        for batch_idx, (image, text) in enumerate(test_loader):
            if args.cuda:
                image, text = image.cuda(), text.cuda()
            image, text = Variable(image), Variable(text)

            recon_image, recon_text, mu, logvar = model(image, text) 
            loss = loss_function(recon_image, image, recon_text, text, mu, logvar)
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
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder='./trained_models')
            