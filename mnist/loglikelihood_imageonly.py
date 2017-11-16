"""Compute log-likelihoods for test dataset 
but for image_only VAE.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from train import load_checkpoint


def compute_nll(model, loader, n_samples=1, use_cuda=False):
    model.eval()
    test_nll = 0

    for batch_idx, (image, _) in enumerate(loader):
        if use_cuda:
            image = image.cuda()
        image = Variable(image, volatile=True)
        image = image.view(-1, 784)

        _, mu, logvar = model(image)
        batch_size, n_latents = mu.size(0), mu.size(1)
        sample = Variable(torch.randn(n_samples, n_latents))

        if use_cuda:
            sample = sample.cuda()

        std = logvar.mul(0.5).exp_()
        mu = mu.unsqueeze(1).repeat(1, n_samples, 1)
        std = std.unsqueeze(1).repeat(1, n_samples, 1)

        sample = sample.unsqueeze(0).repeat(batch_size, 1, 1)
        sample = sample.mul(std).add_(mu)

        nll = 0
        for i in xrange(n_samples):
            recon_image = model.decode_image(sample[:, i])
            nll += F.binary_cross_entropy(recon_image, image, size_average=False)

        test_nll += (nll / n_samples)
        print('Evaluating: [{}/{} ({:.0f}%)]'.format(batch_idx * len(image), len(loader.dataset),
                                                     100. * batch_idx / len(loader)))   

    test_nll /= len(loader.dataset)
    return test_nll


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model file')
    parser.add_argument('--n_samples', type=int, default=100, 
                        help='number of samples to use to estimate the ELBO')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # loader for MNIST
    loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=64, shuffle=True)

    vae = load_checkpoint(args.model_path, use_cuda=args.cuda)
    vae.eval()

    nll = compute_nll(vae, loader, use_cuda=args.cuda, n_samples=args.n_samples)
    print('\nTest NLL: {:.4f}'.format(nll))
