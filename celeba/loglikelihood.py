"""Compute log-likelihoods for entire dataset"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

import datasets
from train import load_checkpoint


def compute_nll(model, loader, image_only=False, attrs_only=False, 
                n_samples=1, use_cuda=False):
    assert not (image_only and attrs_only)

    model.eval()
    test_image_nll, test_attrs_nll = 0, 0

    for batch_idx, (image, attrs) in enumerate(loader):
        if use_cuda:
            image, attrs = image.cuda(), attrs.cuda()
        image = Variable(image, volatile=True)
        attrs = Variable(attrs, volatile=True)

        if not image_only and not attrs_only:
            _, _, mu, logvar = model(image, attrs)
        elif image_only:
            _, _, mu, logvar = model(image=image)
        elif attrs_only:
            _, _, mu, logvar = model(attrs=attrs)

        batch_size, n_latents = mu.size(0), mu.size(1)
        sample = Variable(torch.randn(n_samples, n_latents))

        if use_cuda:
            sample = sample.cuda()

        std = logvar.mul(0.5).exp_()
        mu = mu.unsqueeze(1).repeat(1, n_samples, 1)
        std = std.unsqueeze(1).repeat(1, n_samples, 1)

        sample = sample.unsqueeze(0).repeat(batch_size, 1, 1)
        sample = sample.mul(std).add_(mu)

        image_nll, attrs_nll = 0, 0
        for i in xrange(n_samples):
            recon_image = model.image_decoder(sample[:, i])
            recon_attrs = model.attrs_decoder(sample[:, i])
            image_nll += F.binary_cross_entropy(recon_image, image, size_average=False)
            attrs_nll += F.nll_loss(recon_attrs, attrs, size_average=False)

        test_image_nll += (image_nll / n_samples)
        test_attrs_nll += (attrs_nll / n_samples)

        print('Evaluating: [{}/{} ({:.0f}%)]'.format(batch_idx * len(image), len(loader.dataset),
                                                     100. * batch_idx / len(loader)))

    test_image_nll /= len(loader.dataset)
    test_attrs_nll /= len(loader.dataset)

    return test_image_nll, test_attrs_nll


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model file')
    # modality options
    parser.add_argument('--image_only', action='store_true', default=False,
                        help='compute NLL of test data using reconstructions from image only')
    parser.add_argument('--attrs_only', action='store_true', default=False,
                        help='compute NLL of test data using reconstructions from attributes only')
    parser.add_argument('--n_samples', type=int, default=100, 
                        help='number of samples to use to estimate the ELBO')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    assert not (args.image_only and args.attrs_only), \
        "--image_only and --attrs_only cannot both be supplied."

    # loader for MultiMNIST
    loader = torch.utils.data.DataLoader(
        datasets.CelebAttributes('./data', partition='test'),
        batch_size=64, shuffle=True)

    vae = load_checkpoint(args.model_path, use_cuda=args.cuda)
    vae.eval()
    if args.cuda:
        vae.cuda()

    image_nll, attrs_nll = compute_nll(vae, loader, use_cuda=args.cuda, n_samples=args.n_samples,
                                      image_only=args.image_only, attrs_only=args.attrs_only)

    image_nll = image_nll.cpu().data[0]
    attrs_nll = attrs_nll.cpu().data[0]
    print('\nTest Image NLL: {:.4f}\tTest Attrs NLL: {:.4f}'.format(image_nll, attrs_nll))
