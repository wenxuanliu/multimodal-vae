"""Compute log-likelihoods for entire dataset"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from train import load_checkpoint


def compute_nll(model, loader, image_only=False, text_only=False, 
                n_samples=1, use_cuda=False):
    assert not (image_only and text_only)

    model.eval()
    test_image_nll, test_text_nll = 0, 0

    for batch_idx, (image, text) in enumerate(loader):
        if use_cuda:
            image, text = image.cuda(), text.cuda()
        image = Variable(image, volatile=True)
        text = Variable(text, volatile=True)
        image = image.view(-1, 784)

        if not image_only and not text_only:
            _, _, mu, logvar = model(image, text)
        elif image_only:
            _, _, mu, logvar = model(image=image)
        elif text_only:
            _, _, mu, logvar = model(text=text)

        batch_size, n_latents = mu.size(0), mu.size(1)
        sample = Variable(torch.randn(n_samples, n_latents))

        if use_cuda:
            sample = sample.cuda()

        std = logvar.mul(0.5).exp_()
        mu = mu.unsqueeze(1).repeat(1, n_samples, 1)
        std = std.unsqueeze(1).repeat(1, n_samples, 1)

        sample = sample.unsqueeze(0).repeat(batch_size, 1, 1)
        sample = sample.mul(std).add_(mu)

        image_nll, text_nll = 0, 0
        for i in xrange(n_samples):
            recon_image = model.decode_image(sample[:, i])
            recon_text = model.decode_text(sample[:, i])
            image_nll += F.binary_cross_entropy(recon_image, image, size_average=False)
            text_nll += F.nll_loss(recon_text, text)

        test_image_nll += (image_nll / n_samples)
        test_text_nll += (text_nll / n_samples)

        print('Evaluating: [{}/{} ({:.0f}%)]'.format(batch_idx * len(image), len(loader.dataset),
                                                     100. * batch_idx / len(loader)))

    return -test_image_nll, -test_text_nll


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model file')
    parser.add_argument('--image_only', action='store_true', default=False,
                        help='compute NLL of test data using reconstructions from image only')
    parser.add_argument('--text_only', action='store_true', default=False,
                        help='compute NLL of test data using reconstructions from text only')
    parser.add_argument('--n_samples', type=int, default=100, 
                        help='number of samples to use to estimate the ELBO')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    assert not (args.image_only and args.text_only), \
        "--image_only and --text_only cannot both be supplied."

    # loader for MNIST
    loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=64, shuffle=True)

    vae = load_checkpoint(args.model_path, use_cuda=args.cuda)
    vae.eval()

    image_nll, text_nll = compute_nll(vae, loader, use_cuda=args.cuda, n_samples=args.n_samples,
                                      image_only=args.image_only, text_only=args.text_only)

    image_nll = image_nll.cpu().data[0]
    text_nll = text_nll.cpu().data[0]
    print('\nTest Image NLL: {:.4f}\tTest Text NLL: {:.4f}'.format(image_nll, text_nll))
    
