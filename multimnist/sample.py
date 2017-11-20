"""Generate samples from trained model from all kinds of 
different ways.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

import datasets
from train import load_checkpoint
from utils import char_tensor, charlist_tensor, tensor_to_string

EMPTY = '{}'


def fetch_multimnist_image(_label):
    if _label == EMPTY:
        _label = ''

    loader = torch.utils.data.DataLoader(
        datasets.MultiMNIST('./data', train=False, download=True,
                            transform=transforms.ToTensor(),
                            target_transform=charlist_tensor),
        batch_size=1, shuffle=True)

    images = []
    for image, label in loader:
        if tensor_to_string(label.squeeze(0)) == _label:
            images.append(image)

    if len(images) == 0:
        sys.exit('No images with label (%s) found.' % _label)

    images = torch.cat(images).cpu().numpy()
    ix = np.random.choice(np.arange(images.shape[0]))
    image = images[ix]

    image = torch.from_numpy(image).float() 
    image = image.unsqueeze(0)
    return Variable(image, volatile=True)


def fetch_multimnist_text(label):
    if label == EMPTY:
        label = ''
    text = char_tensor(label).unsqueeze(0)
    return Variable(text, volatile=True)


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model file.')
    parser.add_argument('--n_samples', type=int, default=64, 
                        help='Number of images and texts to sample.')
    parser.add_argument('--condition_on_image', type=str, default=None,
                        help='If True, generate text conditioned on an image.')
    parser.add_argument('--condition_on_text', type=str, default=None, 
                        help='If True, generate images conditioned on a text.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    n_latents = torch.load(args.model_path)['n_latents']
    vae = load_checkpoint(args.model_path, use_cuda=args.cuda)
    vae.eval()
    if args.cuda:
        vae.cuda()

    # mode 1: unconditional generation
    if not args.condition_on_image and not args.condition_on_text:
        mu = Variable(torch.Tensor([0]))
        std = Variable(torch.Tensor([1]))
        if args.cuda:
            mu = mu.cuda()
            std = std.cuda()

    # mode 2: generate conditioned on image
    elif args.condition_on_image and not args.condition_on_text:
        image = fetch_multimnist_image(args.condition_on_image)
        if args.cuda:
            image = image.cuda()
        mu, logvar = vae.encode_image(image)
        std = logvar.mul(0.5).exp_()

    # mode 3: generate conditioned on text
    elif args.condition_on_text and not args.condition_on_image:
        text = fetch_multimnist_text(args.condition_on_text)
        if args.cuda:
            text = text.cuda()
        mu, logvar = vae.encode_text(text)
        std = logvar.mul(0.5).exp_()

    # mode 4: generate conditioned on image and text
    elif args.condition_on_text and args.condition_on_image:
        image = fetch_multimnist_image(args.condition_on_image)
        text = fetch_multimnist_text(args.condition_on_text)
        if args.cuda:
            image = image.cuda()
            text = text.cuda()
        image_mu, image_logvar = vae.encode_image(image)
        text_mu, text_logvar = vae.encode_text(text)
        mu = torch.stack((image_mu, text_mu), dim=0)
        logvar = torch.stack((image_logvar, text_logvar), dim=0)
        mu, logvar = vae.experts(mu, logvar)
        std = logvar.mul(0.5).exp_()

    # sample from uniform gaussian
    sample = Variable(torch.randn(args.n_samples, n_latents))
    if args.cuda:
        sample = sample.cuda()

    # sample from particular gaussian by multiplying + adding
    mu = mu.expand_as(sample)
    std = std.expand_as(sample)
    sample = sample.mul(std).add_(mu)

    # generate image and text
    image_recon = vae.decode_image(sample).cpu()
    text_recon = vae.decode_text(sample).cpu()
    text_recon = torch.max(text_recon, dim=2)[1]

    if not os.path.isdir('./results'):
        os.mkdirs('./results')

    # save image samples to filesystem
    save_image(image_recon.data.view(args.n_samples, 1, 50, 50),
               './results/sample_image.png')

    # save text samples to filesystem
    with open('./results/sample_text.txt', 'w') as fp:
        for i in xrange(text_recon.size(0)):
            text_recon_str = tensor_to_string(text_recon[i].data)
            fp.write('%s\n' % text_recon_str)
