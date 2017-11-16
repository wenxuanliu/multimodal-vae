from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

from train_imageonly import load_checkpoint


def fetch_mnist_image(_label):
    # find an example of the image in our dataset
    loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, 
                       transform=transforms.ToTensor()),
        batch_size=128, shuffle=False)
    # load all data into a single structure
    images, labels = [], []
    for batch_idx, (image, label) in enumerate(loader):
        image = image.view(-1, 784)
        images.append(image)
        labels.append(label)
    images = torch.cat(images).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    # take all the ones where it's an image of the correct label
    images = images[labels == _label]
    # randomly choose one
    image = images[np.random.choice(np.arange(images.shape[0]))]
    image = torch.from_numpy(image).float() 
    image = image.unsqueeze(0)
    return Variable(image, volatile=True)


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('n_samples', type=int, help='Number of images and texts to sample.')
    parser.add_argument('--condition_on_image', type=int, default=None,
                        help='If True, generate text conditioned on an image.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # load trained model
    vae = load_checkpoint('./trained_models/model_best.pth.tar', use_cuda=args.cuda)
    vae.eval()
    if args.cuda:
        vae.cuda()

    # mode 1: unconditional generation
    if not args.condition_on_image:
        mu = Variable(torch.Tensor([0]))
        std = Variable(torch.Tensor([1]))
    else:  # mode 2: generate conditioned on image
        image = fetch_mnist_image(args.condition_on_image)
        if args.cuda:
            image = image.cuda()
        mu, logvar = vae.encode_image(image)
        std = logvar.mul(0.5).exp_()

    # sample from uniform gaussian
    n_latents = vae.encoder.n_latents
    sample = Variable(torch.randn(args.n_samples, n_latents))
    if args.cuda:
        sample = sample.cuda()
    
    # sample from particular gaussian by multiplying + adding
    mu = mu.expand_as(sample)
    std = std.expand_as(sample)
    sample = sample.mul(std).add_(mu)
    # generate image and text
    image_recon = vae.decode(sample).cpu()

    if not os.path.isdir('./results'):
        os.mkdirs('./results')

    # save image samples to filesystem
    save_image(image_recon.data.view(args.n_samples, 1, 28, 28),
               './results/sample_image.png')

    # save text samples to filesystem
    with open('./results/sample_texts.txt', 'w') as fp:
        text_recon_np = text_recon.data.numpy()
        text_recon_np = np.argmax(text_recon_np, axis=1).tolist()
        for i, item in enumerate(text_recon_np):
            fp.write('Text (%d): %s\n' % (i, item))
