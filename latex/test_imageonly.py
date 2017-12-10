"""Test the MMVAE trained on MultiMNIST in predicting 
1) the number of characters in a MultiMNIST image and 2) 
the characters in the image themselves. Report accuracy score.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

import datasets
from train_imageonly import load_checkpoint
from train_imageonly import loss_function


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model file')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    preprocess_render = transforms.Compose([transforms.Scale(64),
                                            transforms.CenterCrop(64),
                                            transforms.ToTensor()])
    preprocess_formula = datasets.string_to_tensor
    loader = torch.utils.data.DataLoader(
        datasets.Image2Latex(partition='test', 
                             render_transform=preprocess_render,
                             formula_transform=preprocess_formula),
        batch_size=128, shuffle=True)

    vae = load_checkpoint(args.model_path, use_cuda=args.cuda)
    vae.eval()
    if args.cuda:
        vae.cuda()

    loss = 0
    for batch_idx, (data, _) in enumerate(loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_data, mu, logvar = vae(data)
        loss += loss_function(recon_data, data, mu, logvar, 
                              kl_lambda=1.).data[0]

    loss /= len(loader)
    print('====> Test Epoch\tLoss: {:.4f}'.format(loss))

