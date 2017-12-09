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
from train import load_checkpoint
from train import loss_function


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model file')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    preprocess_data = transforms.Compose([transforms.Scale(64),
                                          transforms.CenterCrop(64),
                                          transforms.ToTensor()])

    loader = torch.utils.data.DataLoader(
        datasets.CelebAttributes(partition='test',
                                 image_transform=preprocess_data),
        batch_size=args.batch_size, shuffle=True)

    vae = load_checkpoint(args.model_path, use_cuda=args.cuda)
    vae.eval()
    if args.cuda:
        vae.cuda()

    joint_loss = 0
    image_loss = 0
    attrs_loss = 0

    for batch_idx, (image, attrs) in enumerate(loader):
        if args.cuda:
            image, attrs = image.cuda(), attrs.cuda()
        image = Variable(image, volatile=True)
        attrs = Variable(attrs, volatile=True)

        recon_image_1, recon_attrs_1, mu_1, logvar_1 = vae(image=image, attrs=attrs)
        recon_image_2, recon_attrs_2, mu_2, logvar_2 = vae(image=image)
        recon_image_3, recon_attrs_3, mu_3, logvar_3 = vae(attrs=attrs)

        joint_loss += loss_function(mu_1, logvar_1, recon_x=recon_image_1, x=image, 
                                    recon_y=recon_attrs_1, y=attrs, kl_lambda=1., 
                                    lambda_x=1., lambda_y=1.).data[0]
        image_loss += loss_function(mu_2, logvar_2, recon_x=recon_image_2, x=image, 
                                    recon_y=recon_attrs_2, y=attrs, kl_lambda=1., 
                                    lambda_x=1., lambda_y=1.).data[0]
        attrs_loss += loss_function(mu_3, logvar_3, recon_x=recon_image_3, x=image, 
                                    recon_y=recon_attrs_3, y=attrs, kl_lambda=1., 
                                    lambda_x=1., lambda_y=1.).data[0]
    joint_loss /= len(loader)
    image_loss /= len(loader)
    attrs_loss /= len(loader)
    print('====> Test Epoch\tJoint loss: {:.4f}\tImage loss: {:.4f}\tAttrs loss:{:.4f}'.format(
            joint_loss, image_loss, attrs_loss))
