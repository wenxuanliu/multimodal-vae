"""Get insights into the latent space by using the generator 
to plot reconstructions at the positions in the latent space 
from which they have been generated.

This requires the latent space to be of dimension 2 so its 
really only applicable for MNIST...
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
from torch.autograd import Variable

import numpy as np
from train_imageonly import load_checkpoint


def array2d_string(A):
    return '\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in A])


if __name__ == "__main__":
    import argparse

    import os
    import pickle
    import argparse
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model file')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    vae = load_checkpoint(args.model_path, use_cuda=args.cuda)
    vae.eval()
    
    if args.cuda:
        vae.cuda()

    nx = ny = 20
    x_values = torch.linspace(-3, 3, nx)
    y_values = torch.linspace(-3, 3, ny)

    image_canvas = torch.zeros((28 * ny, 28 * nx))

    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            data = Variable(torch.Tensor([[xi, yi]]))
            if args.cuda:
                data = data.cuda()

            recon_image = vae.decode(data)
            image_canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = \
                recon_image[0].cpu().data.reshape(28, 28)

    if not os.path.exists('./results'):
        os.makedirs('./results')

    plt.figure(figsize=(8, 10))        
    Xi, Yi = np.meshgrid(x_values.numpy(), y_values.numpy())
    plt.imshow(image_canvas.numpy(), origin="upper", cmap="gray")
    plt.tight_layout()
    plt.savefig('./results/latent_viz_image.png')

