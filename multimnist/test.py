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
from utils import max_length, FILL
from utils import charlist_tensor


def test_multimnist(model, loader, use_cuda=False, verbose=True):
    """Compute prediction accuracy on MultiMNIST; if scramble is True, the label
    is correct if any scramble of the text matches.

    :param model: trained MMVAE model
    :param loader: MultiMNIST loader
    :param use_cuda: if True, cast CUDA on Variables (default: False)
    :param verbose: if True, print more statuses (default: True)
    """
    model.eval()
    char_correct = 0
    len_correct = 0

    for image, text in loader:
        if use_cuda:
            image, text = image.cuda(), text.cuda()
        image = Variable(image, volatile=True)

        # reconstruct text from image
        _, recon_text, _, _ = model(image=image)
        pred = torch.max(recon_text.data, dim=2)[1].cpu().numpy()
        gt = text.cpu().numpy()

        char_correct += float(np.sum(pred == gt))
        len_correct += float(np.sum(np.sum(pred == FILL, axis=1) == 
                                    np.sum(gt == FILL, axis=1)))


    _char_correct = char_correct / (len(loader.dataset) * max_length)
    _len_correct = len_correct / len(loader.dataset)

    if verbose:
        print('\nTest set: Character Accuracy: {}/{} ({:.0f}%)\tLength Accuracy: {}/{} ({:.0f}%)\n'.format(
            int(char_correct), len(loader.dataset) * max_length, 100. * _char_correct, 
            int(len_correct), len(loader.dataset), 100. * _len_correct))

    return _char_correct, _len_correct


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model file')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    loader = torch.utils.data.DataLoader(
        datasets.MultiMNIST('./data', train=False, download=True,
                            transform=transforms.ToTensor(),
                            target_transform=charlist_tensor),
        batch_size=128, shuffle=True)

    vae = load_checkpoint(args.model_path, use_cuda=args.cuda)
    vae.eval()

    test_multimnist(vae, loader, use_cuda=args.cuda, verbose=True)
