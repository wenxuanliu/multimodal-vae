"""Test the MMVAE trained on MNIST on the MNIST task: 
Given the test set of images, generate the text and see if 
it is the correct label. Report the accuracy score.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from train import load_checkpoint


def test_mnist(model, loader, use_cuda=False, verbose=True):
    """Functionalize this so we can call it from our weakly-supervised experiments."""
    model.eval()
    correct = 0
    for image, text in loader:
        if use_cuda:
            image, text = image.cuda(), text.cuda()
        image, text = Variable(image, volatile=True), Variable(text)
        image = image.view(-1, 784)
        
        _, recon_text, _, _ = model(image=image)
        pred = recon_text.data.max(1, keepdim=True)[1]
        correct += pred.eq(text.data.view_as(pred)).cpu().sum()

    if verbose:
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(loader.dataset), 100. * correct / len(loader.dataset)))

    return correct / float(len(loader.dataset))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model file')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()


    # loader for MNIST
    loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=128, shuffle=True)

    vae = load_checkpoint(args.model_path, use_cuda=args.cuda)
    vae.eval()

    test_mnist(vae, loader, use_cuda=args.cuda)
