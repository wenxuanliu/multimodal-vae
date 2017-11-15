"""Compute log-likelihoods for entire dataset"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from train import load_checkpoint


def compute_nll(model, loader, image_only=False, text_only=False, use_cuda=False):
    assert not (image_only and text_only)

    model.eval()
    image_nll, text_nll = 0, 0

    for image, text in loader:
        if use_cuda:
            image, text = image.cuda(), text.cuda()
        image = Variable(image, volatile=True)
        text = Variable(text, volatile=True)
        image = image.view(-1, 784)

        if not image_only and not text_only:
            recon_image, recon_text, _, _ = model(image, text)
        elif image_only:
            recon_image, recon_text, _, _ = model(image=image)
        elif text_only:
            recon_image, recon_text, _, _ = model(text=text)

        _image_nll = -torch.log(F.binary_cross_entropy(recon_image, image))
        _text_nll = -torch.log(F.nll_loss(recon_text, text))

        image_nll += _image_nll
        text_nll += _text_nll

    return image_nll, text_nll


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model file')
    parser.add_argument('--image_only', action='store_true', default=False,
                        help='compute NLL of test data using reconstructions from image only')
    parser.add_argument('--text_only', action='store_true', default=False,
                        help='compute NLL of test data using reconstructions from text only')
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
        batch_size=128, shuffle=True)

    vae = load_checkpoint(args.model_path, use_cuda=args.cuda)
    vae.eval()

    image_nll, text_nll = compute_nll(vae, loader, use_cuda=args.cuda, 
                                      image_only=args.image_only, text_only=args.text_only)

    image_nll = image_nll.cpu().data[0]
    text_nll = text_nll.cpu().data[0]
    print('Test Image NLL: {:.4f}\tTest Text NLL: {:.4f}'.format(image_nll, text_nll))
    
