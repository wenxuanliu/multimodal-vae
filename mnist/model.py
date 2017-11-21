from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class MultimodalVAE(nn.Module):
    def __init__(self, n_latents=20):
        super(MultimodalVAE, self).__init__()
        self.image_encoder = ImageEncoder(n_latents)
        self.image_decoder = ImageDecoder(n_latents)
        self.text_encoder = TextEncoder(n_latents)
        self.text_decoder = TextDecoder(n_latents)
        self.experts = ProductOfExperts()
        self.n_latents = n_latents

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu

    def encode_image(self, x):
        return self.image_encoder(x)

    def decode_image(self, x):
        return self.image_decoder(x)

    def encode_text(self, x):
        return self.text_encoder(x)

    def decode_text(self, x):
        return self.text_decoder(x)

    def prior(self, size, use_cuda=False):
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.log(torch.ones(size)))
        if use_cuda:
            mu = mu.cuda()
            logvar = logvar.cuda()

        return mu, logvar

    def forward(self, image=None, text=None):
        # can't just put nothing
        assert image is not None or text is not None
        
        if image is not None and text is not None:
            # compute separate gaussians per modality
            image_mu, image_logvar = self.encode_image(image)
            text_mu, text_logvar = self.encode_text(text)
            mu = torch.stack((image_mu, text_mu), dim=0)
            logvar = torch.stack((image_logvar, text_logvar), dim=0)
        elif image is not None:
            mu, logvar = self.encode_image(image)
            mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        elif text is not None:
            mu, logvar = self.encode_text(text)
            mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        
        # add p(z) as an expert; regularizes for missing modalities
        # https://arxiv.org/pdf/1705.10762.pdf
        # prior_mu, prior_logvar = self.prior((1, mu.size(1), mu.size(2)),
        #                                      use_cuda=mu.is_cuda)
        # mu = torch.cat((mu, prior_mu), dim=0)
        # logvar = torch.cat((logvar, prior_logvar), dim=0)
        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)        
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        image_recon = self.decode_image(z)
        text_recon = self.decode_text(z)

        return image_recon, text_recon, mu, logvar

    def gen_latents(self, image, text):
        # compute separate gaussians per modality
        image_mu, image_logvar = self.encode_image(image)
        text_mu, text_logvar = self.encode_text(text)
        mu = torch.stack((image_mu, text_mu), dim=0)
        logvar = torch.stack((image_logvar, text_logvar), dim=0)
        
        # grab learned mixture weights and sample
        mu, logvar = self.experts(mu, logvar)
        z = self.reparametrize(mu, logvar)
        return z


class ImageEncoder(nn.Module):
    """MNIST doesn't need CNN, so use a lightweight FNN"""
    def __init__(self, n_latents):
        super(ImageEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, n_latents * 2),
        )
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.net(x)
        return x[:, :n_latents], x[:, n_latents:]


class ImageDecoder(nn.Module):
    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latents, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 784),
        )

    def forward(self, z):
        z = self.net(z)
        return F.sigmoid(z)


class TextEncoder(nn.Module):
    """MNIST has a vocab of size 10 words."""
    def __init__(self, n_latents):
        super(TextEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Embedding(10, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, n_latents * 2)
        )
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.net(x)
        return x[:, :n_latents], x[:, n_latents:]


class TextDecoder(nn.Module):
    """Project back into 10 dimensions and use softmax 
    to pick the word."""
    def __init__(self, n_latents):
        super(TextDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latents, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 10),
        )

    def forward(self, z):
        z = self.net(z)
        return F.log_softmax(z)


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    :param mu: M x D for M experts
    :param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        pd_mu = torch.sum(mu * var, dim=0) / torch.sum(var, dim=0)
        pd_var = 1 / torch.sum(1 / var, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


class VAE(nn.Module):
    def __init__(self, n_latents=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, n_latents * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_latents, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 784),
        )
        self.n_latents = n_latents

    def encode(self, x):
        n_latents = self.n_latents
        x = self.encoder(x)
        return x[:, :n_latents], x[:, n_latents:]

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        z = self.decoder(z)
        return F.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class PixelCNN(nn.Module):
    def __init__(self, data_channels=1):
        super(PixelCNN, self).__init__()
        self.net = nn.Sequential(
            MaskedConv2d('A', data_channels, data_channels, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            MaskedConv2d('B', data_channels, 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            MaskedConv2d('B', data_channels, 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            MaskedConv2d('B', data_channels, 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            MaskedConv2d('B', data_channels, 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            MaskedConv2d('B', data_channels, 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            MaskedConv2d('B', data_channels, 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            MaskedConv2d('B', data_channels, 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.Conv2d(64, 256 * data_channels, 1), 
        )
        self.data_channels = data_channels

    def forward(self, x):
        data_channels = self.data_channels
        x = self.net(x)
        n, c, h, w = x.size()
        x = x.view(n, c // data_channels, data_channels, h, w)
        x = log_softmax_by_dim(x, dim=1)
        return x


class GatedPixelCNN(nn.Module):
    """Improved PixelCNN with blind spot and gated blocks."""
    def __init__(self, data_channels=1):
        super(GatedPixelCNN, self).__init__()
        self.conv1 = ResidualBlock(data_channels, 64, 7, 'A')
        self.blocks = ResidualBlockList(5, 64, 64, 3, 'B')
        self.conv2 = MaskedConv2d(64, 64, 1)
        self.conv4 = MaskedConv2d(64, 256 * data_channels, 1)
        self.data_channels = data_channels

    def forward(self, x):
        x, h = self.conv1(x, x)
        _, h = self.blocks(x, h)
        h = self.conv2(F.relu(h))
        h = self.conv4(F.relu(h))

        batch_size, _, height, width = h.size()
        h = h.view(batch_size, 256, data_channels, height, width)
        h = log_softmax_by_dim(h, dim=1)
        return h


class ResidualBlockList(nn.Module):
    def __init__(self, block_num, *args, **kwargs):
        super(ResidualBlockList, self).__init__()
        self.blocks = [ResidualBlock(*args, **kwargs) for i in range(block_num)]

    def forward(self, x, h):
        for block in self.blocks:
            x_, h_ = block(x, h)
            x, h = x_, h + h_

        return x, h


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, 
                 data_channels=1):
        super(ResidualBlock, self).__init__()
        self.vertical_conv_t = CroppedConv2d(in_channels, out_channels, 
                                             kernel_size=(kernel_size // 2 + 1, kernel_size),
                                             padding=(kernel_size // 2 + 1, kernel_size // 2))
        self.vertical_conv_s = CroppedConv2d(in_channels, out_channels, 
                                             kernel_size=(kernel_size // 2 + 1, kernel_size),
                                             padding=(kernel_size // 2 + 1, kernel_size // 2))
        self.x_to_h_conv_t = Conv2d(out_channels, out_channels, 1)
        self.x_to_h_conv_s = Conv2d(out_channels, out_channels, 1)
        self.horizontal_conv_t = MaskedConv2d(mask_type, data_channels, in_channels, out_channels, 
                                              kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        self.horizontal_conv_s = MaskedConv2d(mask_type, data_channels, in_channels, out_channels, 
                                              kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        self.horizontal_output = MaskedConv2d(mask_type, data_channels, out_channels, out_channels, 1)

    def forward(self, x, h):
        x_t = self.vertical_conv_t(x)
        x_s = self.vertical_conv_s(x)
        x = F.tanh(x_t) * F.sigmoid(x_s)

        to_vertical_t = self.x_to_h_conv_t(x_t)
        to_vertical_s = self.x_to_h_conv_s(x_s)
        h_t = self.horizontal_conv_t(h)
        h_s = self.horizontal_conv_s(h)

        h_t, h_s = h_t + to_vertical_t, h_s + to_vertical_s
        h = self.horizontal_output(F.tanh(h_t) * F.sigmoid(h_s))
        return x, h


class MaskedConv2d(nn.Conv2d):
    # Adapted from https://github.com/igul222/pixel_rnn/blob/master/pixel_rnn.py-0
    def __init__(self, mask_type, data_channels, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        cout, cin, kh, kw = self.weight.size()
        yc, xc = kh // 2, kw // 2
        
        # initialize at all 1s
        self.mask.fill_(1)

        for i in xrange(kh):
            for j in xrange(kw):
                if (j > xc) or (j == xc and i > yc):
                    self.mask[:, :, j, i] = 0.

        for i in xrange(data_channels):
            for j in xrange(data_channels):
                if (mask_type == 'A' and i >= j) or (mask_type == 'B' and i > j):
                    self.mask[j::data_channels, i::data_channels, yc, xc] = 0

        self.mask_type = mask_type
        self.data_channels = data_channels
        
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class CroppedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(CroppedConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = super(CroppedConv2d, self).forward(x)
        _, _, kh, kw = self.weight.size()
        pad_h, pad_w = self.padding
        h_crop = -(kh + 1) if pad_h == kh else None
        w_crop = -(kw + 1) if pad_w == kw else None
        return x[:, :, :h_crop, :w_crop]


def log_softmax_by_dim(input, dim=1):
    input_size = input.size()
    trans_input = input.transpose(dim, len(input_size) - 1)
    trans_size = trans_input.size()
    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.log_softmax(input_2d)
    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(dim, len(input_size)-1)
