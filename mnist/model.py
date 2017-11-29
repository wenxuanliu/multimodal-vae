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


# -- Begin InfoVAE section --


class InfoVAE(nn.Module):
    def __init__(self, n_latents=20):
        super(InfoVAE, self).__init__()
        self.n_latents = n_latents
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, 2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(128 * 5 * 5, 1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, n_latents * 2)   
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(n_latents, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128 * 5 * 5),
            nn.ReLU(True),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2),
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_fc(x)
        n_latents = self.n_latents
        return x[:, :n_latents], x[:, n_latents:]

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = self.decoder_fc(z)
        z = z.view(-1, 128, 5, 5)
        z = self.decoder_conv(z)
        return F.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), z


def compute_kernel(x, y):
    """Apply Gaussian kernel to the i-th vector of x and j-th vector of y.

    :param x: torch.Tensor (x_size, dim)
    :param y: torch.Tensor (y_size, dim)
    """
    x_size, y_size, dim = x.size(0), y.size(0), x.size(1)
    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)
    return torch.exp(-torch.mean(torch.pow(tiled_x - tiled_y, 2), dim=2) / dim)


def compute_mmd(x, y):
    """Compute maximum mean discrepancy."""
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)


# -- Begin PixelCNN Section -- 


class GatedPixelCNN(nn.Module):
    """Improved PixelCNN with blind spot and gated blocks."""
    def __init__(self, data_channels=1, out_dims=256):
        super(GatedPixelCNN, self).__init__()
        self.conv1 = GatedResidualBlock(data_channels, 128, 7, 'A')
        self.blocks = GatedResidualBlockList(5, 128, 128, 3, 'B')
        self.conv2 = MaskedConv2d('B', data_channels, 128, 16, 1)
        self.conv4 = MaskedConv2d('B', data_channels, 16, out_dims * data_channels, 1)
        self.data_channels = data_channels
        self.out_dims = out_dims

    def forward(self, x):
        x, h = self.conv1(x, x)
        _, h = self.blocks(x, h)
        h = self.conv2(F.relu(h))
        h = self.conv4(F.relu(h))

        batch_size, _, height, width = h.size()
        h = h.view(batch_size, self.out_dims, self.data_channels, 
                   height, width)
        return h


class GatedResidualBlockList(nn.Module):
    def __init__(self, block_num, *args, **kwargs):
        super(GatedResidualBlockList, self).__init__()
        blocks = [GatedResidualBlock(*args, **kwargs) for i in xrange(block_num)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, h):
        for block in self.blocks:
            x_, h_ = block(x, h)
            x, h = x_, h + h_

        return x, h


class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, 
                 data_channels=1):
        super(GatedResidualBlock, self).__init__()
        self.vertical_conv = CroppedConv2d(in_channels, 2 * out_channels, 
                                           kernel_size=(kernel_size // 2 + 1, kernel_size),
                                           padding=(kernel_size // 2 + 1, kernel_size // 2))
        self.x_to_h_conv = MaskedConv2d(mask_type, data_channels, 2 * out_channels, 
                                        2 * out_channels, 1)
        self.vertical_gate_conv = nn.Conv2d(2 * out_channels, 2 * out_channels, 1)
        self.horizontal_conv = CroppedConv2d(in_channels, 2 * out_channels, 
                                             kernel_size=(1, kernel_size // 2 + 1), 
                                             padding=(0, kernel_size // 2 + 1))
        self.horizontal_gate_conv = nn.Conv2d(2 * out_channels, 2 * out_channels, 1)
        self.horizontal_output = MaskedConv2d(mask_type, data_channels, out_channels, 
                                              out_channels, 1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.data_channels = data_channels

    def forward(self, x, h):
        x = self.vertical_conv(x)
        to_vertical = self.x_to_h_conv(x)

        x_t, x_s = torch.split(self.vertical_gate_conv(x), self.out_channels, dim=1)
        x = F.tanh(x_t) * F.sigmoid(x_s)

        h_ = self.horizontal_conv(h)
        h_t, h_s = torch.split(self.horizontal_gate_conv(h_ + to_vertical), 
                               self.out_channels, dim=1)
        h = self.horizontal_output(F.tanh(h_t) * F.sigmoid(h_s))
        return x, h


class PixelCNN(nn.Module):
    def __init__(self, data_channels=1, out_dims=256):
        super(PixelCNN, self).__init__()
        self.conv1 = MaskedConv2d('A', data_channels, data_channels, 128, 7, padding=3)
        self.blocks = ResidualBlockList(15, 128, 128, 3, 'B')
        self.conv2 = MaskedConv2d('B', data_channels, 128, 16, 1)
        self.conv4 = MaskedConv2d('B', data_channels, 16, out_dims * data_channels, 1)
        self.data_channels = data_channels
        self.out_dims = out_dims

    def forward(self, x):
        h = self.conv1(x)
        h = self.blocks(h)
        h = self.conv2(F.relu(h))
        h = self.conv4(F.relu(h))

        batch_size, _, height, width = h.size()
        h = h.view(batch_size, self.out_dims, self.data_channels, height, width)
        return h 


class ResidualBlockList(nn.Module):
    def __init__(self, block_num, *args, **kwargs):
        super(ResidualBlockList, self).__init__()
        blocks = [ResidualBlock(*args, **kwargs) for i in xrange(block_num)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        return h


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type,
                 data_channels=1): 
        super(ResidualBlock, self).__init__()
        self.conv1 = MaskedConv2d(mask_type, data_channels, in_channels, in_channels // 2, 1)
        self.conv2 = MaskedConv2d(mask_type, data_channels, in_channels // 2, in_channels // 2, 
                                  3, padding=1)
        self.conv3 = MaskedConv2d(mask_type, data_channels, in_channels // 2, in_channels, 1)

    def forward(self, x):
        h = self.conv1(F.relu(x))
        h = self.conv2(F.relu(h))
        h = self.conv3(F.relu(h))
        return x + h


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
        if kh > 1 and kw > 1:
            self.mask[:, :, yc, xc + (mask_type == 'B'):] = 0
            self.mask[:, :, yc + 1:] = 0

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
    return soft_max_nd.transpose(dim, len(input_size) - 1)


def cross_entropy_by_dim(input, output, dim=1):
    input_size = input.size()
    output_size = output.size()

    trans_input = input.permute(0, 2, 3, 4, 1)
    trans_input_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_input_size[-1])
    output_2d = output.contiguous().view(-1)
    return F.cross_entropy(input_2d, output_2d)


class PixelCNNv2(nn.Module):
    def __init__(self, n_groups=7, data_channels=1, out_dims=256):
        super(PixelCNNv2, self).__init__()
        self.mask_A = nn.Sequential(
            MaskedConv2d('A', data_channels, data_channels, 128, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
        )
        self.mask_B = []
        for i in xrange(n_groups):
            self.mask_B += [
                MaskedConv2d('B', data_channels, 128, 128, 7, 1, 3, bias=False), 
                nn.BatchNorm2d(128), 
                nn.ReLU(True),
            ]
        self.mask_B = nn.Sequential(*mask_B)
        self.net = nn.Conv2d(128, out_dims * data_channels, 1)
        self.data_channels = data_channels
        self.out_dims = out_dims

    def forward(self, x):
        x = self.mask_A(x)
        x = self.mask_B(x)
        x = self.net(x)
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, self.out_dims, self.data_channels, height, width)
        return x

