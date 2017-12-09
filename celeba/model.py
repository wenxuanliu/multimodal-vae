from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from datasets import N_ATTRS


class MultimodalVAE(nn.Module):
    def __init__(self, n_latents=20, use_cuda=False):
        super(MultimodalVAE, self).__init__()
        self.image_encoder = ImageEncoder(n_latents)
        self.image_decoder = ImageDecoder(n_latents)
        self.attrs_encoder = AttributeEncoder(n_latents)
        self.attrs_decoder = AttributeDecoder(n_latents)
        self.experts = ProductOfExperts()

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu

    def forward(self, image=None, attrs=None):
        assert image is not None or attrs is not None
        if image is not None and attrs is not None:
            # compute separate gaussians per modality
            image_mu, image_logvar = self.image_encoder(image)
            attrs_mu, attrs_logvar = self.attrs_encoder(attrs)
            mu = torch.stack((image_mu, attrs_mu), dim=0)
            logvar = torch.stack((image_logvar, attrs_logvar), dim=0)
        elif image is not None:
            mu, logvar = self.image_encoder(image)
            mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        elif attrs is not None:
            mu, logvar = self.attrs_encoder(attrs)
            mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)

        mu, logvar = self.experts(mu, logvar)
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        image_recon = self.image_decoder(z)
        attrs_recon = self.attrs_decoder(z)
        return image_recon, attrs_recon, mu, logvar


class ImageVAE(nn.Module):
    def __init__(self, n_latents=20):
        super(ImageVAE, self).__init__()
        self.encoder = ImageEncoder(n_latents)
        self.decoder = ImageDecoder(n_latents)
        self.n_latents = n_latents

    def encode(self, x):
        return self.encoder(x)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


class ImageEncoder(nn.Module):
    """This task is quite a bit harder than MNIST so we probably need 
    to use an RNN of some form. This will be good to get us ready for
    natural images.

    :param n_latents: size of latent vector
    """
    def __init__(self, n_latents):
        super(ImageEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            Swish(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 1024),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, n_latents * 2)
        )
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class ImageDecoder(nn.Module):
    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.upsample = nn.Sequential(
            nn.Linear(n_latents, 256 * 5 * 5),
            Swish(),
        )
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
        )

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsample(z)
        z = z.view(-1, 256, 5, 5)
        z = self.hallucinate(z)
        return F.sigmoid(z)


class AttributeEncoder(nn.Module):
    def __init__(self, n_latents):
        super(AttributeEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N_ATTRS, 64),
            nn.BatchNorm1d(64),
            Swish(),
            nn.Linear(64, n_latents * 2)
        )
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.net(x)
        return x[:, :n_latents], x[:, n_latents:]


class AttributeDecoder(nn.Module):
    def __init__(self, n_latents):
        super(AttributeDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latents, 64),
            nn.BatchNorm1d(64),
            Swish(),
            nn.Linear(64, N_ATTRS),
        )

    def forward(self, z):
        z = self.net(z)
        # not a one-hotted prediction: this predicts
        # 0 or 1 for every single index
        return F.sigmoid(z)


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


def swish(x):
    return x * F.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)
