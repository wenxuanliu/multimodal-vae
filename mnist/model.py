from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

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

    def forward(self, image, text):
        # compute separate gaussians per modality
        image_mu, image_logvar = self.encode_image(image)
        text_mu, text_logvar = self.encode_text(text)
        mu = torch.stack((image_mu, text_mu), dim=0)
        logvar = torch.stack((image_logvar, text_logvar), dim=0)
        
        # grab learned mixture weights and sample
        mu, logvar = self.experts(mu, logvar)
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

    def generate_given_image(self, image):
        mu, logvar = self.encode_image(image)
        z = self.reparametrize(mu, logvar)
        image_recon = self.decode_image(z)
        text_recon = self.decode_text(z)

        return image_recon, text_recon, mu, logvar

    def generate_given_text(self, text):
        mu, logvar = self.encode_text(text)
        z = self.reparametrize(mu, logvar)
        image_recon = self.decode_image(z)
        text_recon = self.decode_text(z)

        return image_recon, text_recon, mu, logvar


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
