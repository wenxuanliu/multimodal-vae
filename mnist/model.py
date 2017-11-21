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
    def __init__(self):
        super(PixelCNN, self).__init__()
        self.net = nn.Sequential(
            MaskedConv2d('A', 1,  64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.Conv2d(64, 256, 1),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class RGBPixelCNN(nn.Module):
    def __init__(self):
        super(PixelCNN, self).__init__()
        self.net = nn.Sequential(
            RGBMaskedConv2d('A', 3,  64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            RGBMaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            RGBMaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            RGBMaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            RGBMaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            RGBMaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            RGBMaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            RGBMaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.Conv2d(64, 256 * 3, 1),  # RGB needs 256 times 3
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 256, 3, 64, 64)  # give it RGB channels
        return x


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class RGBMaskedConv2d(nn.Conv2d):
    # Adapted from http://sergeiturukin.com/2017/02/22/pixelcnn.html & 
    # https://github.com/rampage644/wavenet/blob/master/wavenet/models.py
    def __init__(self, mask_type, *args, **kwargs):
        super(RGBMaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        cout, cin, kh, kw = self.weight.size()
        yc, xc = kh // 2, kw // 2
        
        # initialize at all 1s
        self.mask.fill_(1)

        # create the mask in NumPy because we get so much better
        # indexing/slicing. We don't need this part to be differentiable anyway.
        mask_np = self.mask.numpy()

        # context masking: subsequent pixels won't have access to next pixels
        mask_np[:, :, yc+1:, :] = 0
        mask_np[:, :, yc:, xc+1:] = 0

        def bmask(i_out, i_in):
            # same pixel masking - pixel won't access next color (conv filter dim)
            cout_idx = np.expand_dims(np.arange(Cout) % 3 == i_out, 1)
            cin_idx = np.expand_dims(np.arange(Cin) % 3 == i_in, 0)
            a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
            return a1 * a2

        for j in xrange(3):
            mask_np[bmask(j, j), yc, xc] = 1 - (mask_type == 'A')

        mask_np[bmask(0, 1), yc, xc] = 0
        mask_np[bmask(0, 2), yc, xc] = 0
        mask_np[bmask(1, 2), yc, xc] = 0

        self.mask = torch.from_numpy(mask_np).float()

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
