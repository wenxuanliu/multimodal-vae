"""This model will be quite similar to mnist/model.py 
except we will need to be slightly fancier in the 
encoder/decoders for each modality. Likely, we will need 
convolutions/deconvolutions and RNNs.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from utils import n_characters, max_length
from utils import SOS, FILL


class MultimodalVAE(nn.Module):
    def __init__(self, n_latents=20, use_cuda=False):
        super(MultimodalVAE, self).__init__()
        self.image_encoder = ImageEncoder(n_latents)
        self.image_decoder = ImageDecoder(n_latents)
        self.text_encoder = TextEncoder(n_latents, n_characters, bidirectional=True)
        # self.text_encoder = ConcatTextEncoder(n_latents, n_characters)
        self.text_decoder = TextDecoder(n_latents, n_characters, use_cuda=use_cuda)
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

    def decode_image(self, z):
        return self.image_decoder(z)

    def encode_text(self, x):
        return self.text_encoder(x)

    def decode_text(self, z):
        return self.text_decoder(z)

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
        #                                     use_cuda=mu.is_cuda)
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


class TextVAE(nn.Module):
    def __init__(self, n_latents=20, use_cuda=False):
        super(TextVAE, self).__init__()
        self.encoder = TextEncoder(n_latents, n_characters)
        self.decoder = TextDecoder(n_latents, n_characters, use_cuda=use_cuda)
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
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.Conv2d(128, 256, 4, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            Swish(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 400),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(400, 200),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(200, n_latents * 2)
        )
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class ImageDecoder(nn.Module):
    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.upsample = nn.Sequential(
            nn.Linear(n_latents, 256 * 2 * 2),
            Swish(),
        )
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 0, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.ConvTranspose2d(64, 32, 5, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
        )

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsample(z)
        z = z.view(-1, 256, 2, 2)
        z = self.hallucinate(z)
        return F.sigmoid(z)


class TextEncoder(nn.Module):
    """Given variable length text, we move past a single embedding 
    to a bidirectional RNN.

    :param n_latents: size of latent vector
    :param n_characters: number of possible characters (10 for MNIST)
    :param n_hiddens: number of hidden units in GRU
    """
    def __init__(self, n_latents, n_characters, n_hiddens=50, bidirectional=True):
        super(TextEncoder, self).__init__()
        self.embed = nn.Embedding(n_characters, n_hiddens)
        self.gru = nn.GRU(n_hiddens, n_hiddens, 1, dropout=0.1, 
                          bidirectional=bidirectional)
        self.h2p = nn.Linear(n_hiddens, n_latents * 2)  # hiddens to parameters
        self.n_latents = n_latents
        self.n_hiddens = n_hiddens
        self.bidirectional = bidirectional

    def forward(self, x):
        n_hiddens = self.n_hiddens
        n_latents = self.n_latents
        x = self.embed(x)
        x = x.transpose(0, 1)  # GRU expects (seq_len, batch, ...)
        x, h = self.gru(x, None)
        x = x[-1]  # take only the last value
        if self.bidirectional:
            x = x[:, :n_hiddens] + x[:, n_hiddens:]  # sum bidirectional outputs
        x = self.h2p(x)
        return x[:, :n_latents], x[:, n_latents:]


class TextDecoder(nn.Module):
    """GRU for text decoding. 

    :param n_latents: size of latent vector
    :param n_characters: size of characters (10 for MNIST)
    :param use_cuda: whether to use cuda tensors
    :param n_hiddens: number of hidden units in GRU
    """
    def __init__(self, n_latents, n_characters, n_hiddens=50, use_cuda=False):
        super(TextDecoder, self).__init__()
        self.embed = nn.Embedding(n_characters, n_hiddens)
        self.z2h = nn.Linear(n_latents, n_hiddens)
        self.gru = nn.GRU(n_hiddens + n_latents, n_hiddens, 2, dropout=0.1)
        self.h2o = nn.Linear(n_hiddens + n_latents, n_characters)
        self.use_cuda = use_cuda
        self.n_latents = n_latents
        self.n_characters = n_characters

    def forward(self, z):
        n_latents = self.n_latents
        n_characters = self.n_characters
        batch_size = z.size(0)
        # first input character is SOS
        c_in = Variable(torch.LongTensor([SOS]).repeat(batch_size))
        # store output word here
        words = Variable(torch.zeros(batch_size, max_length, n_characters))
        if self.use_cuda:
            c_in = c_in.cuda()
            words = words.cuda()
        # get hiddens from latents
        h = self.z2h(z).unsqueeze(0).repeat(2, 1, 1)
        # look through n_steps and generate characters
        for i in xrange(max_length):
            c_out, h = self.step(i, z, c_in, h)
            sample = torch.max(c_out, dim=1)[1]
            words[:, i] = c_out
            c_in = sample

        return words  # (batch_size, seq_len, ...)

    def generate(self, z):
        """Like, but we are not given an input text"""
        words = self.forward(z)
        batch_size = words.size(0)
        char_size = words.size(2)
        sample = torch.multinomial(words.view(-1, char_size), 1)
        return sample.view(batch_size, max_length)

    def step(self, ix, z, c_in, h):
        c_in = swish(self.embed(c_in))
        c_in = torch.cat((c_in, z), dim=1)
        c_in = c_in.unsqueeze(0)
        c_out, h = self.gru(c_in, h)
        c_out = c_out.squeeze(0)
        c_out = torch.cat((c_out, z), dim=1)
        c_out = self.h2o(c_out)
        c_out = F.log_softmax(c_out)
        return c_out, h


class ConcatTextEncoder(nn.Module):
    """An RNN might be the best choice for ScrambleMNIST because 
    RNNs are definitely not invariant to scrambling. What might work
    is just an MLP.

    :param n_latents: size of latent vector
    :param n_characters: number of possible characters (10 for MNIST)
    :param n_hiddens: number of hidden units in GRU
    """
    def __init__(self, n_latents, n_characters):
        super(ConcatTextEncoder, self).__init__()
        self.embed = nn.Embedding(n_characters, 50)
        self.fnn = nn.Sequential(
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            Swish(),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            Swish(),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            Swish(),
            # hiddens to parameters
            nn.Linear(200, n_latents * 2)
        )
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.embed(x)
        x = x.view(-1, 50 * 4)
        x = self.fnn(x)
        return x[:, :n_latents], x[:, n_latents:]


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


class Attention(nn.Module):
    def __init__(self, in_features, out_features):
        super(Attention, self).__init__()
        self.out_features = out_features
        self.params = Parameter(torch.normal(torch.zeros(in_features), 1))

    def forward(self, x):
        x = self.params * x
        x = torch.topk(x, self.out_features, dim=1, sorted=False)[0]
        return x


def swish(x):
    return x * F.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)
