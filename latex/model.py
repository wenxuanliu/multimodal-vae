from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from utils import CHAR_VOCAB, N_CHAR_VOCAB, MAX_LENGTH
from utils import SOS, FILL


class MultimodalVAE(nn.Module):
    def __init__(self, n_latents=20, n_hiddens=100, use_cuda=False):
        super(MultimodalVAE, self).__init__()
        self.render_encoder = RenderEncoder(n_latents)
        self.render_decoder = RenderDecoder(n_latents)
        self.formula_encoder = FormulaEncoder(n_latents, N_CHAR_VOCAB, 
                                              n_hiddens=n_hiddens, bidirectional=True)
        self.formula_decoder = FormulaDecoder(n_latents, N_CHAR_VOCAB, 
                                              n_hiddens=n_hiddens, use_cuda=use_cuda)
        self.experts = ProductOfExperts()
        self.n_latents = n_latents
        self.n_hiddens = n_hiddens
        self.use_cuda = use_cuda

    def weight_init(self, mean, std):
        self.image_encoder.weight_init(mean=mean, std=std)
        self.image_decoder.weight_init(mean=mean, std=std)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu

    def forward(self, render=None, formula=None):
        # can't just put nothing
        assert render is not None or formula is not None
        # map examples to distributional parameters
        if render is not None and formula is not None:
            # compute separate gaussians per modality
            render_mu, render_logvar = self.render_encoder(render)
            formula_mu, formula_logvar = self.formula_encoder(formula)
            mu = torch.stack((render_mu, formula_mu), dim=0)
            logvar = torch.stack((render_logvar, formula_logvar), dim=0)
        elif render is not None:
            mu, logvar = self.render_encoder(render)
            mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        elif formula is not None:
            mu, logvar = self.formula_encoder(formula)
            mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        render_recon = self.render_decoder(z)
        formula_recon = self.formula_decoder(z)
        return render_recon, formula_recon, mu, logvar


class RenderVAE(nn.Module):
    def __init__(self, n_latents=20):
        super(RenderVAE, self).__init__()
        self.encoder = RenderEncoder(n_latents)
        self.decoder = RenderDecoder(n_latents)
        self.n_latents = n_latents

    def weight_init(self, mean, std):
        self.encoder.weight_init(mean=mean, std=std)
        self.decoder.weight_init(mean=mean, std=std)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return self.decoder(z), mu, logvar


class FormulaVAE(nn.Module):
    def __init__(self, n_latents=20, use_cuda=False):
        super(FormulaVAE, self).__init__()
        self.encoder = FormulaEncoder(n_latents, N_CHAR_VOCAB)
        self.decoder = FormulaDecoder(n_latents, N_CHAR_VOCAB, use_cuda=use_cuda)
        self.n_latents = n_latents
        self.use_cuda = use_cuda

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return self.decoder(z), mu, logvar


class RenderEncoder(nn.Module):
    def __init__(self, n_latents):
        super(RenderEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
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

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class RenderDecoder(nn.Module):
    def __init__(self, n_latents):
        super(RenderDecoder, self).__init__()
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
            nn.ConvTranspose2d(64, 32, 5, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsample(z)
        z = z.view(-1, 256, 5, 5)
        z = self.hallucinate(z)
        return F.sigmoid(z)


class FormulaEncoder(nn.Module):
    """Given variable length text, we move past a single embedding 
    to a bidirectional RNN.

    :param n_latents: size of latent vector
    :param n_characters: number of possible characters (10 for MNIST)
    :param n_hiddens: number of hidden units in GRU
    """
    def __init__(self, n_latents, n_characters, n_hiddens=50, bidirectional=True):
        super(FormulaEncoder, self).__init__()
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


class FormulaDecoder(nn.Module):
    """GRU for text decoding. 

    :param n_latents: size of latent vector
    :param n_characters: size of characters (10 for MNIST)
    :param use_cuda: whether to use cuda tensors
    :param n_hiddens: number of hidden units in GRU
    """
    def __init__(self, n_latents, n_characters, n_hiddens=50, use_cuda=False):
        super(FormulaDecoder, self).__init__()
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
        words = Variable(torch.zeros(batch_size, MAX_LENGTH, n_characters))
        if self.use_cuda:
            c_in = c_in.cuda()
            words = words.cuda()
        # get hiddens from latents
        h = self.z2h(z).unsqueeze(0).repeat(2, 1, 1)
        # look through n_steps and generate characters
        for i in xrange(MAX_LENGTH):
            c_out, h = self.step(i, z, c_in, h)
            sample = torch.max(c_out, dim=1)[1]
            words[:, i] = c_out
            c_in = sample
        # (batch_size, seq_len, ...)
        return words

    def generate(self, z):
        """Like forward(.), but we are not given an input text"""
        words = self.forward(z)
        batch_size = words.size(0)
        char_size = words.size(2)
        sample = torch.multinomial(words.view(-1, char_size), 1)
        return sample.view(batch_size, MAX_LENGTH)

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


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def swish(x):
    return x * F.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)
