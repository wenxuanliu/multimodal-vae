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

    def forward(self, image=None, text=None):
        # can't just put nothing
        assert image is not None or text is not None
        
        if image is not None and text is not None:
            # compute separate gaussians per modality
            image_mu, image_logvar = self.encode_image(image)
            text_mu, text_logvar = self.encode_text(text)
            mu = torch.stack((image_mu, text_mu), dim=0)
            logvar = torch.stack((image_logvar, text_logvar), dim=0)
            # grab learned mixture weights and sample
            mu, logvar = self.experts(mu, logvar)
        elif image is not None:
            mu, logvar = self.encode_image(image)
        elif text is not None:
            mu, logvar = self.encode_text(text)
        
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

    def decode(self, z, x):
        return self.decoder(z, x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, x), mu, logvar


class ImageEncoder(nn.Module):
    """This task is quite a bit harder than MNIST so we probably need 
    to use an RNN of some form. This will be good to get us ready for
    natural images.

    :param n_latents: size of latent vector
    """
    def __init__(self, n_latents):
        super(ImageEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(20),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1620, 400),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(400, 50),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(50, n_latents * 2)
        )
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(-1, 20 * 9 * 9)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class TextEncoder(nn.Module):
    """Given variable length text, we move past a single embedding 
    to a bidirectional RNN.

    :param n_latents: size of latent vector
    :param n_characters: number of possible characters (10 for MNIST)
    """
    def __init__(self, n_latents, n_characters):
        super(TextEncoder, self).__init__()
        self.embed = nn.Embedding(n_characters, 50)
        self.gru = nn.GRU(50, 50, 1, dropout=0.1, bidirectional=True)
        self.h2p = nn.Linear(50, n_latents * 2)  # hiddens to parameters
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.embed(x)
        x = x.transpose(0, 1)  # GRU expects (seq_len, batch, ...)
        x, h = self.gru(x, None)
        x = x[-1]  # take only the last value
        x = x[:, :50] + x[:, 50:]  # sum bidirectional outputs
        x = self.h2p(x)
        return x[:, :n_latents], x[:, n_latents:]


class ImageDecoder(nn.Module):
    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.upsample = nn.Sequential(
            nn.Linear(n_latents, 50),
            nn.ReLU(),
            nn.Linear(50, 10 * 20 * 20),
            nn.ReLU(),
        )
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(10, 20, kernel_size=12),
            nn.ConvTranspose2d(20, 20, kernel_size=12),
            nn.ConvTranspose2d(20, 20, kernel_size=10),
            nn.Conv2d(20, 1, kernel_size=2),
        )

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsample(z)
        z = z.view(-1, 10, 20, 20)
        z = self.hallucinate(z)
        return F.sigmoid(z)

    def generate(self, z):
        z = self.upsample(z)
        z = z.view(-1, 10, 20, 20)
        z = self.hallucinate(z)
        image = F.sigmoid(z)
        return image * 255.


class TextDecoder(nn.Module):
    """GRU for text decoding. 

    :param n_latents: size of latent vector
    :param n_characters: size of characters (10 for MNIST)
    :param use_cuda: whether to use cuda tensors
    """
    def __init__(self, n_latents, n_characters, use_cuda=False):
        super(TextDecoder, self).__init__()
        self.embed = nn.Embedding(n_characters, 50)
        self.z2h = nn.Linear(n_latents, 50)
        self.gru = nn.GRU(50 + n_latents, 50, 2, dropout=0.1)
        self.h2o = nn.Linear(50 + n_latents, n_characters)
        self.use_cuda = use_cuda
        self.n_latents = n_latents
        self.n_characters = n_characters

    def forward(self, z, x):
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
            words[:, i] = c_out
            c_in = x[:, i]

        return words  # (batch_size, seq_len, ...)

    def generate(self, z):
        """Like, but we are not given an input text"""
        batch_size = z.size(0)
        c_in = Variable(torch.LongTensor([SOS]))
        words = Variable(torch.zeros(batch_size, max_length, n_characters))
        if self.use_cuda:
            c_in = c_in.cuda()
            words = words.cuda()
        h = self.z2h(z).unsqueeze(0).repeat(2, 1, 1)
        for i in xrange(max_length):
            c_out, h = self.step(i, z, c_in, h)
            words[:, i] = c_out
            top_i = torch.multinomial(c_out, 1)[0]   
            c_in = Variable(torch.LongTensor([top_i]))
            if self.use_cuda:
                c_in = c_in.cuda()
        return words

    def step(self, ix, z, c_in, h):
        c_in = F.relu(self.embed(c_in))
        c_in = torch.cat((c_in, z), dim=1)
        c_in = c_in.unsqueeze(0)
        c_out, h = self.gru(c_in, h)
        c_out = c_out.squeeze(0)
        c_out = torch.cat((c_out, z), dim=1)
        c_out = self.h2o(c_out)
        c_out = F.log_softmax(c_out)
        return c_out, h
