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

from utils import n_characters
from utils import SOS, EOS


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
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(50, n_latents * 2)
        )
        self.n_latents = n_latents

    def forward(self, x)
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(-1)
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
        x = self.embed(x).unsqueeze(1)
        x = F.relu(x)
        x, h = self.gru(x, None)
        x = x[-1]  # take only the last value
        x = x[:, :50] + x[:, 50:]  # sum bidirectional outputs
        x = F.relu(x)
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
            nn.ConvTranspose2d(20, 20, kernel_size=10),
            nn.ConvTranspose2d(20, 20, kernel_size=8),
            nn.Conv2d(20, 3, kernel_size=3),
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
        n_steps = len(x)
        n_latents = self.n_latents
        n_characters = self.n_characters

        # first input character is SOS
        c_in = Variable(torch.LongTensor([SOS]))
        # store output word here
        w = Variable(torch.zeros(n_steps, 1, n_characters))
        if self.use_cuda:
            c_in = c_in.cuda()
            w = w.cuda()
        # get hiddens from latents
        h = self.z2h(z).unsqueeze(0).repeat(2, 1, 1)
        # look through n_steps and generate characters
        for i in xrange(n_steps):
            c_out, h = self.step(i, z, c_in, h)
            w[i] = c_out
            c_in = x[i]

        return w.squeeze(1)

    def generate(self, z, n_steps):
        """Like, but we are not given an input text"""
        c_in = Variable(torch.LongTensor([SOS]))
        w = Variable(torch.zeros(n_steps, 1, n_characters))
        if self.use_cuda:
            c_in = c_in.cuda()
            w = w.cuda()
        h = self.z2h(z).unsqueeze(0).repeat(2, 1, 1)
        for i in xrange(n_steps):
            c_out, h = self.step(i, z, c_in, h)
            w[i] = c_out
            c_in, top_i = self.sample(c_out, False)
            if top_i == EOS:
                break

        return c_out.squeeze(1)

    def sample(self, c_out, deterministic=False):
        """Sample a word from output distribution.

        :param deterministic: if True, return argmax
        """
        if deterministic:
            top_i = c_out.data.topk(1)[1][0][0]
        else:
            # sample from multinomial
            probas = c_out.data.view(-1).exp()
            top_i = torch.multinomial(probas, 1)[0]

        sample = Variable(torch.LongTensor([top_i]))
        if self.use_cuda:
            sample = sample.cuda()

        return sample, top_i

    def step(self, ix, z, c_in, h):
        c_in = F.relu(self.embed(c_in))
        c_in = torch.cat((c_in, z), dim=1)
        c_in = c_in.unsqueeze(0)
        c_out, h = self.gru(c_in, h)
        c_out = c_out.squeeze(0)
        c_out = torch.cat((c_out, z), dim=1)
        c_out = self.h2o(c_out)
        return c_out, h
