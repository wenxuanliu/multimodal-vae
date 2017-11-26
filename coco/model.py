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

import numpy as np
from glove import GloVe
from utils import MAX_WORDS, SOS, EOS


class MultimodalVAE(nn.Module):
    def __init__(self, n_latents=20, use_cuda=False):
        super(MultimodalVAE, self).__init__()
        self.image_encoder = ImageEncoder(n_latents)
        self.image_decoder = ImageDecoder(n_latents)
        self.text_encoder = TextEncoder(n_latents, n_characters)
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
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            Swish(),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            Swish(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 256),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(256, n_latents * 2)
        )
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class ImageDecoder(nn.Module):
    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.upsample = nn.Sequential(
            nn.Linear(n_latents, 512 * 4 * 4),
            Swish(),
        )
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            Swish(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
        )

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsample(z)
        z = z.view(-1, 512, 4, 4)
        z = self.hallucinate(z)
        return F.sigmoid(z)


class TextEncoder(nn.Module):
    """Given variable length text, we move past a single embedding 
    to a bidirectional RNN.

    :param n_latents: size of latent vector
    :param n_embedding: size of GloVe embedding
                        (default: 300)
    :param n_characters: number of possible characters (10 for MNIST)
    """
    def __init__(self, n_latents, n_embedding=300):
        super(TextEncoder, self).__init__()
        # input size is 300 since GloVe vectors are size 300
        self.gru = nn.GRU(n_embedding, 200, 1, dropout=0.1, bidirectional=True)
        self.h2p = nn.Linear(200, n_latents * 2)  # hiddens to parameters
        self.n_latents = n_latents
        self.n_embedding = n_embedding

    def forward(self, x):
        # input x is a vector of (batch, seq_len, 300)
        # where seq_len = MAX_WORDS
        n_latents = self.n_latents
        x = x.transpose(0, 1)  # GRU expects (seq_len, batch, ...)
        x, h = self.gru(x, None)
        x = x[-1]  # take only the last value
        x = x[:, :200] + x[:, 200:]  # sum bidirectional outputs
        x = self.h2p(x)
        return x[:, :n_latents], x[:, n_latents:]


class TextDecoder(nn.Module):
    """GRU for text decoding. 

    :param n_latents: size of latent vector
    :param n_embedding: size of GloVe embedding
                        (default: 300)
    :param use_cuda: whether to use cuda tensors
    """
    def __init__(self, n_latents, n_embedding=300, use_cuda=False):
        super(TextDecoder, self).__init__()
        self.z2h = nn.Linear(n_latents, 200)
        self.gru = nn.GRU(n_embedding + n_latents, 200, 2, dropout=0.1)
        self.h2o = nn.Linear(n_embedding + n_latents, n_embedding)
        self.glove = GloVe()
        self.use_cuda = use_cuda
        self.n_latents = n_latents
        self.n_embedding = n_embedding

    def forward(self, z):
        n_latents = self.n_latents
        batch_size = z.size(0)
        # when generating, first word is always SOS - get GloVe embedding for it
        sos = self.glove.get_word(SOS)
        w_in = Variable(sos.repeat(batch_size).view(batch_size, self.n_embedding))
        # when generating, the default is EOS - we can initialize the entire sentence
        # defaulted to EOS.
        eos = self.glove.get_word(EOS)
        sentence = Variable(eos.repeat(batch_size, MAX_WORDS).view(
            batch_size, MAX_WORDS, self.n_embedding))
        if self.use_cuda:
            w_in = w_in.cuda()
            sentence = sentence.cuda()
        # get initial hiddens from latents
        h = self.z2h(z).unsqueeze(0).repeat(2, 1, 1)
        # look through n_steps and generate characters
        for i in xrange(MAX_WORDS):
            w_out, h = self.step(i, z, w_in, h)
            words[:, i] = w_out
            w_in = w_out

        return words  # (batch_size, seq_len, ...)

    def generate(self, z):
        words = self.forward(z)  # shape is (batch_size, seq_len, n_embedding)
        words = words.view(-1, self.n_embedding)
        # this returns a list of strings of shape |batch_size * seq_len|
        strings = self.glove.closest_batch(words)
        # reshape this back into a list of size batch_size with space separated
        # words.
        reshape = []
        for i in xrange(words.size(0)):
            sentence = []
            for j in xrange(words.size(1)):
                ix = i * words.size(0) + j
                sentence.append(strings[ix])
            sentence = ' '.join(sentence)
            reshape.append(sentence)

        return reshape

    def step(self, ix, z, w_in, h):
        # w_in is a (batch, n_embedding)
        w_in = torch.cat((w_in, z), dim=1)  # n_embedding + n_latents
        w_in = w_in.unsqueeze(0)
        w_out, h = self.gru(w_in, h)
        w_out = w_out.squeeze(0)
        w_out = torch.cat((w_out, z), dim=1)
        w_out = self.h2o(w_out)
        return w_out, h


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


class PixelCNN(nn.Module):
    """Vanilla PixelCNN from first Van de Oord paper."""
    def __init__(self, n_blocks=15, data_channels=1, hid_dims=128, out_dims=256):
        super(PixelCNN, self).__init__()
        self.conv1 = MaskedConv2d('A', data_channels, hid_dims, 7, 1, 3)
        blocks = []
        for _ in xrange(n_blocks):
            conv = MaskedConv2d('B', hid_dims, hid_dims, 3, 1, 1)
            relu = nn.ReLU(True)
            blocks += [conv, relu]
        self.blocks = nn.Sequential(*blocks)
        self.conv2 = MaskedConv2d('B', hid_dims, hid_dims, 1)
        self.conv3 = MaskedConv2d('B', hid_dims, out_dims * data_channels, 1)
        self.data_channels = data_channels
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.n_blocks = n_blocks

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, self.out_dims, self.data_channels, height, width)         
        return x


class GatedPixelCNN(nn.Module):
    """Improved PixelCNN with blind spot and gated blocks."""
    def __init__(self, n_blocks=15, data_channels=1, hid_dims=128, out_dims=256):
        super(GatedPixelCNN, self).__init__()
        self.conv1 = GatedResidualBlock('A', hid_dims, 7)
        self.blocks = GatedResidualBlockList(n_blocks, 'B', hid_dims, hid_dims, 3)
        self.conv2 = MaskedConv2d('B', hid_dims, hid_dims, 1)
        self.conv3 = MaskedConv2d('B', hid_dims, hid_dims, 1)
        self.conv4 = MaskedConv2d('B', hid_dims, out_dims * data_channels, 1)
        self.data_channels = data_channels
        self.out_dims = out_dims
        self.n_blocks = n_blocks

    def forward(self, x):
        x, h = self.conv1(x, x)
        _, h = self.blocks(x, h)
        h = self.conv2(F.relu(h))
        h = self.conv3(F.relu(h))
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
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GatedResidualBlock, self).__init__()
        self.vertical_conv = CroppedConv2d(in_channels, 2 * out_channels, 
                                           kernel_size=(kernel_size // 2 + 1, kernel_size),
                                           padding=(kernel_size // 2 + 1, kernel_size // 2))
        self.x_to_h_conv = MaskedConv2d(mask_type, 2 * out_channels, 2 * out_channels, 1)
        self.vertical_gate_conv = nn.Conv2d(2 * out_channels, 2 * out_channels, 1)
        self.horizontal_conv = CroppedConv2d(in_channels, 2 * out_channels, 
                                             kernel_size=(1, kernel_size // 2 + 1), 
                                             padding=(0, kernel_size // 2 + 1))
        self.horizontal_gate_conv = nn.Conv2d(2 * out_channels, 2 * out_channels, 1)
        self.horizontal_output = MaskedConv2d(mask_type, out_channels, out_channels, 1)
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


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type,  *args, **kwargs):
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
        
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class MaskedConv2dRGB(nn.Conv2d):
    # Adapted from https://github.com/igul222/pixel_rnn/blob/master/pixel_rnn.py-0
    def __init__(self, mask_type, data_channels, *args, **kwargs):
        super(MaskedConv2dRGB, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        cout, cin, kh, kw = self.weight.size()
        yc, xc = kh // 2, kw // 2
        self.mask.fill_(1)
        
        for i in xrange(xc):
            for j in xrange(yc):
                if (j > yc) or (j == yc and i > xc):
                    mask[:, :, j, i] = 0.
        
        for i in xrange(data_channels):
            for j in xrange(data_channels):
                if (mask_type == 'A' and i >= j) or (mask_type == 'B' and i > j):
                    self.mask[j::data_channels, i::data_channels, yc, xc] = 0.
 
        self.mask_type = mask_type
        self.data_channels = data_channels

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2dRGB, self).forward(x)


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
