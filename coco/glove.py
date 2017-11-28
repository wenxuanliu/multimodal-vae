"""PyTorch wrapper for GloVe embeddings.

Based one https://github.com/spro/practical-pytorch/blob/\
            master/glove-word-vectors/glove-word-vectors.ipynb
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
from torch.autograd import Variable
import torchtext.vocab as vocab


class GloVe(object):
    def __init__(self):
        super(GloVe, self).__init__()
        self.glove = vocab.GloVe(name='840B', dim=300)

    def get_word(self, word):
        if word in self.glove.stoi:
            return self.glove.vectors[self.glove.stoi[word]]
        return None

    def closest(self, vec, n=10):
        """
        Find the closest words for a given vector; this is a 
        naive implementation that goes through each word in the
        dictionary. 

        :param vec: PyTorch vector
                    size 300
        :param n: number of "close-by" vectors to return
                  (default: 10)
        """
        all_dists = [(w, torch.dist(vec, self.get_word(w))) 
                     for w in self.glove.itos]
        return sorted(all_dists, key=lambda t: t[1])[:n]

    def closest_batch(self, vec_batch):
        """Find closest words for a batch of vectors. This 
        loops through all glove training vectors once.

        :param vec: PyTorch vector
                    size N x 300
        """
        batch_size = len(vec_batch)
        dict_size = len(self.glove.itos)
        all_dists = torch.zeros((batch_size, dict_size))
        use_cuda = vec_batch.is_cuda

        for iw, word in enumerate(self.glove.itos):
            for iv, vec in enumerate(vec_batch):
                word_vec = self.get_word(word)
                if use_cuda:
                    word_vec = word_vec.cuda()
                word_vec = Variable(word_vec)
                dist = torch.dist(vec, word_vec)
                all_dists[iv, iw] = dist.data[0]

        top_dist = torch.zeros(batch_size)
        for ix in xrange(batch_size):
            top_dist[ix] = torch.sort(all_dists[ix, :], dim=1)[1][0]

        words = []
        for ix in xrange(batch_size):
            words.append(self.glove.itos[top_dist[ix]])

        return words

    def analogy(w1, w2, w3, n=5, filter_given=True):
        """Return 4th word in a 4 word analogy game.

        :param w1: string
        :param w2: string
        :param w3: string
        :param n: number of closest words to return
        :param filter_given: if True, don't return words 
                             that are 1 of 3 input words.
        """
        # print('\n[%s : %s :: %s : ?]' % (w1, w2, w3))
       
        # w2 - w1 + w3 = w4
        closest_words = self.closest(self.get_word(w2) - 
                                     self.get_word(w1) + 
                                     self.get_word(w3))
        
        # Optionally filter out given words
        if self.filter_given:
            closest_words = [t for t in closest_words if t[0] not in [w1, w2, w3]]
            
        return closest_words[:n]
