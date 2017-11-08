"""PyTorch wrapper for GloVe embeddings.

Based one https://github.com/spro/practical-pytorch/blob/\
            master/glove-word-vectors/glove-word-vectors.ipynb
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torchtext.vocab as vocab


class GloVe(object):
    def __init__():
        self.glove = vocab.GloVe(name='840B', dim=300)

    def get_word(self, word):
        return self.glove.vectors[self.glove.stoi[word]]

    def closest(self, vec, n=10):
        """
        Find the closest words for a given vector; this is a 
        naive implementation that goes through each word in the
        dictionary. 

        :param vec: PyTorch vector
        :param n: number of "close-by" vectors to return
                  (default: 10)
        """
        all_dists = [(w, torch.dist(vec, get_word(w))) 
                     for w in self.glove.itos]
        return sorted(all_dists, key=lambda t: t[1])[:n]

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
