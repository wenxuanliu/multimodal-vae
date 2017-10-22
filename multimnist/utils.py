"""Utilities for handling text, images, etc."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import string
import random
import time
import math
import torch
from torch.autograd import Variable


all_characters = '0123456789'
n_characters = len(all_characters)
SOS = n_characters  # start of sentence
EOS = n_characters + 1  # end of sentence
n_characters += 2


def char_tensor(string):
    """Turn a string into a tensor.

    :param string: str object
    :return tensor: torch.Tensor object. Not a Variable.
    """
    size = len(string) + 1
    tensor = torch.zeros(size).long()
    for c in xrange(len(string)):
        tensor[c] = all_characters.index(string[c])
    tensor[-1] = EOS
    return tensor


def tensor_to_string(tensor):
    """Turn a tensor of indices into a string.

    :param tensor: torch.Tensor object. Not a Variable.
    :return string: str object
    """
    string = ''
    for i in xrange(tensor.size(0)):
        ti = t[i]
        top_k = ti.topk(1)
        top_i = top_k[1][0]
        string += index_to_char(top_i)
        if top_i == EOS: break
    return string


def longtensor_to_string(tensor):
    """Identical to tensor_to_string but for LongTensors."""
    string = ''
    for i in range(tensor.size(0)):
        top_i = tensor[i]
        string += index_to_char(top_i)
    return string


def index_to_char(top_i):
    if top_i == EOS:
        return '$'
    elif top_i == SOS:
        return '^'
    else:
        return all_characters[top_i]
