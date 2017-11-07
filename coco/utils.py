"""Utilities for handling text, images, etc.
Unlike MultiMNIST, we need to include far more characters.
This is a much more difficult learning problem.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import string
import random
import time
import math
import torch
from torch.autograd import Variable

max_length = 100  # max number of characters in a description
all_characters = string.printable
n_characters = len(all_characters)
SOS = n_characters
# in MultiMNIST, we could just do a FILLER character. here I 
# think we want to actually learn the an end of sentence character.
EOS = n_characters + 1
n_characters += 2


def coco_char_tensor(string_arr, deterministic=False):
    """Randomly sample a caption from an array of captions. Then
    call char_tensor(.) to convert to a tensor.

    :param string_arr: list of strings
    :param deterministic: if True, return 1st index
    :return tensor: torch.Tensor object. Not a Variable.
    """
    string = (string_arr[0] if deterministic 
              else random.choice(string_arr))
    return char_tensor(string)


def char_tensor(string):
    """Turn a string into a tensor. If we want to do 
    batch processing, we are forced to make this tensor 
    uniform shaped; pad with EOS characters.

    :param string: str object
    :return tensor: torch.Tensor object. Not a Variable.
    """
    if len(string) > max_length:
        string = string[:max_length]
    tensor = torch.ones(max_length).long() * EOS
    for c in xrange(len(string)):
        tensor[c] = all_characters.index(string[c])
    return tensor
    

def tensor_to_string(tensor):
    """Identical to tensor_to_string but for LongTensors."""
    string = ''
    for i in range(tensor.size(0)):
        top_i = tensor[i]
        string += index_to_char(top_i)
    return string


def index_to_char(top_i):
    if top_i == SOS:
        return '^'
    elif top_i == EOS:
        return '$'
    else:
        return all_characters[top_i]

