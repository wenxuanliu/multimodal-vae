from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import random
import numpy as np
from PIL import Image
from random import shuffle
from scipy.misc import imresize

import torch
import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset

VALID_PARTITIONS = {'train': 'im2latex_train.lst', 
                    'val': 'im2latex_val.lst', 
                    'test': 'im2latex_test.lst'}

CHAR_VOCAB = gen_vocab()
N_CHAR_VOCAB = len(CHAR_VOCAB)
MAX_LENGTH = 1000  # 1000 characters is the most to generate
SOS = N_CHAR_VOCAB
FILL = N_CHAR_VOCAB + 1


class Image2Latex(Dataset):
    def __init__(self, partition='train', render_transform=None, formula_transform=None):
        self.partition = partition
        self.render_transform = render_transform
        self.formula_transform = formula_transform

        assert partition in VALID_PARTITIONS.keys()
        
        with open('./data/im2latex_formulas.lst') as fp:
            self.formulas = fp.readlines()

        with open(os.path.join('./data', VALID_PARTITIONS[partition])) as fp:
            data = fp.readlines()
            row_ids, row_render_ids = [], []
            for row in data:
                row_elements = row.split(' ')
                row_id = int(row_elements[0])
                row_render_id = int(row_elements[1])
                row_ids.append(row_id)
                row_render_ids.append(row_render_id)

        self.ids = row_ids
        self.render_ids = row_render_ids
        self.size = len(self.ids)

    def __getitem__(self, index):
        render_path = os.path.join('./data/%s.png' % self.render_ids[index])
        render = Image.open(render_path)
        render = alpha_composite_with_color(render.convert('RGBA'))
        render = render.convert('L')

        if self.render_transform is not None:
            render = self.render_transform(render)

        formula = self.formulas[index]

        if self.formula_transform is not None:
            formula = self.formula_transform(formula)

        return render, formula

    def __len__(self):
        return self.size


def gen_vocab():
    with open('./data/im2latex_formulas.lst') as fp:
        formulas = fp.readlines()

    char_set = set()
    for formula in formulas:
        char_set = char_set.union(set(formula))

    return sorted(list(char_set))  # 110 unique chars


def gen_max_length():
    with open('./data/im2latex_formulas.lst') as fp:
        formulas = fp.readlines()
    return max([len(formula) for formula in formulas])


def string_to_tensor(string):
    """Turn a string into a Tensor.

    :param string: string
    :return tensor: torch.Tensor object
    """
    tensor = torch.ones(MAX_LENGTH).long() * FILL
    for ix in xrange(len(string)):
        tensor[ix] = CHAR_VOCAB.index(string[ix])
    return tensor


def tensor_to_stirng(tensor):
    """Turn a string into a Tensor.

    :param tensor: torch.Tensor object
    :param string: string
    """
    string = ''
    for ix in xrange(tensor.size(0)):
        top_i = tensor[ix]
        string += index_to_char(top_i)
    return string


def index_to_char(top_i):
    # extra characters map to empty string.
    if top_i == SOS or top_i == FILL:
        return ''
    else:
        return CHAR_VOCAB[top_i]


def alpha_composite(front, back):
    """Alpha composite two RGBA images.
    Source: http://stackoverflow.com/a/9166671/284318
    Keyword Arguments:
    front -- PIL RGBA Image object
    back -- PIL RGBA Image object
    """
    front = np.asarray(front)
    back = np.asarray(back)
    result = np.empty(front.shape, dtype='float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    falpha = front[alpha] / 255.0
    balpha = back[alpha] / 255.0
    result[alpha] = falpha + balpha * (1 - falpha)
    old_setting = np.seterr(invalid='ignore')
    result[rgb] = (front[rgb] * falpha + back[rgb] * balpha * (1 - falpha)) / result[alpha]
    np.seterr(**old_setting)
    result[alpha] *= 255
    np.clip(result, 0, 255)
    # astype('uint8') maps np.nan and np.inf to 0
    result = result.astype('uint8')
    result = Image.fromarray(result, 'RGBA')
    return result


def alpha_composite_with_color(image, color=(255, 255, 255)):
    """Alpha composite an RGBA image with a single color image of the
    specified color and the same size as the original image.
    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)
    """
    back = Image.new('RGBA', size=image.size, color=color + (255,))
    return alpha_composite(image, back)
