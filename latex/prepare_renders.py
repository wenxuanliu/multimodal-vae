# loop through each render and convert by applying
# alpha composite.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from PIL import Image

from datasets import alpha_composite_with_color


if __name__ == "__main__":
    if not os.path.isdir('./data/formula_images_processed'):
        os.mkdir('./data/formula_images_processed')

    render_files = os.listdir('./data/formula_images/')
    for ix, render_file in enumerate(render_files):
        print('Saving latex example [{}/{}]'.format(ix + 1, len(render_files)))
        render_path = os.path.join('./data/formula_images/', render_file)

        with Image.open(render_path) as render:
            render = render.convert('RGBA')
            render = alpha_composite_with_color(render)
            render = render.convert('L')
            render.save(os.path.join('./data/formula_images_processed/', render_file))
