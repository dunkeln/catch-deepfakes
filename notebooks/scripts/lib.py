from functools import reduce
import matplotlib.image as mpimg
import numpy as np
import os

def compose(*fns):
    def __inner__(x):
        return reduce(lambda acc, fn: fn(acc), fns, x)
    return __inner__

img_to_matrix = compose(mpimg.imread, np.asarray)

read_dir = lambda path: map(
    img_to_matrix, compose(
        os.listdir,
        lambda x: map(lambda file: os.path.join(path, file), x)
    )(path)
)

get_size = lambda path: len(os.listdir(path + '/fake')) + len(os.listdir(path + '/real'))