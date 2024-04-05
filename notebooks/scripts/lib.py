from functools import reduce
import matplotlib.image as mpimg
import torch
import polars as pl
import os

def compose(*fns):
    def __inner__(x):
        return reduce(lambda acc, fn: fn(acc), fns, x)
    return __inner__

# img_to_matrix = compose(mpimg.imread, lambda x: torch.as_tensor(x, dtype=torch.float32))

def img_to_matrix(x: str):
    x = mpimg.imread(x)
    return torch.as_tensor(x, dtype=torch.float32)

get_size = compose(os.listdir, list.__len__)

create_df = lambda path: pl.DataFrame({
    'file': (fake_faces := os.listdir(path + 'fake')) + (real_faces := os.listdir(path + 'real')),
    'label': [ 0 for _ in range(len(fake_faces))] + [1 for _ in range(len(real_faces)) ],
})