from functools import reduce
import matplotlib.image as mpimg
import torch
import polars as pl
import os

def compose(*fns):
    def __inner__(x):
        return reduce(lambda acc, fn: fn(acc), fns, x)
    return __inner__

def img_to_matrix(x: str):
    x = mpimg.imread(x).copy()
    return torch.Tensor(x)

get_size = compose(os.listdir, list.__len__)

create_df = lambda path: pl.DataFrame({
    'file': (fake_faces := os.listdir(path + 'fake')) + (real_faces := os.listdir(path + 'real')),
    'label': [ 0 for _ in range(len(fake_faces))] + [1 for _ in range(len(real_faces)) ],
})

def clip_img(img: torch.Tensor, shape=224) -> torch.Tensor:
    prune = (x:=img.shape[0]) - shape
    prune = prune // 2
    return img[prune: x - prune, prune: x - prune, :]