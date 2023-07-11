#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
from typing import Callable, Union, Tuple, List, Dict

import torch
from torch import tensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from . import dic_tools as dic
from . import Hermite
###


# %%
class Dataset0(Dataset):
    """Cass to generate base images
    """
    def __init__(self, N: int, shape: Tuple[int], scale: int):
        super().__init__()
        self.N = N
        self.shape = shape
        self.scale = scale

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sparse = False
        largeSpeckles = False
        lowContrast = False

        state_value = torch.rand(1).item()
        if state_value < 0.05:
            sparse = True
        elif state_value < 0.35:
            largeSpeckles = True
        elif state_value < 0.4:
            lowContrast = True
        img = dic.generate_image(
            self.shape, self.scale,
            sparse, largeSpeckles, lowContrast
        )
        return img, tensor([])
###



# init Dataset (concat list comprehension) -> to(dic.DEVICE)

# __getitem__ :
## get image

## k = i % N0
## i -= k * N0
## transform (rotation / flip / inv)
if k % 2:
    img = 1. - img
k = k // 2
if k > 4:
    img = img.flip(-1)
    k -= 4
if k > 0:
    if k == 3:
        k = -1
    img = img.rot(k, [0, 1])

# add noise
img *= 1. + torch.randn(img.shape) * 0.01

if torch.rand(1).item() < 0.05:
    img += torch.randn(img.shape) * (2/250)
img = (250 * img).int()

# %% END OF FILE
###
