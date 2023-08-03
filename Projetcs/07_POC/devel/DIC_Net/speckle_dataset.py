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

import multiprocessing

DEVICE = dic.DEVICE
NUM_WORKERS = multiprocessing.cpu_count()
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


def process_digitalization(
            image: tensor, supplementary_noise: bool
        ) -> tensor:
    """Simulation of digitalization, with photon like noise, and added random
    noise if desired

    Args:
        image (tensor): input image
        supplementary_noise (bool): wether to add uniform random noise or not

    Returns:
        tensor: modified image
    """
    # approximation of photonic noise
    # proportionnal to the illumination
    image *= 1. + torch.randn(image.shape, device=image.device) * 0.01

    if supplementary_noise:
        image += torch.randn(image.shape, device=image.device) * (2/250)

    return (250 * image).int().float()


def update_u_ux_uy_translate_reshape_inplace(data, u, v, x0, y0, ax, ay):
    ax, ay = 1./ax, 1./ay
    data[0] = u * (ax - 1.) + (data[0] - x0) * ax
    data[1] = (ax - 1.) + data[1] * ax
    data[2] *= ax

    data[3] = v * (ay - 1.) + (data[3] - y0) * ay
    data[4] *= ay
    data[5] = (ay - 1.) + data[5] * ay
    return data


def restore_u_ux_uy_translate_reshape_inplace(data, u, v, x0, y0, ax, ay):
    if data.shape[0] == 2:
        # uv
        data[0] = ax * data[0] - u * (1. - ax) + x0
        data[1] = ay * data[1] - v * (1. - ay) + y0
    else:
        # strain
        data[0] = ax * data[0] - (1. - ax)
        data[1] *= ax

        data[2] *= ay
        data[3] = ay * data[3] - (1. - ay)
    return data


class SpeckleDataset(Dataset):
    def __init__(
                self,
                N0: int,
                scale: int,
                mode: str,
                transform: bool = False,
                i_case_min: int = 0,
            ):
        super().__init__()
        self.N0 = N0
        self.N = 16 * N0
        self.transform = transform
        self.i_case_min = i_case_min
        self.shape = Hermite.OUT0.shape[-2:]
        self.margin = 4 + 3 + 4  # deform, interp, supplement
        shape = tuple(n + 2*self.margin for n in self.shape)
        self.dataset0 = Dataset0(N0, shape, scale)
        self.data_loader0 = DataLoader(
            self.dataset0, batch_size=8,
            num_workers=NUM_WORKERS, shuffle=False
        )
        self.images = torch.zeros((N0,) + shape, device=DEVICE)

        ur, vr = torch.meshgrid(
            torch.arange(self.shape[1]),
            torch.arange(self.shape[0]),
            indexing='xy'
        )
        self.uvr = ur.to(DEVICE), vr.to(DEVICE)

        self.full_data = False
        if mode == 'full':
            print('DATALOADER : FULL DATA')
            self.full_data = True
        elif mode == 'displacement':
            print('DATALOADER : DISPLACEMENT')
            self.displacement = True
        else:
            print('DATALOADER : STRAIN')
            self.displacement = False

        # self.cases = torch.zeros(self.N, dtype=torch.int)
        # self.resample_info = torch.zeros(
        #     (self.N, 4), dtype=torch.float, device=DEVICE)

    def __len__(self):
        return self.N

    def __getitem__(self, idx, case: int = None):
        # print('idx:', idx)
        # print('images:', self.images.shape)
        k = idx // self.N0
        image = self.images[idx - k * self.N0]
        # transform (rotation / flip / inv)
        if k % 2:
            image = 1. - image
        k = k // 2
        if k > 4:
            image = image.flip(-1)
            k -= 4
        if k > 0:
            if k == 3:
                k = -1
            image = image.rot90(k, [0, 1])

        # compute deformation
        if case is None:
            case = torch.randint(
                low=self.i_case_min, high=5, size=[1]
            ).item()
        # self.cases[idx] = case

        y = Hermite.Hermite2d_case(case)

        # interp image ref
        u2 = self.margin + Hermite.UU + y[0].ravel()
        v2 = self.margin + Hermite.VV + y[3].ravel()
        image_ref = dic.interp_img(
            image, u2, v2
        ).reshape(self.shape)

        # compute image def limites
        u20, u21 = int(u2.min()), int(u2.max() + .5)
        v20, v21 = int(v2.min()), int(v2.max() + .5)

        # interp image def (linear interp)
        u = torch.linspace(u20, u21, self.shape[1], device=DEVICE)
        v = torch.linspace(v20, v21, self.shape[0], device=DEVICE)
        ax, ay = u[1] - u[0], v[1] - v[0]
        v, u = torch.meshgrid(v, u, indexing='ij')
        image_def = dic.interp_img(
            image, u.ravel(), v.ravel()
        ).reshape(self.shape)

        u20 -= self.margin
        v20 -= self.margin
        # self.resample_info[idx] = torch.tensor([u20, v20, ax, ay])
        # modif target u, ux, uy + v
        if self.transform:
            update_u_ux_uy_translate_reshape_inplace(
                y,
                Hermite.UU.reshape(y.shape[1:]),
                Hermite.VV.reshape(y.shape[1:]),
                u20, v20, ax.item(), ay.item()
            )

        # simulate digitalization with potential extra noise
        image_ref = process_digitalization(
            image_ref, torch.rand(1).item() < 0.05
        )
        image_def = process_digitalization(
            image_def, torch.rand(1).item() < 0.05
        )

        # concatenate images
        image_pair = torch.cat((image_ref[None], image_def[None]))

        if not self.full_data:
            if self.displacement:
                y = y[[0, 3]]
            else:
                # exy
                # y = torch.stack(
                #     (y[:, 1], y[:, 5], 0.5*(y[:, [2, 4]].sum(1))),
                #     dim=1
                # )
                y = y[[1, 2, 4, 5]]

        return image_pair, y, torch.tensor([u20, v20, ax, ay])

    def init_dataset0(self):
        n1 = 0
        for images, _ in self.data_loader0:
            n2 = n1 + images.shape[0]
            self.images[n1:n2, :, :] = images[:, :, :].to(DEVICE)
            n1 = n2

# %% END OF FILE
###
