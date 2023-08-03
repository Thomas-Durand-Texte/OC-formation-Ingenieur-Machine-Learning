#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
from typing import Callable, Union, Tuple, List, Dict
import os

import numpy as np
import torch
from torch import tensor

import time
###

ELEMENT_SIZES = [5, 9, 17, 33, 65]
MAX_DISPS = [1., 1., 2., 3., 4.]

# print('HERMITE : MAX DISPS MODIFIED')
# MAX_DISPS[4] = 0.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'

OUT0 = torch.empty((6, 128, 128), dtype=torch.float, device=DEVICE)

VV, UU = torch.meshgrid(
    torch.arange(OUT0.shape[1]),
    torch.arange(OUT0.shape[2]),
    indexing='ij'
)
UU, VV = UU.ravel().to(DEVICE), VV.ravel().to(DEVICE)

IDX = torch.tensor([1, 1, 0, 0], device=DEVICE)
IDY = torch.tensor([0, 1, 1, 0], device=DEVICE)


# %% 1D c2 Hermite polyunomial shape functions
def N1_0(k):
    return -(k - 1)**3 * (3 * k**2 + 9 * k + 8) * 0.0625  # /16


def N1_0_k(k):
    return -(
        (6 * k + 9) * (k - 1)**3
        + 3*(k - 1)**2 * (3 * k**2 + 9 * k + 8)
    ) * 0.0625


def N2_0(k):
    return (k + 1)**3 * (3 * k**2 - 9 * k + 8) * 0.0625


def N2_0_k(k):
    return (
        (6 * k - 9) * (k + 1)**3
        + 3 * (k + 1)**2 * (3 * k**2 - 9 * k + 8)
    ) * 0.0625


def N1_1(k):
    return -(k - 1)**3 * (3 * k + 5) * (k + 1) * 0.0625


def N1_1_k(k):
    return -(
        3*(k - 1)**3*(k + 1)
        + (3*k + 5)*(k - 1)**3
        + 3*(3*k + 5)*(k - 1)**2*(k + 1)
    ) * 0.0625


def N2_1(k):
    return -(k + 1)**3 * (3 * k - 5) * (k - 1)*0.0625


def N2_1_k(k):
    return -(
        3*(k - 1) * (k + 1)**3
        + (3 * k - 5) * (k + 1)**3
        + 3*(3 * k - 5) * (k - 1)*(k + 1)**2
    ) * 0.0625


def N1_2(k):
    return -(k - 1)**3 * (k + 1)**2 * 0.0625


def N1_2_k(k):
    return -(
        (2 * k + 2) * (k - 1)**3
        + 3 * (k - 1)**2 * (k + 1)**2
    ) * 0.0625


def N2_2(k):
    return (k + 1)**3 * (k - 1)**2 * 0.0625


def N2_2_k(k):
    return (
        (2 * k - 2)*(k + 1)**3
        + 3 * (k - 1)**2 * (k + 1)**2
    ) * 0.0625

###


# %% 2D c2 Hermite polynomial shape function
def N1_00(k, q):
    return N1_0(k) * N1_0(q)


def N1_00_x(k, q):
    return N1_0_k(k) * N1_0(q)


def N1_00_y(k, q):
    return N1_0(k) * N1_0_k(q)


def N1_10(k, q):
    return N1_1(k) * N1_0(q)


def N1_10_x(k, q):
    return N1_1_k(k) * N1_0(q)


def N1_10_y(k, q):
    return N1_1(k) * N1_0_k(q)


def N1_01(k, q):
    return N1_0(k) * N1_1(q)


def N1_01_x(k, q):
    return N1_0_k(k) * N1_1(q)


def N1_01_y(k, q):
    return N1_0(k) * N1_1_k(q)


def N1_11(k, q):
    return N1_1(k) * N1_1(q)


def N1_11_x(k, q):
    return N1_1_k(k) * N1_1(q)


def N1_11_y(k, q):
    return N1_1(k) * N1_1_k(q)


def N1_20(k, q):
    return N1_2(k) * N1_0(q)


def N1_20_x(k, q):
    return N1_2_k(k) * N1_0(q)


def N1_20_y(k, q):
    return N1_2(k) * N1_0_k(q)


def N1_02(k, q):
    return N1_0(k) * N1_2(q)


def N1_02_x(k, q):
    return N1_0_k(k) * N1_2(q)


def N1_02_y(k, q):
    return N1_0(k) * N1_2_k(q)


def N1_21(k, q):
    return N1_2(k) * N1_1(q)


def N1_21_x(k, q):
    return N1_2_k(k) * N1_1(q)


def N1_21_y(k, q):
    return N1_2(k) * N1_1_k(q)


def N1_12(k, q):
    return N1_1(k) * N1_2(q)


def N1_12_x(k, q):
    return N1_1_k(k) * N1_2(q)


def N1_12_y(k, q):
    return N1_1(k) * N1_2_k(q)


def N1_22(k, q):
    return N1_2(k) * N1_2(q)


def N1_22_x(k, q):
    return N1_2_k(k) * N1_2(q)


def N1_22_y(k, q):
    return N1_2(k) * N1_2_k(q)


def N2_00(k, q):
    return N2_0(k) * N1_0(q)


def N2_00_x(k, q):
    return N2_0_k(k) * N1_0(q)


def N2_00_y(k, q):
    return N2_0(k) * N1_0_k(q)


def N2_10(k, q):
    return N2_1(k) * N1_0(q)


def N2_10_x(k, q):
    return N2_1_k(k) * N1_0(q)


def N2_10_y(k, q):
    return N2_1(k) * N1_0_k(q)


def N2_01(k, q):
    return N2_0(k) * N1_1(q)


def N2_01_x(k, q):
    return N2_0_k(k) * N1_1(q)


def N2_01_y(k, q):
    return N2_0(k) * N1_1_k(q)


def N2_11(k, q):
    return N2_1(k) * N1_1(q)


def N2_11_x(k, q):
    return N2_1_k(k) * N1_1(q)


def N2_11_y(k, q):
    return N2_1(k) * N1_1_k(q)


def N2_20(k, q):
    return N2_2(k) * N1_0(q)


def N2_20_x(k, q):
    return N2_2_k(k) * N1_0(q)


def N2_20_y(k, q):
    return N2_2(k) * N1_0_k(q)


def N2_02(k, q):
    return N2_0(k) * N1_2(q)


def N2_02_x(k, q):
    return N2_0_k(k) * N1_2(q)


def N2_02_y(k, q):
    return N2_0(k) * N1_2_k(q)


def N2_21(k, q):
    return N2_2(k) * N1_1(q)


def N2_21_x(k, q):
    return N2_2_k(k) * N1_1(q)


def N2_21_y(k, q):
    return N2_2(k) * N1_1_k(q)


def N2_12(k, q):
    return N2_1(k) * N1_2(q)


def N2_12_x(k, q):
    return N2_1_k(k) * N1_2(q)


def N2_12_y(k, q):
    return N2_1(k) * N1_2_k(q)


def N2_22(k, q):
    return N2_2(k) * N1_2(q)


def N2_22_x(k, q):
    return N2_2_k(k) * N1_2(q)


def N2_22_y(k, q):
    return N2_2(k) * N1_2_k(q)


def N3_00(k, q):
    return N2_0(k) * N2_0(q)


def N3_00_x(k, q):
    return N2_0_k(k) * N2_0(q)


def N3_00_y(k, q):
    return N2_0(k) * N2_0_k(q)


def N3_10(k, q):
    return N2_1(k) * N2_0(q)


def N3_10_x(k, q):
    return N2_1_k(k) * N2_0(q)


def N3_10_y(k, q):
    return N2_1(k) * N2_0_k(q)


def N3_01(k, q):
    return N2_0(k) * N2_1(q)


def N3_01_x(k, q):
    return N2_0_k(k) * N2_1(q)


def N3_01_y(k, q):
    return N2_0(k) * N2_1_k(q)


def N3_11(k, q):
    return N2_1(k) * N2_1(q)


def N3_11_x(k, q):
    return N2_1_k(k) * N2_1(q)


def N3_11_y(k, q):
    return N2_1(k) * N2_1_k(q)


def N3_20(k, q):
    return N2_2(k) * N2_0(q)


def N3_20_x(k, q):
    return N2_2_k(k) * N2_0(q)


def N3_20_y(k, q):
    return N2_2(k) * N2_0_k(q)


def N3_02(k, q):
    return N2_0(k) * N2_2(q)


def N3_02_x(k, q):
    return N2_0_k(k) * N2_2(q)


def N3_02_y(k, q):
    return N2_0(k) * N2_2_k(q)


def N3_21(k, q):
    return N2_2(k) * N2_1(q)


def N3_21_x(k, q):
    return N2_2_k(k) * N2_1(q)


def N3_21_y(k, q):
    return N2_2(k) * N2_1_k(q)


def N3_12(k, q):
    return N2_1(k) * N2_2(q)


def N3_12_x(k, q):
    return N2_1_k(k) * N2_2(q)


def N3_12_y(k, q):
    return N2_1(k) * N2_2_k(q)


def N3_22(k, q):
    return N2_2(k) * N2_2(q)


def N3_22_x(k, q):
    return N2_2_k(k) * N2_2(q)


def N3_22_y(k, q):
    return N2_2(k) * N2_2_k(q)


def N4_00(k, q):
    return N1_0(k) * N2_0(q)


def N4_00_x(k, q):
    return N1_0_k(k) * N2_0(q)


def N4_00_y(k, q):
    return N1_0(k) * N2_0_k(q)


def N4_10(k, q):
    return N1_1(k) * N2_0(q)


def N4_10_x(k, q):
    return N1_1_k(k) * N2_0(q)


def N4_10_y(k, q):
    return N1_1(k) * N2_0_k(q)


def N4_01(k, q):
    return N1_0(k) * N2_1(q)


def N4_01_x(k, q):
    return N1_0_k(k) * N2_1(q)


def N4_01_y(k, q):
    return N1_0(k) * N2_1_k(q)


def N4_11(k, q):
    return N1_1(k) * N2_1(q)


def N4_11_x(k, q):
    return N1_1_k(k) * N2_1(q)


def N4_11_y(k, q):
    return N1_1(k) * N2_1_k(q)


def N4_20(k, q):
    return N1_2(k) * N2_0(q)


def N4_20_x(k, q):
    return N1_2_k(k) * N2_0(q)


def N4_20_y(k, q):
    return N1_2(k) * N2_0_k(q)


def N4_02(k, q):
    return N1_0(k) * N2_2(q)


def N4_02_x(k, q):
    return N1_0_k(k) * N2_2(q)


def N4_02_y(k, q):
    return N1_0(k) * N2_2_k(q)


def N4_21(k, q):
    return N1_2(k) * N2_1(q)


def N4_21_x(k, q):
    return N1_2_k(k) * N2_1(q)


def N4_21_y(k, q):
    return N1_2(k) * N2_1_k(q)


def N4_12(k, q):
    return N1_1(k) * N2_2(q)


def N4_12_x(k, q):
    return N1_1_k(k) * N2_2(q)


def N4_12_y(k, q):
    return N1_1(k) * N2_2_k(q)


def N4_22(k, q):
    return N1_2(k) * N2_2(q)


def N4_22_x(k, q):
    return N1_2_k(k) * N2_2(q)


def N4_22_y(k, q):
    return N1_2(k) * N2_2_k(q)
###


# %% local displacement and strain
def t_uv(k, q):
    return torch.stack([
        N1_00(k, q), N1_10(k, q), N1_01(k, q), N1_11(k, q), N1_20(k, q),
        N1_02(k, q), N1_21(k, q), N1_12(k, q), N1_22(k, q),
        N2_00(k, q), N2_10(k, q), N2_01(k, q), N2_11(k, q), N2_20(k, q),
        N2_02(k, q), N2_21(k, q), N2_12(k, q), N2_22(k, q),
        N3_00(k, q), N3_10(k, q), N3_01(k, q), N3_11(k, q), N3_20(k, q),
        N3_02(k, q), N3_21(k, q), N3_12(k, q), N3_22(k, q),
        N4_00(k, q), N4_10(k, q), N4_01(k, q), N4_11(k, q), N4_20(k, q),
        N4_02(k, q), N4_21(k, q), N4_12(k, q), N4_22(k, q)
    ])


def t_uvx(k, q):
    return torch.stack([
        N1_00_x(k, q), N1_10_x(k, q), N1_01_x(k, q), N1_11_x(k, q),
        N1_20_x(k, q), N1_02_x(k, q), N1_21_x(k, q), N1_12_x(k, q),
        N1_22_x(k, q),
        N2_00_x(k, q), N2_10_x(k, q), N2_01_x(k, q), N2_11_x(k, q),
        N2_20_x(k, q), N2_02_x(k, q), N2_21_x(k, q), N2_12_x(k, q),
        N2_22_x(k, q),
        N3_00_x(k, q), N3_10_x(k, q), N3_01_x(k, q), N3_11_x(k, q),
        N3_20_x(k, q), N3_02_x(k, q), N3_21_x(k, q), N3_12_x(k, q),
        N3_22_x(k, q),
        N4_00_x(k, q), N4_10_x(k, q), N4_01_x(k, q), N4_11_x(k, q),
        N4_20_x(k, q), N4_02_x(k, q), N4_21_x(k, q), N4_12_x(k, q),
        N4_22_x(k, q)
    ])


def t_uvy(k, q):
    return torch.stack([
        N1_00_y(k, q), N1_10_y(k, q), N1_01_y(k, q), N1_11_y(k, q),
        N1_20_y(k, q), N1_02_y(k, q), N1_21_y(k, q), N1_12_y(k, q),
        N1_22_y(k, q),
        N2_00_y(k, q), N2_10_y(k, q), N2_01_y(k, q), N2_11_y(k, q),
        N2_20_y(k, q), N2_02_y(k, q), N2_21_y(k, q), N2_12_y(k, q),
        N2_22_y(k, q),
        N3_00_y(k, q), N3_10_y(k, q), N3_01_y(k, q), N3_11_y(k, q),
        N3_20_y(k, q), N3_02_y(k, q), N3_21_y(k, q), N3_12_y(k, q),
        N3_22_y(k, q),
        N4_00_y(k, q), N4_10_y(k, q), N4_01_y(k, q), N4_11_y(k, q),
        N4_20_y(k, q), N4_02_y(k, q), N4_21_y(k, q), N4_12_y(k, q),
        N4_22_y(k, q)
    ])


def __init_tensors__(a):
    x = torch.arange(-a, a) / a
    # t = torch.empty((3, 36) + (2*a,)*2, dtype=torch.float)

    xx, yy = torch.meshgrid(x, x, indexing='ij')
    xx, yy = xx.to(DEVICE), yy.to(DEVICE)
    t = torch.stack((
        t_uv(xx, yy),
        t_uvx(xx, yy) * (1. / a),
        t_uvy(xx, yy) * (-1. / a)
    ))
    return t.flip(3).swapaxes(1, 3)


filename = 'data/hermiteTensors.pt'
if os.path.isfile(filename):
    SHAPE_FUNC_TENSORS = torch.load(filename, map_location=DEVICE)
else:
    # compute tensors once -> speed up computation
    print('Hermite: computing shape tensors...', end='', flush=True)
    SHAPE_FUNC_TENSORS = {
        size: __init_tensors__(size//2) for size in [5, 9, 17, 33, 65]
    }
    print(' done.')
    torch.save(SHAPE_FUNC_TENSORS, filename)


# %% shape function
def shapeFunction(nodeData, size, out):
    t = SHAPE_FUNC_TENSORS[size]
    # a = size//2
    # x = torch.arange(0, a+a, device=DEVICE)
    # x = torch.arange(0, a, device=DEVICE)
    # xdata, ydata = nodeData[:, 0].ravel(), nodeData[:, 1].ravel()
    # # t : 3 x 36 x a*a x a*a
    # # xdata : 36
    # # out : 6 x a*a x a*a
    # for i in x:
    #     i2 = a + i
    #     for j in x:
    #         j2 = a + j
    #         # uu, uux  uuy
    #         out[:3, -1-j, i] = t[:, :, i, j] @ xdata
    #         out[:3, -1-j, i2] = t[:, :, i2, j] @ xdata
    #         out[:3, -1-j2, i] = t[:, :, i, j2] @ xdata
    #         out[:3, -1-j2, i2] = t[:, :, i2, j2] @ xdata
    #         # vv, vvx, vvy
    #         out[3:6, -1-j, i] = t[:, :, i, j] @ ydata
    #         out[3:6, -1-j2, i] = t[:, :, i, j2] @ ydata
    #         out[3:6, -1-j, i2] = t[:, :, i2, j] @ ydata
    #         out[3:6, -1-j2, i2] = t[:, :, i2, j2] @ ydata
    out[:3] = t @ nodeData[:, 0].ravel()
    out[3:] = t @ nodeData[:, 1].ravel()
    # print('check x:', torch.abs(out[:3] - outx).sum())


def Hermite2D(
        maxDisp: float,
        nNodes: Tuple[int],
        elementSize: int,
        ):
    delta = elementSize - 1
    # u, ux, uy, v, vx, vy
    out = OUT0

    nodeData = torch.rand(nNodes + (2, 9), device=DEVICE) - 0.5

    # u, v
    nodeData[:, :, :, 0] *= maxDisp
    # ux, uy, vx, vy
    nodeData[:, :, :, 1:3] *= 0.06
    # nodeData[:, :, :, 2] *= 0.06
    # uxy, uxx, uyy + v
    nodeData[:, :, :, 3:6] *= 0.002
    # nodeData[:, :, :, 4] *= 0.002
    # nodeData[:, :, :, 5] *= 0.002
    # uxxy, uxyy
    nodeData[:, :, :, 6:8] *= 0.0002
    # nodeData[:, :, :, 7] *= 0.0002
    # uxxyy
    nodeData[:, :, :, 8] *= 0.00002

    vj = torch.arange(nNodes[1]-1, device=DEVICE)
    i0 = 0
    for i in torch.arange(nNodes[0]-1, device=DEVICE):
        i1, j0 = i0 + delta, 0
        for j in vj:
            j1 = j0 + delta
            shapeFunction(
                nodeData[IDX+i, IDY+j],
                elementSize,
                out[:, i0:i1, j0:j1]
            )
            j0 = j1
        i0 = i1
    return out


def Hermite2d_case(case):
    elementSize = ELEMENT_SIZES[case]
    return Hermite2D(
        MAX_DISPS[case],
        (
            1+OUT0.shape[1]//(elementSize-1),
            1+OUT0.shape[2]//(elementSize-1)
        ),
        elementSize
    )

# %% END OF FILE
###
