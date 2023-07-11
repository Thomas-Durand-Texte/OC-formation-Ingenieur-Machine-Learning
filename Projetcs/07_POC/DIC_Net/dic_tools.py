#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
from typing import Callable, Union, Tuple, List, Dict
import numpy as np
import torch
from torch import pi, tensor

from torchvision import transforms
import torch.nn.functional as F

import funcs

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
###

# TODO : add noise & add extra noise whe nloading images

INTERP_DATA = None
IMG0 = torch.empty((2048,) * 2, dtype=torch.float)


# %%
def load_interp_kernels(filename: str = None):
    if filename is None:
        filename = 'data/interp_kernels'
    kernels, kernel_dx, dx0 = funcs.load_pickle(filename)
    if False:
        print('kernels:', kernels.shape)
        print('kernel_dx:', kernel_dx.shape)
        print('dx0:', dx0)
    size = kernels.shape[1]
    n = size//2

    kernel_dx /= kernels[0].sum()
    kernels /= kernels[0].sum()
    # kerenels are from 0 to +0.5
    # to improve calculation speed, as memory usage is relatively low
    # use symmetry in the kernels to compute -0.5 - 0.5 kernels
    kernels = torch.concat((kernels[1:].flip((0, 1)), kernels))
    # print("kernels:", kernels.shape)

    kernels = kernels.reshape(kernels.shape[0], kernels.shape[-1], 1)
    # print('kernels:', kernels.shape)
    kernels = kernels.to(DEVICE)
    kernel_dx = kernel_dx.reshape(1, 9, 1).to(DEVICE)

    v_slice0 = torch.arange(-n, n+1, device=DEVICE)
    _, x_slices = torch.meshgrid(v_slice0, v_slice0, indexing='ij')

    v_slice0_dx = torch.arange(-n-1, n+2, device=DEVICE)
    _, x_slices_dx = torch.meshgrid(v_slice0, v_slice0_dx, indexing='ij')
    _, x_slices_dy = torch.meshgrid(v_slice0_dx, v_slice0, indexing='ij')
    v_slice0 = v_slice0.reshape(1, 7, 1)
    v_slice0_dx = v_slice0_dx.reshape(1, 9, 1)

    del _
    global INTERP_DATA
    INTERP_DATA = (kernels, kernel_dx, dx0, v_slice0, v_slice0_dx,
                   x_slices, x_slices_dx, x_slices_dy)


def interp_img(img, x, y):
    (kernels, kernel_dx, dx0, v_slice0, v_slice0_dx,
     x_slices, x_slices_dx, x_slices_dy) = INTERP_DATA
    x0 = (x+0.5).int()
    i_dx = ((x-x0.float()+0.5)/dx0+0.5).int()
    # print('i_dx:', i_dx.min(), i_dx.max())

    y0 = (y+0.5).int()
    i_dy = ((y-y0.float()+0.5)/dx0+0.5).int()
    # print('i_dy:', i_dy.min(), i_dy.max())

    out = kernels[i_dy].reshape(len(x0), 1, 7) @ \
        (img[
                v_slice0 + y0.reshape(-1, 1, 1),
                x_slices + x0.reshape(-1, 1, 1)
            ]
         @ kernels[i_dx])
    return out.flatten()


def dxy_img(img, x, y):
    ic = kernels.shape[0]//2
    kernel = kernels[ic:ic+1]
    Ix = kernel.reshape(1, 1, 7) @ \
        (img[
                v_slice0 + y.reshape(-1, 1, 1),
                x_slices_dx + x.reshape(-1, 1, 1)
            ]
         @ kernel_dx)

    Iy = kernel_dx.reshape(1, 1, 9) @ \
        (img[
                v_slice0_dx + y.reshape(-1, 1, 1),
                x_slices_dy + x.reshape(-1, 1, 1)
            ]
         @ kernel)
    return Ix, Iy

###


# %%
def generate_speckles(img, n_ell, rmin, rmax, graymax):
    shape = torch.tensor(img.shape)

    values = torch.rand((6, n_ell))
    xyc = values[:2] * shape.reshape(2, 1)
    rxy = rmin + values[2:4] * (rmax-rmin)
    rxy *= rxy
    tta = values[4] * (pi*0.5)
    grayscale = 0.008 + values[5] * graymax

    dxy = xyc - xyc.int()
    cos_tta = torch.cos(tta)
    sin_tta = torch.sin(tta)
    Rot = torch.empty((n_ell, 2, 2))
    # Rot[:, [0, 0], [1, 1]] = cos_tta.reshape(-1, 1)
    Rot[:, 0, 0] = cos_tta
    Rot[:, 1, 1] = cos_tta
    Rot[:, 0, 1] = -sin_tta
    Rot[:, 1, 0] = sin_tta

    nmax = int(rmax+1.5)
    y0, x0 = torch.meshgrid(
        torch.arange(-nmax, nmax+1), torch.arange(-nmax, nmax+1), indexing='ij'
    )
    x0, y0 = x0.flatten(), y0.flatten()
    xy0 = torch.stack((x0, y0)).float()

    # print(xy0.shape)
    for i in np.arange(n_ell):
        # Rot = torch.tensor(
        #     [[np.cos(tta[i]), -np.sin(tta[i])],
        #     [np.sin(tta[i]), np.cos(tta[i])]], dtype=torch.float)

        xy1 = Rot[i] @ (xy0 - dxy[:, i:i+1])
        b_in = (xy1**2 / rxy[:, i:i+1]).sum(0) < 1.
        # print('y0:', y0.shape)
        # b_in = (xy1[0]**2 / rxy[0,i]**2 + xy1[1]**2 / ry[i]**2) < 1.
        y = y0[b_in]+int(xyc[1, i])
        x = x0[b_in]+int(xyc[0, i])
        b_ok = (y > -1) & (y < shape[0]) & (x > -1) & (x < shape[1])
        img[y[b_ok], x[b_ok]] = grayscale[i]

    return


def generate_image(
        shape: Tuple[int],
        scale: int,
        sparse: bool,
        largeSpeckles: bool,
        lowContrast: bool):
    # img = torch.zeros(tuple(n*scale for n in shape), dtype=torch.float)
    img = IMG0[:scale * shape[0], :scale*shape[1]]
    img[:, :] = 0.
    factor = shape[0] * shape[1] / 512**2
    n_ell = torch.randint(
        low=int(11200*factor),  # 2800*4
        high=int(18000*factor),  # 4500*4
        size=(1,),
    ).item()

    # rmin = 0.6 * scale
    # rmax = (1.75 if sparse else 3.4) * scale
    rmin = 0.6 * scale
    rmax = (1.75 if sparse else 3.4) * scale
    graymax = 0.6 if lowContrast else 0.9

    generate_speckles(img, n_ell, rmin, rmax, graymax)
    if largeSpeckles:
        low = max(1, int(factor + 0.5))
        n_ell = torch.randint(
            low=low,
            high=max(low+1, int(factor*5 + 0.5)),
            size=(1,)
        ).item()
        rmin = 6.5 * scale
        rmax = 10.5 * scale
        generate_speckles(img, n_ell, rmin, rmax, graymax)

    img = F.avg_pool2d(
        img.reshape((1,) + img.shape), kernel_size=scale,
        stride=scale, padding=0)
    img = transforms.functional.gaussian_blur(img, 5)[0]
    return img

###


def compute_phi_reduced(phi, x, y):
    phi[1, :] = x[:]
    phi[2, :] = y[:]
    phi[3, :] = x[:] * x[:]
    phi[4, :] = x[:] * y[:]
    phi[5, :] = y[:] * y[:]
    return phi


def p_xy_reduced(phi, p):
    return p[:, :1] + p[:, 1:] @ phi[1:]


# %%
class Correlator():
    def __init__(
            self, Iref: tensor,
            size: Tuple[int],
            u0: float,
            v0: float,
            factor_grad: float = None):
        x = torch.arange(size) - (size-1) * 0.5
        yy, xx = torch.meshgrid(x, x, indexing='ij')
        yy, xx = yy.flatten().to(DEVICE), xx.flatten().to(DEVICE)
        phi = torch.empty(
            (6, size*size), dtype=torch.float, device=DEVICE
        )
        # 2nd order 2D polynomial function
        phi[0, :] = 1.
        compute_phi_reduced(phi, xx, yy)
        self.uv0 = torch.tensor([u0, v0])
        self.max_dxy = 0.
        self.phi = phi
        self.phi_tmp = phi.clone()
        # coeffients corresponding to identity centered on (u0, v0)
        self.p = torch.empty((2, 6), device=DEVICE)
        self.reset_p()

        # estimator for new p when composing p(p'(phi))
        # torch.linalg.lstsq(A, B).solution == A.pinv() @ B
        self.p_estimator = torch.linalg.lstsq(
            phi[1:] @ phi[1:].transpose(0, 1), phi[1:]
        ).solution.transpose(0, 1)
        # self.p_estimator = (torch.linalg.pinv(
        #     phi @ phi.transpose(0, 1), hermitian=True
        # ) @ phi).transpose(0, 1)

        uv = self.get_uv()
        uv_int = (uv + 0.5).int()
        if torch.abs(uv - uv_int).sum() > 1e-5:
            print('\n!!! CORRELATOR: initial uv are not integers !!! \n')

        Iref = Iref.to(DEVICE)
        Ix, Iy = dxy_img(Iref, uv_int[0], uv_int[1])
        if factor_grad is not None:
            Ix, Iy = factor_grad*Ix, factor_grad*Iy
        phiIx = phi * Ix.reshape(1, -1)
        phiIy = phi * Iy.reshape(1, -1)
        A = torch.empty((12, 12), dtype=torch.float, device=DEVICE)
        A[:6, :6] = phiIx @ phiIx.transpose(0, 1)
        A[:6, 6:] = phiIx @ phiIy.transpose(0, 1)
        A[6:, :6] = A[:6, 6:].transpose(0, 1)
        A[6:, 6:] = phiIy @ phiIy.transpose(0, 1)

        self.dp_estimator = torch.linalg.lstsq(
            A, torch.concatenate((phiIx, phiIy))
        ).solution.reshape(2, 6, -1)

        # interp to be consistent with Ix, Iy
        self.F = interp_img(Iref, uv[0], uv[1])
        self.F -= self.F.mean()
        self.sumF2 = self.F.dot(self.F)

    def reset_p(self):
        self.p[:, :] = 0.
        self.p[:, 0] = self.uv0
        self.p[0, 1] = 1.
        self.p[1, 2] = 1.

    def get_uv(self):
        return p_xy_reduced(self.phi, self.p)

    def get_uvc(self):
        return self.p[:, 0]

    def compute_correlation(self, Idef):
        uv = self.get_uv()
        G = interp_img(Idef, uv[0], uv[1])
        G -= G.mean()
        return self.F.dot(G) / torch.sqrt(self.sumF2 * G.dot(G))

    def estim_dp(self, Idef):
        uv = self.get_uv()
        G = interp_img(Idef, uv[0], uv[1])
        G -= G.mean()
        # G *= self.F.dot(G) / G.dot(G)

        dp = self.dp_estimator @ (self.F - G)
        dp[0, 1] += 1.
        dp[1, 2] += 1.
        return dp

    def compute_p_dp(self, dp):
        xy = p_xy_reduced(self.phi, dp)
        self.max_dxy = torch.abs(xy - self.phi[1:3]).max()
        return p_xy_reduced(
            compute_phi_reduced(self.phi_tmp, xy[0], xy[1]),
            self.p
        )

    def update_p(self, dp):
        compute_phi_reduced(
            self.phi_tmp[:, :1], dp[0, :1], dp[1, :1]
        )
        uv2c = p_xy_reduced(self.phi_tmp[:, :1], self.p)
        uv2 = self.compute_p_dp(dp)
        self.p[:, :1] = uv2c
        self.p[:, 1:] = (uv2-uv2c) @ self.p_estimator

    def optim_correlation(self, Idef):
        k = 0
        self.max_dxy = 1e3
        while (self.max_dxy > 5e-4) and k < 30:
            self.update_p(self.estim_dp(Idef))
            k += 1
###
# %% END OF FILE
###
