#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import numpy as np
import torch
from torch import pi
from torchvision import transforms
import torch.nn.functional as F

import funcs

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
###

# TODO : add noise & add extra noise whe nloading images

# %%
kernels, kernel_dx, dx0 = funcs.load_pickle('data/interp_kernels')
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
print("kernels:", kernels.shape)

kernels = kernels.reshape(kernels.shape[0], kernels.shape[-1], 1)
print('kernels:', kernels.shape)
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


def interp_img(img, x, y):
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

        xy1 = Rot[i] @ (xy0 - dxy[:,i:i+1])
        b_in = (xy1**2 / rxy[:, i:i+1]**2).sum(0) < 1.
        # print('y0:', y0.shape)
        # b_in = (xy1[0]**2 / rxy[0,i]**2 + xy1[1]**2 / ry[i]**2) < 1.
        y = y0[b_in]+int(xyc[1, i])
        x = x0[b_in]+int(xyc[0, i])
        b_ok = (y > -1) & (y < shape[0]) & (x > -1) & (x < shape[1])
        img[y[b_ok], x[b_ok]] = grayscale[i]

    return


def generate_image(shape, scale, sparse, largeSpeckles, lowContrast):
    img = torch.zeros(tuple(n*scale for n in shape), dtype=torch.float)
    factor = shape[0] * shape[1] / 512**2
    n_ell = torch.randint(
        low=int(2800*factor),
        high=int(4500*factor),
        size=(1,),
    ).item()

    # rmin = 0.6 * scale
    # rmax = (1.75 if sparse else 3.4) * scale
    rmin = 1.2 * scale
    rmax = (3.5 if sparse else 6.8) * scale
    graymax = 0.6 if lowContrast else 0.9

    generate_speckles(img, n_ell, rmin, rmax, graymax)
    if largeSpeckles:
        n_ell = torch.randint(
            low=max(1, int(factor + 0.5)),
            high=max(5, int(factor*5 + 0.5)),
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
    def __init__(self, Iref, size, u0, v0):
        x = torch.arange(size) - (size-1) * 0.5
        yy, xx = torch.meshgrid(x, x, indexing='ij')
        yy, xx = yy.flatten().to(DEVICE), xx.flatten().to(DEVICE)
        phi = torch.empty(
            (6, size*size), dtype=torch.float, device=DEVICE
        )
        # 2nd order 2D polynomial function
        phi[0, :] = 1.
        compute_phi_reduced(phi, xx, yy)
        self.phi = phi
        self.phi_tmp = phi.clone()
        # coeffients corresponding to identity centered on (u0, v0)
        self.p = torch.tensor([
            [u0, 1., 0., 0., 0., 0.],
            [v0, 0., 1., 0., 0., 0.]
        ], device=DEVICE)
        # estimator for new p when composing p(p'(phi))
        # torch.linalg.lstsq(A, B).solution == A.pinv() @ B
        self.p_estimator = torch.linalg.lstsq(
            phi @ phi.transpose(0, 1), phi
        ).solution.transpose(0, 1)

        uv = self.get_uv()
        uv_int = (uv + 0.5).int()
        if torch.abs(uv - uv_int).sum() > 1e-5:
            print('\n!!! CORRELATOR: initial uv are not integers !!! \n')

        Iref = Iref.to(DEVICE)
        Ix, Iy = dxy_img(Iref, uv_int[0], uv_int[1])
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

    def get_uv(self):
        return p_xy_reduced(self.phi, self.p)

    def estim_dp(self, Idef):
        uv = self.get_uv()
        G = interp_img(Idef, uv[0], uv[1])
        G -= G.mean()
        G *= self.F.dot(G) / G.dot(G)

        dp = self.dp_estimator @ (self.F - G)
        dp[0, 1] += 1.
        dp[1, 2] += 1.
        return dp

    def compute_p_dp(self, dp):
        xy = p_xy_reduced(self.phi, dp)
        return p_xy_reduced(
            compute_phi_reduced(self.phi_tmp, xy[0], xy[1]),
            self.p
        )

    def update_p(self, dp):
        print(self.p_estimator.shape)
        print(self.compute_p_dp(dp).shape)
        self.p = self.compute_p_dp(dp) @ self.p_estimator
        print('self.p:', self.p.shape)

###
# %% END OF FILE
###
