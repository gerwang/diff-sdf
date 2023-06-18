from __future__ import annotations

from typing import List

import drjit as dr
import mitsuba as mi


class VonMisesFisher:
    def __init__(self, mu: mi.Vector3f, kappa: mi.Float):
        super().__init__()
        self.mu = mu
        self.kappa = kappa

    @classmethod
    def from_theta_phi(cls, theta: mi.Float, phi: mi.Float, kappa: mi.Float):
        x, y, z = square_to_sphere(theta, phi)
        return cls(mi.Vector3f(x, y, z), kappa)

    def sample_direction(self, sample: mi.Point2f):
        frame = mi.Frame3f(self.mu)
        d = mi.warp.square_to_von_mises_fisher(sample, self.kappa)
        d = frame.to_world(d)
        return d

    def pdf_direction(self, d: mi.Vector3f):
        frame = mi.Frame3f(self.mu)
        return mi.warp.square_to_von_mises_fisher_pdf(frame.to_local(d), self.kappa)


def affine_left(M, x, is_pos=True):
    res = x @ M[:3, :3].t()
    if is_pos:
        res += M[:3, 3]
    return res


def affine_right(x, M, is_pos=True):
    res = x @ M[:3, :3]
    if is_pos:
        res /= x @ M[:3, 3:] + 1
    return res


def kernel_gather(value: List[mi.Float], index: mi.Int | int, active: bool = True):
    if isinstance(index, int):
        return value[index]
    res = mi.Float(0.)
    for i in range(len(value)):
        mask = active & dr.eq(index, i)
        res[mask] = value[i]
    return res


def kernel_gather_vmf(value: List[VonMisesFisher], index: mi.Int | int, active: bool = True):
    if isinstance(index, int):
        return value[index]
    res = VonMisesFisher(mi.Vector3f(0.), mi.Float(0.))
    for i in range(len(value)):
        mask = active & dr.eq(index, i)
        res.mu[mask] = value[i].mu
        res.kappa[mask] = value[i].kappa
    return res


def kernel_linear_search(cdf: List[mi.Float], value: mi.Float, active: bool = True) -> mi.Int:
    res = mi.Int(0)
    for i in range(len(cdf)):
        mask = active & (cdf[i] >= value)
        res[mask] = i
        active = active & ~mask
    return res


def sigmoid(x):
    return 1.0 / (1.0 + dr.exp(-x))


def trunc_exp(x):
    return dr.exp(dr.clamp(x, -10, 10)) + 1e-5


def square_to_sphere(u, v):
    theta = dr.pi * u
    phi = 2 * dr.pi * v
    sin_theta, cos_theta = dr.sincos(theta)
    sin_phi, cos_phi = dr.sincos(phi)
    x = sin_theta * cos_phi
    y = sin_theta * sin_phi
    z = cos_theta
    return x, y, z
