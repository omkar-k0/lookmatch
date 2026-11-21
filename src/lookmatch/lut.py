"""3D LUT: identity, apply (trilinear) and the paired fit (binning + nearest fill)."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from lookmatch.types import Lut

DEFAULT_SIZE = 33


def grid(n: int = DEFAULT_SIZE) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def identity(n: int = DEFAULT_SIZE) -> Lut:
    g = grid(n)
    return Lut(np.stack(np.meshgrid(g, g, g, indexing="ij"), -1), method="identity")


def apply(lut: Lut, rgb: np.ndarray) -> np.ndarray:
    """Apply a LUT to RGB values in [0, 1] of any leading shape."""
    g = grid(lut.size)
    interp = RegularGridInterpolator((g, g, g), lut.table, bounds_error=False, fill_value=None)
    flat = np.clip(rgb.reshape(-1, 3), 0.0, 1.0)
    return np.clip(interp(flat).reshape(rgb.shape), 0.0, 1.0)


def fit_paired(src: np.ndarray, dst: np.ndarray, n: int = DEFAULT_SIZE) -> Lut:
    """Fit a LUT from ungraded src and graded dst pixels (same shape, last axis 3).

    Each cell takes the mean graded colour of the source pixels nearest to it. Empty cells
    take the identity plus the offset of the nearest filled cell, so colours the source never
    showed get a plausible shift instead of a jump. meta['coverage'] is the share of filled cells.
    """
    if src.shape != dst.shape:
        raise ValueError("src and dst must have the same shape")
    idx = np.clip(np.rint(src.reshape(-1, 3) * (n - 1)).astype(np.int64), 0, n - 1)
    flat = (idx[:, 0] * n + idx[:, 1]) * n + idx[:, 2]
    cnt = np.bincount(flat, minlength=n**3).astype(np.float64)
    sums = np.stack(
        [np.bincount(flat, weights=dst.reshape(-1, 3)[:, c], minlength=n**3) for c in range(3)], -1
    )
    ident = identity(n).table.reshape(-1, 3)
    filled = cnt > 0
    offset = np.zeros((n**3, 3))
    offset[filled] = sums[filled] / cnt[filled, None] - ident[filled]
    off3 = offset.reshape(n, n, n, 3)
    if not filled.all():
        _, inds = distance_transform_edt(~filled.reshape(n, n, n), return_indices=True)
        off3 = off3[inds[0], inds[1], inds[2]]
    table = np.clip(ident.reshape(n, n, n, 3) + off3, 0.0, 1.0)
    return Lut(table, method="paired_bin_nearest", meta={"coverage": float(filled.mean())})
