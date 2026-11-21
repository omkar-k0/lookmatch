"""Colour maths: Lab conversion and the CIEDE2000 score. Pure functions on arrays."""

from __future__ import annotations

import numpy as np
from skimage import color

from lookmatch.types import Score


def to_lab(rgb: np.ndarray) -> np.ndarray:
    return color.rgb2lab(rgb)


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel CIEDE2000 between two RGB images (any leading shape, last axis 3)."""
    return color.deltaE_ciede2000(to_lab(a), to_lab(b))


def score(pred: np.ndarray, truth: np.ndarray) -> Score:
    d = delta_e(pred, truth)
    return Score(mean_de=float(d.mean()), share_under_3=float((d < 3.0).mean()), n_pixels=int(d.size))
