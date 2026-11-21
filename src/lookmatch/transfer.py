"""Unpaired baselines: copy colour statistics from a reference of a different scene."""

from __future__ import annotations

import warnings

import numpy as np
from skimage import color, exposure


def _to_rgb(lab: np.ndarray) -> np.ndarray:
    # Transferred Lab values can fall outside sRGB; skimage warns and clips. That is expected here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return np.clip(color.lab2rgb(lab), 0.0, 1.0)


def reinhard(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match Lab mean and standard deviation (Reinhard et al. 2001). Works on (h, w, 3) images."""
    s, r = color.rgb2lab(src), color.rgb2lab(ref)
    axes = tuple(range(s.ndim - 1))
    out = (s - s.mean(axes)) / (s.std(axes) + 1e-6) * r.std(axes) + r.mean(axes)
    return _to_rgb(out)


def histogram(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-channel Lab histogram matching to the reference."""
    s, r = color.rgb2lab(src), color.rgb2lab(ref)
    out = exposure.match_histograms(s, r, channel_axis=-1)
    return _to_rgb(out)
