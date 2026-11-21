"""Ten colour looks written for the eval. Each is a function on RGB in [0, 1].

These are our own looks, so the eval never uses anyone's paid presets. The first five are
the spike's looks; the other five were added for M1. Which looks are dev and which are
heldout is fixed in eval/split.json, not here.
"""

from __future__ import annotations

import numpy as np


def _luma(x: np.ndarray) -> np.ndarray:
    return (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]


def _sat(x: np.ndarray, k: float) -> np.ndarray:
    lum = _luma(x)
    return lum + k * (x - lum)


def _scurve(x: np.ndarray, k: float) -> np.ndarray:
    """Smooth contrast curve around 0.5; k > 0 adds contrast."""
    return np.clip(0.5 + (x - 0.5) * (1 + k) - k * 4 * (x - 0.5) ** 3, 0, 1)


def teal_orange(x):
    lum = _luma(x)
    return np.clip(x + np.array([0.0, 0.08, 0.12]) * (1 - lum) + np.array([0.12, 0.05, -0.06]) * lum, 0, 1)


def warm_film(x):
    y = (0.06 + 0.9 * x) * np.array([1.06, 1.0, 0.88])
    return np.clip(_sat(y, 0.8), 0, 1)


def cool_moody(x):
    y = np.clip((x - 0.5) * 1.25 + 0.45, 0, 1) ** 1.1
    return np.clip(y * np.array([0.9, 0.98, 1.1]), 0, 1)


def bleach_bypass(x):
    return np.clip((_sat(x, 0.45) - 0.5) * 1.35 + 0.5, 0, 1)


def faded_pastel(x):
    y = _sat(0.15 + 0.75 * x, 0.7)
    return np.clip(y + np.array([0.02, 0.0, 0.03]), 0, 1)


def golden_hour(x):
    lum = _luma(x)
    y = x + np.array([0.10, 0.05, -0.04]) * lum**0.7
    return np.clip(_sat(_scurve(np.clip(y, 0, 1), 0.15), 1.1), 0, 1)


def nordic_cold(x):
    y = 0.04 + 0.92 * _sat(x, 0.6)
    return np.clip(y * np.array([0.92, 1.0, 1.08]) + np.array([0.0, 0.01, 0.03]), 0, 1)


def sepia_vintage(x):
    lum = _luma(x)
    sepia = np.concatenate([lum * 1.07 + 0.05, lum * 0.95 + 0.02, lum * 0.78], axis=-1)
    return np.clip(0.4 * x + 0.6 * sepia, 0, 1)


def cross_process(x):
    r = np.clip(_scurve(x[..., :1], 0.35), 0, 1)
    g = np.clip(x[..., 1:2] ** 0.9, 0, 1)
    b = np.clip(0.15 + 0.7 * x[..., 2:3], 0, 1)
    return np.concatenate([r, g, b], axis=-1)


def vibrant_pop(x):
    return np.clip(_scurve(_sat(x, 1.35), 0.25), 0, 1)


LOOKS = {
    f.__name__: f
    for f in [
        teal_orange,
        warm_film,
        cool_moody,
        bleach_bypass,
        faded_pastel,
        golden_hour,
        nordic_cold,
        sepia_vintage,
        cross_process,
        vibrant_pop,
    ]
}
