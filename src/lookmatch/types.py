"""The four objects every stage passes on. Each carries method and version so any
result file says exactly what produced it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from lookmatch import __version__


@dataclass(frozen=True)
class FrameSet:
    """Frames as float RGB in [0, 1], shape (n, h, w, 3), with their timestamps in seconds."""

    frames: np.ndarray
    times: tuple[float, ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        f = self.frames
        if f.ndim != 4 or f.shape[-1] != 3:
            raise ValueError(f"frames must be (n, h, w, 3), got {f.shape}")
        if f.dtype.kind != "f" or f.min() < 0 or f.max() > 1:
            raise ValueError("frames must be float RGB in [0, 1]")

    def __len__(self) -> int:
        return self.frames.shape[0]


@dataclass(frozen=True)
class Lut:
    """A 3D LUT of shape (n, n, n, 3), indexed [r, g, b], values in [0, 1]."""

    table: np.ndarray
    method: str
    version: str = __version__
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = self.table
        if t.ndim != 4 or t.shape[-1] != 3 or not (t.shape[0] == t.shape[1] == t.shape[2]):
            raise ValueError(f"LUT must be (n, n, n, 3), got {t.shape}")

    @property
    def size(self) -> int:
        return self.table.shape[0]


@dataclass(frozen=True)
class Score:
    """Mean CIEDE2000 and the share of pixels under 3.0, on frames the method did not fit."""

    mean_de: float
    share_under_3: float
    n_pixels: int


@dataclass(frozen=True)
class Decision:
    outcome: Literal["ship", "fallback", "not_matchable"]
    reason: str
    score: Score | None = None
