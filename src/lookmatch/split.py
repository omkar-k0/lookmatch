"""The dev/heldout split of looks, frozen with a checksum of each look's source code.

The five spike looks were seen while building the spike, so they are dev. The five looks
added in M1 have never been used for tuning, so they are heldout. If a heldout look's code
changes, its checksum no longer matches eval/split.json and the tests fail.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from lookmatch.looks import LOOKS

SPLIT_FILE = Path(__file__).resolve().parents[2] / "eval" / "split.json"
DEV = ["teal_orange", "warm_film", "cool_moody", "bleach_bypass", "faded_pastel"]
HELDOUT = ["golden_hour", "nordic_cold", "sepia_vintage", "cross_process", "vibrant_pop"]


def look_hash(name: str) -> str:
    return hashlib.sha256(inspect.getsource(LOOKS[name]).encode()).hexdigest()[:16]


def current() -> dict:
    return {"dev": {n: look_hash(n) for n in DEV}, "heldout": {n: look_hash(n) for n in HELDOUT}}


def load() -> dict:
    return json.loads(SPLIT_FILE.read_text())


def looks_for(split: str) -> list[str]:
    frozen = load()
    if frozen[split] != current()[split]:
        raise RuntimeError(f"{split} looks changed since the split was frozen; see eval/split.json")
    return list(frozen[split])


if __name__ == "__main__":
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if SPLIT_FILE.exists():
        raise SystemExit("eval/split.json exists: the split is frozen. Delete it only with a written reason.")
    SPLIT_FILE.write_text(json.dumps({**current(), "frozen": "2026-11-21, end of M1"}, indent=1) + "\n")
    print(SPLIT_FILE.read_text())
