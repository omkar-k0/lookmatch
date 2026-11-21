"""M1 eval harness: recover known looks on real footage and score them.

Ground truth is made, not labelled: each look in the split is applied to the sampled frames
of every clip. Then, for every (look, training clip):
  - paired_lut  fits a LUT on the training clip's ungraded and graded frames (the paired case)
  - reinhard / histogram only get the training clip's graded frames as a reference of a
                different scene (the unpaired case)
  - none        does nothing (the baseline every method must beat)
Each method is scored on K other clips it never saw: mean CIEDE2000 against the true grade.
The self-check (fit on all but 2 frames of the training clip, score on those 2) and the LUT's
cell coverage are recorded too; whether they predict test error is the M3 question.

Writes eval/results/<split>.json (summary) and eval/results/<split>_pairs.csv (every pair).
Usage: python -m lookmatch.evaluate --split dev
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from skimage.transform import resize

from lookmatch import __version__, colour, lut, split, transfer
from lookmatch.looks import LOOKS

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "eval" / "results"
METHODS = ["none", "reinhard", "histogram", "paired_lut"]
EVAL_WIDTH = 384
K_TEST = 5
SEED = 20261121

_CLIPS: dict[str, np.ndarray] = {}


def load_frames(frames_dir: Path, min_frames: int = 4) -> dict[str, np.ndarray]:
    clips = {}
    for f in sorted(frames_dir.glob("*.npz")):
        x = np.load(f)["frames"].astype(np.float64) / 255.0
        if len(x) < min_frames:
            continue
        h = int(round(x.shape[1] * EVAL_WIDTH / x.shape[2]))
        clips[f.stem] = np.stack([resize(fr, (h, EVAL_WIDTH), anti_aliasing=True) for fr in x])
    return clips


def test_clips_for(train: str, names: list[str]) -> list[str]:
    rng = np.random.default_rng(SEED + int(hashlib.sha256(train.encode()).hexdigest()[:8], 16))
    others = [n for n in names if n != train]
    return sorted(rng.choice(others, size=min(K_TEST, len(others)), replace=False).tolist())


def _init(clips: dict[str, np.ndarray]) -> None:
    _CLIPS.update(clips)


def _job(args: tuple[str, str]) -> list[dict]:
    look, train = args
    fn = LOOKS[look]
    names = sorted(_CLIPS)
    src = _CLIPS[train]
    ref = fn(src)
    fitted = lut.fit_paired(src, ref)
    held = lut.fit_paired(src[:-2], ref[:-2])
    self_check = colour.score(lut.apply(held, src[-2:]), ref[-2:]).mean_de
    rows = []
    for test in test_clips_for(train, names):
        x = _CLIPS[test]
        truth = fn(x)
        preds = {
            "none": x,
            "reinhard": transfer.reinhard(x, ref),
            "histogram": transfer.histogram(x, ref),
            "paired_lut": lut.apply(fitted, x),
        }
        row = {
            "look": look,
            "train": train,
            "test": test,
            "coverage": round(fitted.meta["coverage"], 4),
            "self_check": round(self_check, 4),
        }
        for m, p in preds.items():
            s = colour.score(p, truth)
            row[m] = round(s.mean_de, 4)
            row[f"{m}_px_under_3"] = round(s.share_under_3, 4)
        rows.append(row)
    return rows


def summarise(rows: list[dict], looks: list[str]) -> dict:
    def block(sub):
        return {
            m: {
                "mean_de": round(float(np.mean([r[m] for r in sub])), 2),
                "pairs_under_3": round(float(np.mean([r[m] < 3.0 for r in sub])), 3),
                "px_under_3": round(float(np.mean([r[f"{m}_px_under_3"] for r in sub])), 3),
            }
            for m in METHODS
        }

    sc = np.array([r["self_check"] for r in rows])
    err = np.array([r["paired_lut"] for r in rows])
    cov = np.array([r["coverage"] for r in rows])
    return {
        "all": block(rows),
        "by_look": {lk: block([r for r in rows if r["look"] == lk]) for lk in looks},
        "paired_lut_signals": {
            "corr_self_check_vs_test_error": round(float(np.corrcoef(sc, err)[0, 1]), 3),
            "corr_coverage_vs_test_error": round(float(np.corrcoef(cov, err)[0, 1]), 3),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "heldout"], default="dev")
    p.add_argument("--frames", type=Path, default=ROOT / "data" / "frames")
    p.add_argument("--milestone", default="M1")
    p.add_argument(
        "--force", action="store_true", help="rerun heldout although this milestone already scored it"
    )
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--results", type=Path, default=RESULTS)
    a = p.parse_args()
    results = a.results

    out = results / f"{a.split}.json"
    if a.split == "heldout" and out.exists() and not a.force:
        prev = json.loads(out.read_text())
        if prev.get("milestone") == a.milestone:
            raise SystemExit(
                f"heldout was already scored for {a.milestone}. Heldout is scored once per milestone."
            )

    looks = split.looks_for(a.split)
    clips = load_frames(a.frames)
    if len(clips) < 2:
        raise SystemExit(f"need at least 2 clips with frames in {a.frames}, found {len(clips)}")
    fingerprint = hashlib.sha256(
        "".join(
            f"{n}:{hashlib.sha256((a.frames / f'{n}.npz').read_bytes()).hexdigest()}" for n in sorted(clips)
        ).encode()
    ).hexdigest()[:16]

    jobs = [(lk, c) for lk in looks for c in sorted(clips)]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(clips,)) as ex:
        for i, rs in enumerate(ex.map(_job, jobs, chunksize=4), 1):
            rows += rs
            if i % 20 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} fits", flush=True)

    results.mkdir(parents=True, exist_ok=True)
    with open(results / f"{a.split}_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = {
        "split": a.split,
        "milestone": a.milestone,
        "version": __version__,
        "date": dt.date.today().isoformat(),
        "looks": looks,
        "clips": len(clips),
        "frames_fingerprint": fingerprint,
        "pairs": len(rows),
        "settings": {
            "eval_width": EVAL_WIDTH,
            "k_test_clips": K_TEST,
            "seed": SEED,
            "lut_size": lut.DEFAULT_SIZE,
        },
        "summary": summarise(rows, looks),
    }
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["summary"]["all"], indent=1))


if __name__ == "__main__":
    main()
