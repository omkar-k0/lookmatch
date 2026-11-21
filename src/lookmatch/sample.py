"""Frame sampler: a video file in, a small set of useful frames out.

How it picks frames, without decoding the whole video:
1. Probe the duration, then grab N_PROBE frames at evenly spaced timestamps (fast seek).
2. Drop unusable frames: too dark, blown out, or flat (almost no detail).
3. Split the rest into scenes where the colour histogram jumps between neighbours.
4. Keep up to PER_SCENE frames per scene, up to MAX_FRAMES in total, spread over the clip.

It also flags clips that look like log footage (very low contrast and saturation), because a
look built on log footage does not transfer to normal footage.
Usage: python -m lookmatch.sample data/footage   (writes data/frames/<clip_id>.npz)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

N_PROBE = 24
PER_SCENE = 2
MAX_FRAMES = 8
WIDTH = 512
CUT_THRESHOLD = 0.35  # total-variation distance between colour histograms


def ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def duration(path: Path) -> float:
    out = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    if not m:
        raise ValueError(f"no duration for {path}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def grab(path: Path, t: float, width: int = WIDTH) -> np.ndarray | None:
    """One frame at time t, scaled to `width`, as float RGB in [0, 1] (sRGB, 8-bit source)."""
    cmd = [
        ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-2",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    raw = subprocess.run(cmd, capture_output=True).stdout
    if not raw:
        return None
    h = len(raw) // (width * 3)
    if h == 0:
        return None
    return np.frombuffer(raw[: width * h * 3], np.uint8).reshape(h, width, 3).astype(np.float64) / 255.0


def usable(frame: np.ndarray) -> tuple[bool, str]:
    luma = frame @ np.array([0.2126, 0.7152, 0.0722])
    if luma.mean() < 0.06:
        return False, "too dark"
    if (frame > 0.98).all(-1).mean() > 0.3:
        return False, "blown out"
    if luma.std() < 0.03:
        return False, "flat"
    return True, ""


def histogram(frame: np.ndarray, bins: int = 8) -> np.ndarray:
    idx = np.clip((frame * bins).astype(int), 0, bins - 1)
    flat = (idx[..., 0] * bins + idx[..., 1]) * bins + idx[..., 2]
    h = np.bincount(flat.ravel(), minlength=bins**3).astype(float)
    return h / h.sum()


def looks_like_log(frames: list[np.ndarray]) -> bool:
    x = np.stack(frames)
    luma = x @ np.array([0.2126, 0.7152, 0.0722])
    sat = x.max(-1) - x.min(-1)
    return bool(luma.std() < 0.12 and sat.mean() < 0.12)


def pick(times: list[float], frames: list[np.ndarray]) -> list[int]:
    """Indices of the frames to keep: scene split by histogram jumps, PER_SCENE per scene."""
    hists = [histogram(f) for f in frames]
    scenes, cur = [], [0]
    for i in range(1, len(frames)):
        if 0.5 * np.abs(hists[i] - hists[i - 1]).sum() > CUT_THRESHOLD:
            scenes.append(cur)
            cur = []
        cur.append(i)
    scenes.append(cur)
    chosen = []
    for sc in scenes:
        step = max(1, len(sc) // PER_SCENE)
        chosen += sc[::step][:PER_SCENE]
    if len(chosen) > MAX_FRAMES:
        keep = np.linspace(0, len(chosen) - 1, MAX_FRAMES).round().astype(int)
        chosen = [chosen[k] for k in keep]
    return sorted(chosen)


def sample_clip(path: Path) -> dict:
    dur = duration(path)
    ts = list(np.linspace(0.05 * dur, 0.95 * dur, N_PROBE))
    times, frames, dropped = [], [], {}
    for t in ts:
        f = grab(path, t)
        if f is None:
            dropped["decode"] = dropped.get("decode", 0) + 1
            continue
        ok, why = usable(f)
        if not ok:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        times.append(float(t))
        frames.append(f)
    if not frames:
        return {"frames": None, "times": [], "dropped": dropped, "scenes": 0, "log_like": False}
    keep = pick(times, frames)
    sel = [frames[i] for i in keep]
    h = min(f.shape[0] for f in sel)
    arr = np.stack([f[:h] for f in sel])
    return {
        "frames": arr,
        "times": [times[i] for i in keep],
        "dropped": dropped,
        "log_like": looks_like_log(sel),
        "duration": dur,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("footage", type=Path, help="folder with clips/ and manifest.csv")
    p.add_argument("--out", type=Path, default=Path("data/frames"))
    p.add_argument("--limit", type=int, default=0, help="stop after this many new clips (0 = all)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.footage / "manifest.csv", encoding="utf-8")))
    report_path = a.out / "sample_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    done = 0
    for r in rows:
        cid = r["clip_id"]
        if cid in report:
            continue
        if a.limit and done >= a.limit:
            break
        res = sample_clip(a.footage / "clips" / r["file"])
        n = 0 if res["frames"] is None else len(res["frames"])
        if n:
            frames8 = np.round(res["frames"] * 255).astype(np.uint8)
            np.savez_compressed(a.out / f"{cid}.npz", frames=frames8, times=np.array(res["times"]))
        report[cid] = {
            "frames": n,
            "dropped": res["dropped"],
            "log_like": res["log_like"],
            "duration_s": round(res.get("duration", 0), 1),
        }
        report_path.write_text(json.dumps(report, indent=1))
        print(f"{cid}: {n} frames, dropped {res['dropped']}, log_like={res['log_like']}", flush=True)
        done += 1
    print(f"sampled {sum(1 for v in report.values() if v['frames'])} clips with frames", file=sys.stderr)


if __name__ == "__main__":
    main()
