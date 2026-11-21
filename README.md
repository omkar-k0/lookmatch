# lookmatch

Upload a clip, get its colour look back as a LUT you can load in your editor, with a score for how close the match is.

**Status: M0 (scaffold).** The 17 Nov spike now runs on the package code and reproduces its numbers exactly
(`make spike`): on 6 still images, a recovered 33-point LUT cuts the error from 7.65 (do nothing) to 3.08 mean
CIEDE2000; copying the look from a different scene with simple statistics makes it worse (21.93 and 23.57).
No eval on real footage yet: that is M1. Milestones and rules are in [CLAUDE.md](CLAUDE.md).

## Run it

```bash
uv sync
make check    # ruff + pytest (includes the spike reproduction, about 25 s)
make spike    # print the spike table from the package code
```

## Layout

```
src/lookmatch/types.py      FrameSet, Lut, Score, Decision (every result says method + version)
src/lookmatch/colour.py     Lab conversion and the CIEDE2000 score
src/lookmatch/lut.py        identity, apply (trilinear), paired fit (binning + nearest fill)
src/lookmatch/transfer.py   unpaired baselines: Reinhard, histogram matching
src/lookmatch/cube.py       .cube read and write (red changes fastest)
src/lookmatch/looks.py      10 looks written for the eval (no third-party presets)
src/lookmatch/spike.py      the spike, rerun on the package
scripts/fetch_footage.py    openly licensed clips from Wikimedia Commons, with a licence manifest
```

The original spike script stays in `spike/` for reference.
