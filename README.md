# lookmatch

Upload a clip, get its colour look back as a LUT you can load in your editor, with a score for how close the match is.

**Status: M1 done (eval set + harness).** 47 real-world clips from Wikimedia Commons (public domain, CC0,
CC BY; credits in [eval/FOOTAGE.md](eval/FOOTAGE.md)) x 10 looks we wrote ourselves, 5 for development and
5 heldout. The paired LUT already recovers looks well on footage it never saw. Copying a look from a
different scene with simple statistics is worse than doing nothing, which is the M3 problem.
Not done yet: `.cube` checks in editors (M2), a confidence gate (M3), a web demo (M4).
Milestones and rules: [CLAUDE.md](CLAUDE.md).

## How the eval works

Ground truth is made, not labelled. Each look is applied to the sampled frames of every clip. Then, for each
look and each training clip:

- **Paired LUT** fits a 33-point LUT on the training clip's ungraded and graded frames (the user has both).
- **Reinhard** and **Histogram** only get the training clip's graded frames, a reference of a different
  scene (the user has only a reference).
- **Do nothing** is the baseline every method has to beat.

Every method is scored on 5 other clips it never saw: mean CIEDE2000 between its output and the true grade,
on frames 384 px wide. About 1 is barely visible, 2 to 3 is visible side by side, above 5 is clearly different.
The dev/heldout split of looks is frozen in [eval/split.json](eval/split.json) with a checksum of each look's
code; heldout is scored once per milestone.

## Results

The tables below are written by `make eval` from `eval/results/`; nothing is typed by hand.

<!-- EVAL:START -->
**Dev looks** (teal_orange, warm_film, cool_moody, bleach_bypass, faded_pastel): 47 clips, 1175 scored pairs, M1, 2026-11-21, frames `bf8fe4a44901688c`

| Method | Mean CIEDE2000 | Pairs under 3.0 | Pixels under 3.0 |
|---|---:|---:|---:|
| Do nothing | 7.66 | 0.3% | 5.2% |
| Reinhard (unpaired) | 15.86 | 0.5% | 5.0% |
| Histogram (unpaired) | 17.13 | 0.0% | 4.9% |
| Paired LUT | 1.33 | 95.5% | 89.9% |

| Look | Do nothing | Reinhard (unpaired) | Histogram (unpaired) | Paired LUT |
|---|---:|---:|---:|---:|
| teal_orange | 7.86 | 17.84 | 18.89 | 1.35 |
| warm_film | 5.77 | 15.26 | 16.24 | 1.24 |
| cool_moody | 9.24 | 17.23 | 18.86 | 1.49 |
| bleach_bypass | 6.09 | 16.05 | 17.95 | 1.19 |
| faded_pastel | 9.34 | 12.90 | 13.73 | 1.36 |

Paired LUT, does anything predict its error? Correlation with test error: self-check -0.03, LUT coverage -0.50.

**Heldout looks** (golden_hour, nordic_cold, sepia_vintage, cross_process, vibrant_pop): 47 clips, 1175 scored pairs, M1, 2026-11-21, frames `bf8fe4a44901688c`

| Method | Mean CIEDE2000 | Pairs under 3.0 | Pixels under 3.0 |
|---|---:|---:|---:|
| Do nothing | 7.36 | 3.6% | 12.2% |
| Reinhard (unpaired) | 17.18 | 0.3% | 3.0% |
| Histogram (unpaired) | 18.41 | 0.0% | 2.9% |
| Paired LUT | 1.62 | 89.1% | 85.2% |

| Look | Do nothing | Reinhard (unpaired) | Histogram (unpaired) | Paired LUT |
|---|---:|---:|---:|---:|
| golden_hour | 6.10 | 18.54 | 19.95 | 1.62 |
| nordic_cold | 8.23 | 14.50 | 15.54 | 1.30 |
| sepia_vintage | 8.36 | 14.41 | 15.54 | 1.65 |
| cross_process | 10.17 | 19.40 | 20.51 | 1.33 |
| vibrant_pop | 3.97 | 19.05 | 20.49 | 2.20 |

Paired LUT, does anything predict its error? Correlation with test error: self-check +0.05, LUT coverage -0.49.
<!-- EVAL:END -->

What the numbers say:

- **Paired works.** With both the ungraded and graded clip, a 33-point LUT lands under 3.0 on 89% of
  heldout pairs (mean 1.62, against 7.36 for doing nothing). Its weakest heldout look is `vibrant_pop`
  (2.20): strong saturation pushes colours to the edge of the cube, where the LUT had little data.
- **Unpaired does not work yet.** Copying colour statistics from a reference of a different scene is more
  than twice as bad as doing nothing, because it copies the scene along with the look. That is M3.
- **A first lead for the gate.** The LUT's own self-check does not predict its error on new clips
  (correlation about 0), but the share of LUT cells the training clip covered does (about -0.5:
  more coverage, less error). M3 has to turn that into confidence bands.
- The eval clips are openly licensed real-world footage instead of our own clips (decision D4,
  21 Nov 2026); the 10 looks are our own.

## Reproduce

```bash
uv sync
make footage   # downloads the 50 clips in eval/footage_manifest.csv (1.7 GB) and checks each sha256
make frames    # samples up to 8 usable frames per clip into data/frames (3 clips are too dark: 47 remain)
make eval      # dev looks; SPLIT=heldout make eval only once per milestone
make check     # ruff + pytest, what CI runs
```

A dev run takes about 30 minutes on 2 CPU cores.

## What the sampler does

24 fast-seek probes per clip; drops frames that are too dark, blown out or flat; splits scenes where the
colour histogram jumps; spreads 8 frames over the scenes; flags clips that look like log footage (2 of 47).
A few clips are graphics or old black-and-white film rather than camera footage; they stay in, because a
user can upload anything.

## Layout

```
src/lookmatch/types.py      FrameSet, Lut, Score, Decision (every result says method + version)
src/lookmatch/colour.py     Lab conversion and the CIEDE2000 score
src/lookmatch/lut.py        identity, apply (trilinear), paired fit (binning + nearest fill)
src/lookmatch/transfer.py   unpaired baselines: Reinhard, histogram matching
src/lookmatch/cube.py       .cube read and write (red changes fastest)
src/lookmatch/looks.py      10 looks written for the eval (no third-party presets)
src/lookmatch/sample.py     frame sampler
src/lookmatch/split.py      frozen dev/heldout split of looks
src/lookmatch/evaluate.py   the eval harness
src/lookmatch/report.py     writes the tables above from eval/results/
src/lookmatch/spike.py      the 17 Nov spike, rerun on the package (numbers reproduce exactly)
scripts/fetch_footage.py    openly licensed clips from Wikimedia Commons, with licence and sha256
```
