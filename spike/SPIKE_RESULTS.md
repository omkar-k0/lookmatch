# Spike results, 17 Nov 2026

Run: `python spike.py` (writes `results.json` and one example `.cube`).
Setup: 6 public-domain scikit-image sample images, 5 hand-written looks, fit on one
image and test on each of the other five. 150 scored pairs. Metric: mean CIEDE2000,
lower is better (about 1 barely visible, 2 to 3 noticeable side by side, over 5 clearly different).

| Look | Do nothing | Reinhard (unpaired) | Histogram (unpaired) | Paired LUT |
|---|---|---|---|---|
| teal_orange | 7.05 | 24.31 | 25.94 | 1.75 |
| warm_film | 4.88 | 21.30 | 22.66 | 2.67 |
| cool_moody | 8.95 | 23.38 | 25.35 | 2.82 |
| bleach_bypass | 7.86 | 22.54 | 24.65 | 3.96 |
| faded_pastel | 9.51 | 18.11 | 19.24 | 4.22 |
| **All** | **7.65** | **21.93** | **23.57** | **3.08** |

## What it shows

1. **Paired recovery works.** Given an ungraded frame and its graded version, a fitted
   33-point LUT cuts the error from 7.65 to 3.08 on images it never saw.
2. **Simple unpaired transfer fails badly.** Copying colour statistics from a different
   scene is about three times worse than doing nothing (21.9 vs 7.7). It copies the
   scene's colours, not the look. This is the hard problem M3 has to solve, and why
   commercial tools use learned models.
3. **The first confidence signal failed.** LUT cell coverage correlated with error in
   the wrong direction (+0.58). The cause: the synthetic colour wheel covers many cells
   but is a bad training frame (paired error 7.03 when trained on it, about 1.5 to 2.9
   on real photos). Coverage is not a quality signal. M3 needs a better one.

## Limits of this spike

Still images, not video frames. Hand-written looks, not real creator LUTs. Six images.
Treat the numbers as a feasibility check, not results.
