# lookmatch: project instructions

Upload a video, get its colour look back as a LUT you can load in your editor, with a
score for how close the match is. Later: find existing presets that look like it.
Portfolio project first; the cross-marketplace search is the business test.

The repo is judged on two things: the eval numbers in the README, and whether a
stranger can reproduce them with one command.

## Principles

1. **Confidence gating.** Plain colour maths first. A learned model only for cases the
   maths flags as low confidence. If both fail, the answer is "look not matchable
   from this clip". Never ship a LUT the system itself scores as bad.
2. **Every LUT carries a score.** Mean CIEDE2000 from a self-check, shown to the user.
3. **Own footage only in the eval.** Clips and LUTs we made ourselves, or public data
   whose licence allows it (record the licence). Never other creators' paid presets.
4. **Uploads are analysed, then deleted.** No user video is stored or used for training.
5. **Report numbers only from files in `eval/results/`.** Never type a metric by hand.

## Eval rules

- Ground truth is made, not labelled: apply a known LUT, then try to recover it.
- LUTs are split into dev and heldout. Heldout LUTs are frozen after M1 and are never
  opened while tuning thresholds or methods. Score heldout once per milestone.
- Main metric: mean CIEDE2000 between the true grade and the recovered grade, on frames
  the method did not see. Also report: share under 3.0, and accuracy per confidence band.
- Always report the "do nothing" baseline. A method that loses to it is reported, not hidden.
- Two cases, reported separately:
  - **paired**: user gives their ungraded clip and the graded version (same content)
  - **unpaired**: user gives only a reference with a different scene (the hard case)

## Stack

Python 3.12, `uv`, `numpy`, `scipy`, `scikit-image`, `pytest`, `ruff`. OpenCV or
PySceneDetect for frame sampling (ask before adding). Learned models only behind the
gate, and only with a licence that allows it.

## Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 | Scaffold | spike ported into `src/`, `make check` passes, CI green, README states status honestly |
| M1 | Eval set + harness | 40 own clips x 10 own LUTs, frame sampler, dev/heldout split frozen, `make eval` writes baselines (none, Reinhard, histogram, paired LUT) |
| M2 | Paired recovery | paired-case table on heldout, .cube export loads in Resolve and Premiere (checked by hand, screenshot in docs) |
| M3 | Unpaired case + gate | a confidence signal that actually predicts error (bands in README), learned fallback behind the gate, PST50 benchmark reported separately |
| M4 | Web demo | upload a clip, get LUT + score, file deleted after analysis |
| M5 | Find presets like this | only after affiliate feed access is confirmed; ranked matches with links |

V2 (later): edit breakdown (cuts, pacing, transitions, beat sync) and review of the user's own draft.

## Time

4 h a week until Omkar sets a budget. Target: M0 and M1 by Sun 20 Dec 2026, M2 by Tue 26 Jan 2027.
Never at the cost of a flagship deadline.

## GitHub workflow

Never push to `main`. One branch per milestone (`m<N>-<name>`), draft PR early, Omkar merges.
