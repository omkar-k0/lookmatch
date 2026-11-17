# lookmatch

Upload a clip, get its colour look back as a LUT you can load in your editor, with a score for how close the match is.

**Status: feasibility spike only.** No app yet. The first test ([spike/SPIKE_RESULTS.md](spike/SPIKE_RESULTS.md)) shows that recovering a look from an ungraded and a graded frame works (mean CIEDE2000 7.65 with no correction, 3.08 with the recovered LUT, on 6 sample images). Copying a look from a different scene does not work yet with simple methods. Milestones and rules are in [CLAUDE.md](CLAUDE.md).

Run the spike:

```bash
pip install numpy scipy scikit-image
python spike/spike.py
```
