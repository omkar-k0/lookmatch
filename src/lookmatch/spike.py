"""The 17 Nov feasibility spike, rerun on the package code (M0: the numbers must reproduce).

6 scikit-image sample images at 256 x 256, the 5 spike looks, fit on one image, test on the
other five. Expected means: none 7.65, reinhard 21.93, histogram 23.57, paired LUT 3.08.
"""

from __future__ import annotations

import json

import numpy as np
from skimage import data, transform

from lookmatch import colour, lut, transfer
from lookmatch.looks import LOOKS

SPIKE_LOOKS = ["teal_orange", "warm_film", "cool_moody", "bleach_bypass", "faded_pastel"]
IMAGES = ["astronaut", "coffee", "chelsea", "rocket", "immunohistochemistry", "colorwheel"]


def load_images() -> dict[str, np.ndarray]:
    out = {}
    for name in IMAGES:
        im = getattr(data, name)()
        if im.ndim != 3:
            continue
        im = im[..., :3].astype(np.float64) / 255.0
        out[name] = transform.resize(im, (256, 256), anti_aliasing=True)
    return out


def run() -> dict:
    imgs = load_images()
    names = list(imgs)
    rows = []
    for look in SPIKE_LOOKS:
        fn = LOOKS[look]
        for train in names:
            src, ref = imgs[train], fn(imgs[train])
            fitted = lut.fit_paired(src, ref)
            for test in names:
                if test == train:
                    continue
                truth = fn(imgs[test])
                rows.append(
                    {
                        "look": look,
                        "train": train,
                        "test": test,
                        "none": colour.score(imgs[test], truth).mean_de,
                        "reinhard": colour.score(transfer.reinhard(imgs[test], ref), truth).mean_de,
                        "histogram": colour.score(transfer.histogram(imgs[test], ref), truth).mean_de,
                        "paired_lut": colour.score(lut.apply(fitted, imgs[test]), truth).mean_de,
                    }
                )
    methods = ["none", "reinhard", "histogram", "paired_lut"]
    summary = {m: round(float(np.mean([r[m] for r in rows])), 2) for m in methods}
    return {"pairs": len(rows), "summary": summary, "rows": rows}


def main() -> None:
    res = run()
    print(json.dumps({"pairs": res["pairs"], "summary": res["summary"]}, indent=1))


if __name__ == "__main__":
    main()
