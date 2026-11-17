"""Feasibility spike: can we recover a colour look and measure how close we got?

Ground truth is created, not labelled: five hand-written "looks" are applied to
public-domain test images (scikit-image sample data). Each method then tries to
reproduce the look on images it never saw, and we score the result against the
true grade with CIEDE2000 (mean per pixel, lower is better; ~1 is barely visible,
~2-3 noticeable side by side, >5 clearly different).

Methods
  none        : do nothing (how far apart ungraded and graded are)
  reinhard    : unpaired. Only a graded reference of a DIFFERENT image is given.
                Match Lab mean and std (Reinhard et al. 2001).
  hist        : unpaired. Per-channel Lab histogram matching to the reference.
  paired_lut  : paired. Source + graded version of the same frame are given.
                Fit a 33-point 3D LUT by binning, then apply to unseen images.
Confidence for paired_lut = error on its own training frame (self-check).
"""
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from skimage import data, color, exposure, transform

N = 33
rng = np.random.default_rng(0)


def load_images():
    names = ["astronaut", "coffee", "chelsea", "rocket", "immunohistochemistry", "colorwheel"]
    imgs = {}
    for n in names:
        try:
            im = getattr(data, n)()
        except Exception:
            continue
        if im.ndim != 3:
            continue
        im = im[..., :3].astype(np.float64) / 255.0
        imgs[n] = transform.resize(im, (256, 256), anti_aliasing=True)
    return imgs


# ---- five ground-truth looks (functions on RGB in [0,1]) ----
def luma(x):
    return (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]


def teal_orange(x):
    l = luma(x)
    shadows = np.array([0.0, 0.08, 0.12]) * (1 - l)
    highs = np.array([0.12, 0.05, -0.06]) * l
    return np.clip(x + shadows + highs, 0, 1)


def warm_film(x):
    y = 0.06 + 0.9 * x  # lifted blacks
    y = y * np.array([1.06, 1.0, 0.88])
    l = luma(y)
    return np.clip(l + 0.8 * (y - l), 0, 1)


def cool_moody(x):
    y = np.clip((x - 0.5) * 1.25 + 0.45, 0, 1) ** 1.1
    return np.clip(y * np.array([0.9, 0.98, 1.1]), 0, 1)


def bleach_bypass(x):
    l = luma(x)
    y = l + 0.45 * (x - l)
    return np.clip((y - 0.5) * 1.35 + 0.5, 0, 1)


def faded_pastel(x):
    y = 0.15 + 0.75 * x
    l = luma(y)
    y = l + 0.7 * (y - l)
    return np.clip(y + np.array([0.02, 0.0, 0.03]), 0, 1)


LOOKS = {f.__name__: f for f in [teal_orange, warm_film, cool_moody, bleach_bypass, faded_pastel]}


# ---- LUT helpers ----
grid = np.linspace(0, 1, N)


def apply_lut(lut, img):
    interp = RegularGridInterpolator((grid, grid, grid), lut, bounds_error=False, fill_value=None)
    return np.clip(interp(img.reshape(-1, 3)).reshape(img.shape), 0, 1)


def fit_lut_paired(src, dst):
    """Average target colour per LUT cell; empty cells take identity + offset of nearest filled cell."""
    idx = np.clip(np.rint(src.reshape(-1, 3) * (N - 1)).astype(int), 0, N - 1)
    flat = idx[:, 0] * N * N + idx[:, 1] * N + idx[:, 2]
    sums = np.zeros((N ** 3, 3))
    cnt = np.zeros(N ** 3)
    np.add.at(sums, flat, dst.reshape(-1, 3))
    np.add.at(cnt, flat, 1)
    ident = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    offset = np.zeros((N ** 3, 3))
    filled = cnt > 0
    offset[filled] = sums[filled] / cnt[filled, None] - ident[filled]
    off3 = offset.reshape(N, N, N, 3)
    mask = filled.reshape(N, N, N)
    _, inds = distance_transform_edt(~mask, return_indices=True)
    off3 = off3[inds[0], inds[1], inds[2]]
    return np.clip(ident.reshape(N, N, N, 3) + off3, 0, 1), filled.mean()


def reinhard(src, ref):
    s, r = color.rgb2lab(src), color.rgb2lab(ref)
    out = (s - s.mean((0, 1))) / (s.std((0, 1)) + 1e-6) * r.std((0, 1)) + r.mean((0, 1))
    return np.clip(color.lab2rgb(out), 0, 1)


def hist_match(src, ref):
    s, r = color.rgb2lab(src), color.rgb2lab(ref)
    out = exposure.match_histograms(s, r, channel_axis=-1)
    return np.clip(color.lab2rgb(out), 0, 1)


def de2000(a, b):
    return float(color.deltaE_ciede2000(color.rgb2lab(a), color.rgb2lab(b)).mean())


def write_cube(lut, path, title):
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {N}\n')
        for b in range(N):          # .cube order: red changes fastest
            for g in range(N):
                for r in range(N):
                    f.write("%.6f %.6f %.6f\n" % tuple(lut[r, g, b]))


def main():
    imgs = load_images()
    names = list(imgs)
    rows = []
    for look, fn in LOOKS.items():
        for i, train in enumerate(names):          # leave-one-in: fit on one image, test on the rest
            src, ref = imgs[train], fn(imgs[train])
            lut, coverage = fit_lut_paired(src, ref)
            self_check = de2000(apply_lut(lut, src), ref)
            if look == "teal_orange" and train == names[0]:
                write_cube(lut, "teal_orange_recovered.cube", "teal_orange recovered by spike")
            for test in names:
                if test == train:
                    continue
                truth = fn(imgs[test])
                rows.append(dict(
                    look=look, train=train, test=test, coverage=round(coverage, 3), self_check=self_check,
                    none=de2000(imgs[test], truth),
                    reinhard=de2000(reinhard(imgs[test], ref), truth),
                    hist=de2000(hist_match(imgs[test], ref), truth),
                    paired_lut=de2000(apply_lut(lut, imgs[test]), truth),
                ))
    json.dump(rows, open("results.json", "w"), indent=1)

    methods = ["none", "reinhard", "hist", "paired_lut"]
    print(f"images: {names}\npairs scored: {len(rows)}\n")
    print("Mean CIEDE2000 by look (lower is better)")
    print(f"{'look':15s}" + "".join(f"{m:>12s}" for m in methods))
    for look in LOOKS:
        sub = [r for r in rows if r["look"] == look]
        print(f"{look:15s}" + "".join(f"{np.mean([r[m] for r in sub]):12.2f}" for m in methods))
    print(f"{'ALL':15s}" + "".join(f"{np.mean([r[m] for r in rows]):12.2f}" for m in methods))

    # confidence check: does the LUT's self-check / colour coverage predict its test error?
    cov = np.array([r["coverage"] for r in rows])
    err = np.array([r["paired_lut"] for r in rows])
    print(f"\npaired_lut: LUT cells covered by the training frame, mean {cov.mean():.3f}")
    print(f"corr(coverage, test error) = {np.corrcoef(cov, err)[0, 1]:+.2f}")
    lo, hi = np.quantile(cov, [1 / 3, 2 / 3])
    for name, m in [("low coverage", cov <= lo), ("mid coverage", (cov > lo) & (cov <= hi)), ("high coverage", cov > hi)]:
        print(f"  {name:14s} n={m.sum():3d}  mean error {err[m].mean():.2f}  share under 3.0: {(err[m] < 3).mean():.0%}")


if __name__ == "__main__":
    main()
