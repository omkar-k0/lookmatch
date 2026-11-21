import json

import numpy as np

from lookmatch import evaluate, sample, split


def test_split_is_frozen_and_disjoint():
    frozen = split.load()
    assert set(frozen["dev"]).isdisjoint(frozen["heldout"])
    assert len(frozen["dev"]) == len(frozen["heldout"]) == 5
    # a changed heldout look would fail here
    assert frozen["heldout"] == split.current()["heldout"]
    assert frozen["dev"] == split.current()["dev"]


def test_scene_pick_keeps_two_per_scene():
    rng = np.random.default_rng(1)
    a = [np.full((8, 8, 3), 0.2) + rng.random((8, 8, 3)) * 0.01 for _ in range(6)]
    b = [np.full((8, 8, 3), 0.8) + rng.random((8, 8, 3)) * 0.01 for _ in range(6)]
    keep = sample.pick(list(range(12)), a + b)
    assert len(keep) == 4 and sum(k < 6 for k in keep) == 2


def test_unusable_frames_are_dropped():
    assert sample.usable(np.zeros((8, 8, 3)))[1] == "too dark"
    assert sample.usable(np.ones((8, 8, 3)))[1] == "blown out"


def test_test_clips_are_deterministic_and_exclude_train():
    names = [f"c{i:03d}" for i in range(10)]
    t1, t2 = evaluate.test_clips_for("c003", names), evaluate.test_clips_for("c003", names)
    assert t1 == t2 and "c003" not in t1 and len(t1) == evaluate.K_TEST


def test_harness_end_to_end(tmp_path, monkeypatch):
    from skimage import data

    frames = tmp_path / "frames"
    frames.mkdir()
    for i, name in enumerate(["astronaut", "coffee", "chelsea", "rocket"]):
        im = getattr(data, name)()[..., :3]
        stack = np.stack([np.roll(im, k * 7, axis=1) for k in range(4)])
        np.savez_compressed(frames / f"c{i:03d}.npz", frames=stack, times=np.arange(4.0))
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--split",
            "dev",
            "--frames",
            str(frames),
            "--results",
            str(tmp_path / "res"),
            "--workers",
            "2",
        ],
    )
    evaluate.main()
    res = json.loads((tmp_path / "res" / "dev.json").read_text())
    s = res["summary"]["all"]
    assert res["pairs"] == 5 * 4 * 3  # 5 looks x 4 train clips x 3 test clips
    assert s["paired_lut"]["mean_de"] < s["none"]["mean_de"]
