import numpy as np
import pytest

from lookmatch import colour, cube, lut, transfer
from lookmatch.looks import LOOKS
from lookmatch.types import FrameSet, Lut

rng = np.random.default_rng(0)


def test_identity_lut_is_a_no_op():
    img = rng.random((32, 32, 3))
    assert np.allclose(lut.apply(lut.identity(), img), img, atol=1e-9)


def test_paired_fit_recovers_a_look_on_unseen_pixels():
    train, test = rng.random((20000, 3)), rng.random((5000, 3))
    look = LOOKS["teal_orange"]
    fitted = lut.fit_paired(train, look(train))
    err = colour.score(lut.apply(fitted, test), look(test))
    none = colour.score(test, look(test))
    assert err.mean_de < 0.25 * none.mean_de
    assert fitted.meta["coverage"] > 0.3


def test_fit_fills_cells_the_source_never_showed():
    dark = rng.random((5000, 3)) * 0.4  # only the dark corner of the cube
    fitted = lut.fit_paired(dark, dark * 0.9)
    assert np.isfinite(fitted.table).all()
    assert fitted.meta["coverage"] < 0.2


def test_cube_round_trip(tmp_path):
    fitted = lut.fit_paired(rng.random((5000, 3)), LOOKS["warm_film"](rng.random((5000, 3))))
    path = tmp_path / "x.cube"
    cube.write(fitted, path, title="test")
    back = cube.read(path)
    assert back.size == 33
    assert np.allclose(back.table, fitted.table, atol=1e-6)
    text = path.read_text().splitlines()
    assert "LUT_3D_SIZE 33" in text
    # .cube order: red changes fastest; second data row = (r=1, g=0, b=0)
    data = [ln for ln in text if ln[:1].isdigit()]
    assert np.allclose([float(v) for v in data[1].split()], fitted.table[1, 0, 0], atol=1e-6)


def test_every_look_stays_in_range():
    x = rng.random((64, 64, 3))
    for name, fn in LOOKS.items():
        y = fn(x)
        assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1, name


def test_unpaired_baselines_return_valid_images():
    a, b = rng.random((32, 32, 3)), rng.random((32, 32, 3))
    for f in (transfer.reinhard, transfer.histogram):
        y = f(a, b)
        assert y.shape == a.shape and 0 <= y.min() and y.max() <= 1


def test_types_reject_bad_shapes():
    with pytest.raises(ValueError):
        FrameSet(np.zeros((4, 4, 3)))
    with pytest.raises(ValueError):
        Lut(np.zeros((3, 3, 4, 3)), method="x")
