"""M0 'done when': the spike's numbers come out of the package code."""

import os

import pytest

from lookmatch import spike


@pytest.mark.skipif(os.environ.get("LOOKMATCH_FAST") == "1", reason="slow (about 25 s)")
def test_spike_numbers_reproduce():
    s = spike.run()["summary"]
    assert s == {"none": 7.65, "reinhard": 21.93, "histogram": 23.57, "paired_lut": 3.08}
