"""Exporter unit test: uV -> V scaling in build_raw (skipped if MNE absent)."""

import numpy as np
import pytest

pytest.importorskip("mne")

from ganglion_studio.core import exporter


def test_build_raw_scales_uv_to_volts():
    sr = 200
    n = sr  # 1 second
    eeg = np.zeros((2, n), dtype=np.float64)
    eeg[0, :] = 100.0  # 100 uV constant
    meta = {"sampling_rate": sr, "eeg_channels": [0, 1], "channel_names": ["A", "B"]}

    raw = exporter.build_raw(eeg, meta)
    data = raw.get_data()  # volts

    assert data.shape == (2, n)
    assert raw.ch_names == ["A", "B"]
    # 100 uV -> 1e-4 V
    assert np.allclose(data[0], 1e-4, atol=1e-9)
    assert np.allclose(data[1], 0.0, atol=1e-12)
