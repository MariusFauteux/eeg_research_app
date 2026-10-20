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


def test_build_raw_preserves_recorded_signal_types():
    data = np.ones((4, 200), dtype=np.float64)
    meta = {
        "sampling_rate": 200,
        "eeg_channels": [0, 1, 2, 3],
        "channel_names": ["Brain", "Heart", "Muscle", "Other"],
        "channel_types": ["EEG", "ECG", "EMG", "MISC"],
    }

    raw = exporter.build_raw(data, meta)

    assert raw.get_channel_types() == ["eeg", "ecg", "emg", "misc"]
    assert np.allclose(raw.get_data(), 1e-6)


def test_edf_export_preserves_misc_voltage_amplitudes(tmp_path):
    pytest.importorskip("edfio")
    import mne

    t = np.arange(200) / 200.0
    data = np.vstack([10.0 * np.sin(2 * np.pi * 10 * t),
                      100.0 * np.cos(2 * np.pi * 10 * t)])
    meta = {
        "sampling_rate": 200,
        "eeg_channels": [0, 1],
        "channel_names": ["Brain", "Other"],
        "channel_types": ["EEG", "MISC"],
    }

    path = exporter.export(str(tmp_path / "mixed.edf"), "edf", data, meta)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")

    # EDF has 16-bit quantisation; both signals must keep their original uV scale.
    assert np.allclose(raw.get_data() * 1e6, data, atol=0.004)
