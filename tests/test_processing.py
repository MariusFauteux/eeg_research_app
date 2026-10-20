"""Regression tests for channel selection in the offline processing pipeline."""

import numpy as np
import pytest

from ganglion_studio.core import processing
from ganglion_studio.core.processing import ProcessingConfig, apply_pipeline


@pytest.mark.parametrize("active", [None, [0, 1], [1], []])
def test_car_respects_explicit_active_channels(active):
    eeg = np.array([[1., 2., 4., 8.], [3., 5., 7., 9.]])
    before = eeg.copy()
    config = ProcessingConfig(reref_car=True, active_channels=active)
    config.filters.enabled = False

    out, messages = apply_pipeline(eeg, 200, ["a", "b"], config)

    rows = list(range(eeg.shape[0])) if active is None else active
    expected = eeg - eeg[rows].mean(axis=0) if rows else eeg
    np.testing.assert_allclose(out, expected)
    np.testing.assert_array_equal(eeg, before)
    assert any("Re-reference" in message for message in messages) == bool(rows)


def test_repair_preserves_the_valid_sample_after_a_one_sample_gap():
    # A cropped file can retain only the last held sample of a dropped packet.
    eeg = np.array([[0., 1., 99., 7., 9.]])
    config = ProcessingConfig(repair_gaps=True, loss_samples=[2], loss_lengths=[1])
    config.filters.enabled = False

    out, _messages = apply_pipeline(eeg, 199.8, ["a"], config)

    np.testing.assert_allclose(out, [[0., 1., 4., 7., 9.]])
    np.testing.assert_array_equal(eeg, [[0., 1., 99., 7., 9.]])


def test_repair_clips_gap_spans_and_defaults_legacy_lengths():
    eeg = np.array([[99., 1., 2., 99., 99., 5., 6., 99.]])
    config = ProcessingConfig(repair_gaps=True, loss_samples=[-1, 3, 7, 100],
                              loss_lengths=[2])
    config.filters.enabled = False

    out, messages = apply_pipeline(eeg, 200., ["a"], config)

    np.testing.assert_allclose(out, [[1., 1., 2., 3., 4., 5., 6., 6.]])
    assert messages == ["Repaired 3 packet-loss gaps (interpolated)"]


def test_aas_retains_fractional_rate_for_detection_and_epoch_width(monkeypatch):
    sr = 127.6
    eeg = np.ones((2, 600))
    seen_rates = []

    def detect_reference(channel, sampling_rate):
        seen_rates.append(sampling_rate)
        return np.array([100, 250, 400])

    monkeypatch.setattr(processing, "detect_rpeaks", detect_reference)
    config = processing.AasStepConfig(pre_ms=125, post_ms=125)

    out, message = processing.apply_aas(eeg, sr, config)

    assert seen_rates == [sr]
    # 125 ms is 15.95 samples: round the window width, not the rate to 127 Hz.
    assert out[0, 84] == 0
    assert out[0, 115] == 0
    assert out[0, 116] == 1
    assert "removed cardiac template" in message
