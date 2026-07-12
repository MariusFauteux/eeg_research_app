"""Unit tests for core.pulse_diagnostics on synthetic spike trains.

The whole point of the module is to tell a *systematic* 1 Hz pulse (clock-locked
and/or on a packet-loss seam) apart from a wandering biological rhythm, so the
tests build each kind of signal explicitly and assert the verdict.
"""

import numpy as np
import pytest

from ganglion_studio.core import pulse_diagnostics as pd

SR = 200
RNG = np.random.default_rng(0)


def _baseline(n):
    """A little pink-ish EEG-like background so MAD isn't degenerate."""
    return RNG.normal(0.0, 5.0, size=n)


def _spike_train(period_samples, n_seconds=20, jitter_samples=0, amp=400.0):
    """Background noise with a large spike every ``period_samples`` (+/- jitter)."""
    n = n_seconds * SR
    x = _baseline(n)
    pos = []
    p = period_samples
    while p < n - 1:
        j = int(RNG.normal(0, jitter_samples)) if jitter_samples else 0
        idx = min(n - 1, max(0, p + j))
        x[idx] += amp
        pos.append(idx)
        p += period_samples
    return x, np.array(pos)


def test_clean_1hz_is_frame_locked():
    x, pos = _spike_train(SR, jitter_samples=0)        # exactly 1.000 Hz
    res = pd.diagnose(x, SR)
    assert res.n_pulses >= 15
    assert res.rate_hz == pytest.approx(1.0, abs=0.05)
    assert res.jitter_ms < 15.0
    assert res.phase_r > 0.9
    # No loss flags supplied -> can't be "digital"; it's a frame-locked event.
    assert res.verdict == pd.VERDICT_FRAME_LOCKED


def test_pulses_on_loss_seams_is_digital():
    x, pos = _spike_train(SR, jitter_samples=0)
    loss = np.zeros(x.size)
    loss[pos] = 1.0                                     # a drop at every pulse
    res = pd.diagnose(x, SR, loss)
    assert res.loss_fraction > 0.9
    assert res.loss_fraction > 2 * res.loss_chance
    assert res.verdict == pd.VERDICT_DIGITAL


def test_loss_elsewhere_stays_frame_locked():
    x, pos = _spike_train(SR, jitter_samples=0)
    loss = np.zeros(x.size)
    loss[(pos + SR // 2) % x.size] = 1.0               # drops between pulses
    res = pd.diagnose(x, SR, loss)
    assert res.loss_fraction < 0.5
    assert res.verdict == pd.VERDICT_FRAME_LOCKED


def test_wandering_train_looks_biological():
    # ~0.9 Hz with big spacing jitter, like a heartbeat with HRV.
    x, pos = _spike_train(int(SR / 0.9), jitter_samples=25)
    res = pd.diagnose(x, SR)
    assert res.jitter_ms > 40.0
    assert res.verdict == pd.VERDICT_BIOLOGICAL


def test_no_pulse_on_pure_noise():
    x = _baseline(20 * SR)
    res = pd.diagnose(x, SR)
    assert res.n_pulses < 3
    assert res.verdict == pd.VERDICT_NO_PULSE


def test_detect_pulses_finds_the_spikes():
    x, pos = _spike_train(SR, jitter_samples=0)
    peaks = pd.detect_pulses(x, SR)
    # Every detected peak should be within a couple of samples of a real spike.
    assert peaks.size >= pos.size - 1
    for p in peaks:
        assert np.min(np.abs(pos - p)) <= 2


def test_pulse_triggered_average_is_centered():
    x, pos = _spike_train(SR, jitter_samples=0, amp=400.0)
    peaks = pd.detect_pulses(x, SR)
    t, avg = pd.pulse_triggered_average(x, peaks, SR, half_s=0.1)
    assert t.size and avg.size == t.size
    # The averaged waveform should peak at t=0 (the trigger).
    assert abs(t[int(np.argmax(np.abs(avg)))]) < 0.01


def test_empty_and_short_inputs_are_safe():
    assert pd.detect_pulses(np.array([]), SR).size == 0
    assert pd.detect_pulses(np.zeros(3), SR).size == 0
    res = pd.diagnose(np.zeros(10), SR)
    assert res.verdict == pd.VERDICT_NO_PULSE
    t, avg = pd.pulse_triggered_average(np.zeros(10), np.array([], dtype=int), SR)
    assert t.size == 0 and avg.size == 0
