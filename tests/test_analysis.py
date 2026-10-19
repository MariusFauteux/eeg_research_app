"""Known-spectrum integration and exact-rate offline analysis regressions."""

import numpy as np
import pytest

from ganglion_studio.core import analysis
from ganglion_studio.core.dsp import EEG_BANDS, integrate_band_power


@pytest.mark.parametrize("sampling_rate", [200., 199.8])
def test_flat_psd_band_powers_equal_exact_band_widths(sampling_rate):
    freqs = np.arange(129) * sampling_rate / 256

    powers = analysis._band_powers(freqs, np.ones_like(freqs))

    for name, low, high in EEG_BANDS:
        assert powers[name] == pytest.approx(high - low, abs=1e-12)
    assert sum(powers.values()) == pytest.approx(45 - .5)


def test_linear_psd_integrates_interpolated_band_edges_exactly():
    # These coarse bins leave whole bands (e.g. theta) between two PSD samples.
    freqs = np.array([0., 6., 20., 50.])
    psd = 2 * freqs + 3

    powers = analysis._band_powers(freqs, psd)

    for name, low, high in EEG_BANDS:
        expected = high**2 + 3 * high - low**2 - 3 * low
        assert powers[name] == pytest.approx(expected, abs=1e-12)


def test_band_integration_clips_at_available_frequency_limits():
    freqs = np.array([0., 6., 10.])
    psd = np.ones_like(freqs)

    powers = analysis._band_powers(freqs, psd)

    assert powers["Alpha"] == pytest.approx(2.)
    assert powers["Beta"] == 0
    assert powers["Gamma"] == 0
    assert integrate_band_power(np.array([]), np.array([]), 8, 13) == 0


@pytest.mark.parametrize("sr", [200, 200., 199.8])
def test_channel_metrics_recover_known_alpha_tone_power(sr):
    t = np.arange(4000) / sr
    signal = 3 * np.sin(2 * np.pi * 10 * t)
    before = signal.copy()

    metrics = analysis.channel_metrics(signal, sr)

    assert metrics["dominant_hz"] == pytest.approx(10, abs=.05)
    assert metrics["bands"]["Alpha"] == pytest.approx(4.5, rel=.01)
    assert metrics["rel_bands"]["Alpha"] > .99
    np.testing.assert_array_equal(signal, before)


def test_fractional_coherence_retains_exact_frequency_axis():
    sr = 199.8
    signal = np.random.default_rng(7).normal(size=3996)

    agreement = analysis.pair_agreement(signal, signal, sr)

    assert agreement["coh_freqs"][-1] == pytest.approx(sr / 2, abs=1e-12)
    assert agreement["mean_coherence_1_30"] == pytest.approx(1)
    assert agreement["rmse"] == 0
