"""DSP unit tests, including a regression for the Ganglion full-scale constant.

The railing detector previously used +/- 187500 uV (the *Cyton's* range), which a
real Ganglion signal can never reach -- so railing was dead code. These tests lock
in the correct +/- 15686 uV MCP3912 full-scale.
"""

import numpy as np
import pytest

from ganglion_studio.core import dsp


def test_full_scale_constant_is_ganglion_not_cyton():
    # MCP3912 (Vref 1.2 V, gain 51): 1.2e6 / (1.5 * 51) ~= 15686 uV.
    assert abs(dsp.GANGLION_FULLSCALE_UV - 1.2e6 / (1.5 * 51)) < 1.0
    # Guard against regressing to the Cyton's 187500 uV.
    assert dsp.GANGLION_FULLSCALE_UV < 20000.0


def test_is_railed_at_full_scale():
    sr = 200
    railed = np.full(sr, dsp.GANGLION_FULLSCALE_UV, dtype=np.float64)
    assert dsp.is_railed(railed) is True
    assert dsp.signal_quality(railed, sr)["railed"] is True


def test_clean_signal_not_railed():
    sr = 200
    t = np.arange(sr) / sr
    clean = 50.0 * np.sin(2 * np.pi * 10 * t)  # 50 uV, 10 Hz
    assert dsp.is_railed(clean) is False
    assert dsp.signal_quality(clean, sr)["railed"] is False


def test_is_railed_empty():
    assert dsp.is_railed(np.array([])) is False


def test_quality_label_boundaries():
    assert dsp.quality_label({"railed": True}) == "bad"
    assert dsp.quality_label({"ptp": 50.0, "line_ratio": 0.10}) == "good"
    assert dsp.quality_label({"ptp": 300.0, "line_ratio": 0.10}) == "ok"
    assert dsp.quality_label({"ptp": 50.0, "line_ratio": 0.30}) == "ok"
    assert dsp.quality_label({"ptp": 1500.0, "line_ratio": 0.10}) == "bad"
    assert dsp.quality_label({"ptp": 50.0, "line_ratio": 0.60}) == "bad"


def test_dominant_frequency_recovers_tone():
    sr = 200
    t = np.arange(4 * sr) / sr
    sig = 30.0 * np.sin(2 * np.pi * 12.0 * t)
    assert abs(dsp.dominant_frequency(sig, sr) - 12.0) < 1.0


def test_apply_filters_does_not_modify_the_input():
    """The display filter must never touch the caller's data.

    BrainFlow's DataFilter routines filter in place. A row of the recording is
    already contiguous float64, so the old ``np.ascontiguousarray`` guard handed
    the *same* buffer to BrainFlow and the raw recording got overwritten -- the
    review window's "Display filter" checkbox silently filtered the array that
    was later exported to .fif.
    """
    sr = 200
    rng = np.random.default_rng(0)
    raw = rng.normal(0.0, 20.0, size=(4, 4 * sr))
    before = raw.copy()

    for row in range(raw.shape[0]):
        out = dsp.apply_filters(raw[row, :], sr, dsp.FilterSettings())
        assert not np.shares_memory(out, raw)

    assert np.array_equal(raw, before)


def test_compute_psd_does_not_modify_the_input():
    """compute_psd detrends in place internally -- same aliasing trap."""
    sr = 200
    t = np.arange(4 * sr) / sr
    sig = 30.0 * np.sin(2 * np.pi * 12.0 * t) + 5.0 * t  # tone + linear drift
    before = sig.copy()
    dsp.compute_psd(sig, sr)
    assert np.array_equal(sig, before)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("bandpass,notch,noise_hz", [
    (True, False, 60.0),
    (False, True, 50.0),
    (True, True, 50.0),
])
def test_chebyshev_filters_keep_the_signal_and_reject_noise(causal, bandpass, notch, noise_hz):
    """Both native Chebyshev band operations previously returned only NaNs."""
    sr = 200
    t = np.arange(10 * sr) / sr
    raw = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * noise_hz * t)
    before = raw.copy()
    settings = dsp.FilterSettings(detrend=False, filter_type="Chebyshev",
                                  bandpass_enabled=bandpass, bp_high=45.0,
                                  notch_enabled=notch)

    out = dsp.apply_filters(raw, sr, settings, causal=causal)

    assert np.isfinite(out).all()
    assert np.array_equal(raw, before)
    # Measure steady-state tone amplitudes, allowing either filter phase.
    steady = out[2 * sr:-2 * sr]
    times = t[2 * sr:-2 * sr]
    amplitude = lambda hz: 2 * abs(np.mean(steady * np.exp(-2j * np.pi * hz * times)))
    assert 0.7 < amplitude(10) < 1.1
    assert amplitude(noise_hz) < 0.1


@pytest.mark.parametrize("filter_type", ["Butterworth", "Chebyshev", "Bessel"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("bandpass,notch,noise_hz", [
    (True, False, 80.0),
    (False, True, 50.0),
])
def test_fractional_rate_filters_retain_passband_and_reject_noise(
        filter_type, causal, bandpass, notch, noise_hz):
    sr = 200.4
    t = np.arange(4008) / sr
    raw = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * noise_hz * t)
    before = raw.copy()
    settings = dsp.FilterSettings(detrend=False, filter_type=filter_type,
                                  bandpass_enabled=bandpass, bp_high=35.,
                                  notch_enabled=notch)

    out = dsp.apply_filters(raw, sr, settings, causal=causal)

    assert np.isfinite(out).all()
    np.testing.assert_array_equal(raw, before)
    # Use a long steady-state interval so phase and fractional cycle counts do
    # not hide a failure to filter (previously a float rate returned raw data).
    steady, times = out[1002:-1002], t[1002:-1002]
    amplitude = lambda hz: 2 * abs(np.mean(steady * np.exp(-2j * np.pi * hz * times)))
    assert 0.5 < amplitude(10) < 1.1
    assert amplitude(noise_hz) < 0.1


def test_integer_valued_float_rate_matches_integer_rate():
    t = np.arange(2000) / 200
    raw = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)
    settings = dsp.FilterSettings(bp_high=40)

    np.testing.assert_array_equal(dsp.apply_filters(raw, 200., settings),
                                  dsp.apply_filters(raw, 200, settings))
    float_freqs, float_psd = dsp.compute_psd(raw, 200.)
    int_freqs, int_psd = dsp.compute_psd(raw, 200)
    np.testing.assert_array_equal(float_freqs, int_freqs)
    np.testing.assert_array_equal(float_psd, int_psd)


@pytest.mark.parametrize("sr", [200, 200., 199.8])
def test_psd_retains_exact_rate_and_integrated_sine_power(sr):
    t = np.arange(4000) / sr
    raw = 3 * np.sin(2 * np.pi * 10 * t)
    before = raw.copy()

    freqs, psd = dsp.compute_psd(raw, sr)

    assert freqs[-1] == pytest.approx(sr / 2, abs=1e-12)
    assert freqs[1] == pytest.approx(sr / 256, abs=1e-12)
    assert dsp.integrate_band_power(freqs, psd, 0, sr / 2) == pytest.approx(4.5, rel=0.002)
    assert dsp.integrate_band_power(freqs, psd, 0, sr / 2) == pytest.approx(
        np.mean(raw**2), rel=0.002)
    assert freqs[np.argmax(psd)] == pytest.approx(10, abs=sr / 256)
    np.testing.assert_array_equal(raw, before)


@pytest.mark.parametrize("sr", [200, 199.8])
def test_fft_amplitude_remains_in_input_units(sr):
    n = 4000
    tone_hz = sr * 200 / n  # exact FFT bin, independent of sampling rate
    raw = 3 * np.sin(2 * np.pi * tone_hz * np.arange(n) / sr)

    freqs, magnitude = dsp.compute_fft(raw, sr)
    peak = np.argmax(magnitude)

    assert freqs[peak] == pytest.approx(tone_hz, abs=1e-12)
    assert magnitude[peak] == pytest.approx(3., rel=1e-6)


def test_fractional_fft_and_spectrogram_use_exact_frequency_and_time_axes():
    sr = 199.8
    raw = np.sin(2 * np.pi * 10 * np.arange(3996) / sr)

    fft_freqs, _magnitudes = dsp.compute_fft(raw, sr)
    spec_freqs, times, _spectrum = dsp.compute_spectrogram(raw, sr, nperseg=128)

    assert fft_freqs[-1] == pytest.approx(sr / 2, abs=1e-12)
    assert spec_freqs[-1] == pytest.approx(sr / 2, abs=1e-12)
    assert times[0] == pytest.approx(64 / sr, abs=1e-12)
    assert times[1] - times[0] == pytest.approx(64 / sr, abs=1e-12)


def test_fractional_band_powers_report_dominant_alpha_and_do_not_mutate_data():
    sr = 199.8
    t = np.arange(3996) / sr
    raw = np.stack([np.sin(2 * np.pi * 10 * t), 2 * np.sin(2 * np.pi * 10 * t)])
    before = raw.copy()

    names, powers = dsp.compute_band_powers(raw, sr)

    assert powers.sum() == pytest.approx(1)
    assert powers[names.index("Alpha")] > .98
    np.testing.assert_array_equal(raw, before)


@pytest.mark.parametrize("sr", [200, 200., 199.8])
def test_band_power_average_weights_absolute_channel_power_before_normalizing(sr):
    t = np.arange(4000) / sr
    # Power scales with amplitude squared: 1:3 amplitude gives 1:9 band power.
    channels = np.stack([np.sin(2 * np.pi * 10 * t), 3 * np.sin(2 * np.pi * 20 * t)])

    names, powers = dsp.compute_band_powers(channels, sr)

    assert powers.sum() == pytest.approx(1)
    assert powers[names.index("Alpha")] == pytest.approx(.1, abs=.002)
    assert powers[names.index("Beta")] == pytest.approx(.9, abs=.002)


def test_band_power_delta_uses_the_documented_half_hertz_lower_bound(monkeypatch):
    # An analytically flat density makes every relative power its band width /
    # total width; the old BrainFlow default silently used delta from 2 Hz.
    freqs = np.arange(129) * 200 / 256
    monkeypatch.setattr(dsp, "compute_psd", lambda *args: (freqs, np.ones_like(freqs)))

    names, powers = dsp.compute_band_powers(np.zeros((1, 400)), 200)

    np.testing.assert_allclose(powers, [3.5 / 44.5, 4 / 44.5, 5 / 44.5,
                                       17 / 44.5, 15 / 44.5], atol=1e-12)
