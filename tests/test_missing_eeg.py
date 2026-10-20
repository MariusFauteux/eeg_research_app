"""Timed impedance gaps remain unmeasured through filtering and spectra."""

import numpy as np
import pytest

from ganglion_studio.core import dsp, processing


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("filter_type", ["Butterworth", "Chebyshev", "Bessel"])
def test_filters_preserve_missing_samples_and_do_not_mix_measured_sections(causal, filter_type):
    rng = np.random.default_rng(8)
    first, last = rng.normal(size=800), rng.normal(size=600) + 100
    raw = np.concatenate([first, np.full(400, np.nan), last])
    before = raw.copy()
    settings = dsp.FilterSettings(filter_type=filter_type)

    out = dsp.apply_filters(raw, 200, settings, causal=causal)

    np.testing.assert_array_equal(out[:800], dsp.apply_filters(first, 200, settings, causal=causal))
    np.testing.assert_array_equal(out[-600:], dsp.apply_filters(last, 200, settings, causal=causal))
    assert np.isnan(out[800:1200]).all()
    np.testing.assert_array_equal(raw, before)


def test_ble_interpolation_never_fills_or_crosses_unmeasured_intervals():
    raw = np.array([0., 99., np.nan, np.nan, 7., 8., 99., 10.])
    bad = np.array([False, True, True, True, True, False, True, False])

    out = dsp.interpolate_gaps(raw, bad)

    np.testing.assert_array_equal(out, [0., 0., np.nan, np.nan, 8., 8., 9., 10.])


@pytest.mark.parametrize("sr", [200, 199.8])
def test_psd_weights_complete_measured_windows_and_not_gap_duration(sr):
    first = np.sin(2 * np.pi * 10 * np.arange(384) / sr)
    last = 3 * np.sin(2 * np.pi * 10 * np.arange(768) / sr)
    raw = np.concatenate([first, np.full(2000, np.nan), last])

    freqs, psd = dsp.compute_psd(raw, sr)
    f1, p1 = dsp.compute_psd(first, sr)
    f2, p2 = dsp.compute_psd(last, sr)

    # 256 samples/window and 128 hop: two windows before, five after the gap.
    np.testing.assert_array_equal(freqs, f1)
    np.testing.assert_array_equal(freqs, f2)
    np.testing.assert_allclose(psd, (2 * p1 + 5 * p2) / 7, rtol=1e-13)
    assert dsp.integrate_band_power(freqs, psd, 8, 13) == pytest.approx((2 * .5 + 5 * 4.5) / 7, rel=.01)
    _, longer_gap = dsp.compute_psd(np.concatenate([first, np.full(8000, np.nan), last]), sr)
    np.testing.assert_array_equal(psd, longer_gap)


def test_fft_uses_one_continuous_section_instead_of_joining_across_a_gap():
    sr = 200
    first = np.sin(2 * np.pi * 10 * np.arange(400) / sr)
    last = 3 * np.sin(2 * np.pi * 20 * np.arange(800) / sr)
    raw = np.concatenate([first, np.full(200, np.nan), last])

    freqs, magnitude = dsp.compute_fft(raw, sr)
    expected_freqs, expected_magnitude = dsp.compute_fft(last, sr)

    np.testing.assert_array_equal(freqs, expected_freqs)
    np.testing.assert_array_equal(magnitude, expected_magnitude)
    assert freqs[np.argmax(magnitude)] == 20


def test_spectrogram_masks_every_window_touching_a_gap_without_changing_its_clock():
    raw = np.sin(2 * np.pi * 10 * np.arange(512) / 200)
    missing = raw.copy()
    missing[96:160] = np.nan

    freqs, times, power = dsp.compute_spectrogram(missing, 200, nperseg=64)
    expected_freqs, expected_times, expected_power = dsp.compute_spectrogram(raw, 200, nperseg=64)

    np.testing.assert_array_equal(freqs, expected_freqs)
    np.testing.assert_array_equal(times, expected_times)
    starts = np.arange(times.size) * 32
    affected = (starts < 160) & (starts + 64 > 96)
    assert np.isnan(power[:, affected]).all()
    np.testing.assert_array_equal(power[:, ~affected], expected_power[:, ~affected])


def test_every_sample_missing_yields_no_spectrum_and_filters_keep_it_missing():
    raw = np.full(1000, np.nan)
    assert np.isnan(dsp.apply_filters(raw, 200, dsp.FilterSettings())).all()
    assert all(values.size == 0 for values in dsp.compute_psd(raw, 200))
    assert all(values.size == 0 for values in dsp.compute_fft(raw, 200))
    assert np.isnan(dsp.compute_spectrogram(raw, 200)[2]).all()


def test_pipeline_never_sends_missing_samples_to_brainflow(monkeypatch):
    calls = []
    for name in ("detrend", "perform_bandpass", "perform_bandstop", "perform_wavelet_denoising"):
        original = getattr(dsp.DataFilter, name)

        def checked(data, *args, operation=original, operation_name=name, **kwargs):
            assert np.isfinite(data).all()
            calls.append(operation_name)
            return operation(data, *args, **kwargs)

        monkeypatch.setattr(dsp.DataFilter, name, checked)
    t = np.arange(800) / 200
    first = np.stack([10 + np.sin(2 * np.pi * 10 * t), 20 + np.sin(2 * np.pi * 12 * t)])
    last = first + 100
    raw = np.concatenate([first, np.full((2, 400), np.nan), last], axis=1)
    before = raw.copy()
    config = processing.ProcessingConfig(repair_gaps=True, loss_samples=[800, 1199],
                                         loss_lengths=[400, 2], detrend="linear", reref_car=True)
    config.wavelet.enabled = True

    out, messages = processing.apply_pipeline(raw, 200, ["A", "B"], config)

    assert set(calls) == {"detrend", "perform_bandpass", "perform_bandstop", "perform_wavelet_denoising"}
    assert np.isfinite(out[:, :800]).all()
    assert np.isfinite(out[:, 1200:]).all()
    assert np.isnan(out[:, 800:1200]).all()
    assert "2 measured sections independently" in messages[0]
    np.testing.assert_array_equal(raw, before)


def test_wavelet_direct_call_rejects_missing_input_before_native_code(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("nonfinite input must never reach native wavelet code")

    monkeypatch.setattr(dsp.DataFilter, "perform_wavelet_denoising", unexpected)
    raw = np.full((2, 400), np.nan)

    out, message = processing.apply_wavelet(raw, processing.WaveletStepConfig(enabled=True))

    np.testing.assert_array_equal(out, raw)
    assert "continuous measured section" in message
