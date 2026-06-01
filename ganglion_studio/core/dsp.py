"""Signal-processing helpers built on BrainFlow's DataFilter + scipy.

All BrainFlow ``DataFilter`` operations mutate their input in place, so every
public helper here works on a contiguous float64 *copy* and returns a new array
- callers may pass slices of the raw ring buffer safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

from brainflow.data_filter import (
    DataFilter,
    DetrendOperations,
    FilterTypes,
    WindowOperations,
)

FILTER_TYPES = {
    "Butterworth": FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
    "Chebyshev": FilterTypes.CHEBYSHEV_TYPE_1_ZERO_PHASE.value,
    "Bessel": FilterTypes.BESSEL_ZERO_PHASE.value,
}

# Classic EEG frequency bands (Hz).
EEG_BANDS = [
    ("Delta", 0.5, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta", 13.0, 30.0),
    ("Gamma", 30.0, 45.0),
]

_MIN_FILTER_SAMPLES = 32

# OpenBCI Ganglion full-scale input range (MCP3912 ADC, Vref 1.2 V, gain 51):
#   scale      = 1.2 / ((2**23 - 1) * 1.5 * 51) * 1e6  ~= 1.87e-3 uV/count
#   full-scale = scale * (2**23 - 1) = 1.2e6 / (1.5 * 51)  ~= 15686 uV
# (Note: +/- 187500 uV is the *Cyton's* range -- 4.5 V / gain 24 -- not the
# Ganglion. BrainFlow already returns Ganglion data in uV.)
GANGLION_FULLSCALE_UV = 15686.0
# Fraction of full-scale above which a channel is treated as railed/clipped.
_RAIL_FRACTION = 0.95


@dataclass
class FilterSettings:
    """User-controlled display filter applied to every plot (not to recording)."""

    detrend: bool = True
    bandpass_enabled: bool = True
    bp_low: float = 1.0
    bp_high: float = 50.0
    order: int = 4
    filter_type: str = "Butterworth"
    notch_enabled: bool = True
    notch_freq: int = 50  # 50 or 60 Hz
    notch_width: float = 4.0

    def clone(self) -> "FilterSettings":
        return FilterSettings(**self.__dict__)


def _as_float(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float64)


def apply_filters(channel: np.ndarray, sampling_rate: int,
                  settings: FilterSettings) -> np.ndarray:
    """Return a filtered copy of a single-channel signal."""
    data = _as_float(channel)
    if data.size < _MIN_FILTER_SAMPLES:
        return data
    ftype = FILTER_TYPES.get(settings.filter_type, FilterTypes.BUTTERWORTH_ZERO_PHASE.value)
    nyq = sampling_rate / 2.0
    try:
        if settings.detrend:
            DataFilter.detrend(data, DetrendOperations.LINEAR.value)
        if settings.bandpass_enabled:
            low = max(0.1, settings.bp_low)
            high = min(settings.bp_high, nyq - 1.0)
            if high > low:
                DataFilter.perform_bandpass(
                    data, sampling_rate, low, high, settings.order, ftype, 0.0
                )
        if settings.notch_enabled:
            f = settings.notch_freq
            half = settings.notch_width / 2.0
            if f + half < nyq:
                DataFilter.perform_bandstop(
                    data, sampling_rate, f - half, f + half, settings.order, ftype, 0.0
                )
    except Exception:
        # Bad parameter combination (e.g. order too high for sample count);
        # fall back to the unfiltered copy rather than crashing the UI.
        return _as_float(channel)
    return data


def compute_psd(channel: np.ndarray, sampling_rate: int
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD via BrainFlow. Returns (freqs, amplitudes)."""
    data = _as_float(channel)
    if data.size < 16:
        return np.array([]), np.array([])
    nfft = DataFilter.get_nearest_power_of_two(min(sampling_rate, data.size))
    while nfft > data.size:
        nfft //= 2
    if nfft < 8:
        return np.array([]), np.array([])
    DataFilter.detrend(data, DetrendOperations.LINEAR.value)
    amps, freqs = DataFilter.get_psd_welch(
        data, nfft, nfft // 2, sampling_rate, WindowOperations.HANNING.value
    )
    return freqs, amps


def compute_fft(channel: np.ndarray, sampling_rate: int
                ) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum. Returns (freqs, magnitude)."""
    data = _as_float(channel)
    n = data.size
    if n < 16:
        return np.array([]), np.array([])
    data = data - np.mean(data)
    window = np.hanning(n)
    spectrum = np.fft.rfft(data * window)
    mag = np.abs(spectrum) * (2.0 / np.sum(window))
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    return freqs, mag


def compute_spectrogram(channel: np.ndarray, sampling_rate: int,
                        nperseg: Optional[int] = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (freqs, times, Sxx_dB) for a rolling spectrogram."""
    data = _as_float(channel)
    if data.size < 64:
        return np.array([]), np.array([]), np.zeros((0, 0))
    if nperseg is None:
        nperseg = min(256, data.size // 4 or 1)
    nperseg = max(16, nperseg)
    noverlap = nperseg // 2
    freqs, times, sxx = sp_signal.spectrogram(
        data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap,
        window="hann", scaling="density",
    )
    sxx_db = 10.0 * np.log10(sxx + 1e-12)
    return freqs, times, sxx_db


def compute_band_powers(eeg: np.ndarray, sampling_rate: int
                        ) -> Tuple[List[str], np.ndarray]:
    """Average band powers across channels. Returns (band_names, values)."""
    names = [b[0] for b in EEG_BANDS]
    if eeg.ndim != 2 or eeg.shape[1] < sampling_rate:
        return names, np.zeros(len(EEG_BANDS))
    data = _as_float(eeg)
    channels = list(range(data.shape[0]))
    try:
        avg, _std = DataFilter.get_avg_band_powers(data, channels, sampling_rate, True)
        return names, np.asarray(avg)
    except Exception:
        return names, np.zeros(len(EEG_BANDS))


def is_railed(channel: np.ndarray) -> bool:
    """True if the signal pins near the Ganglion ADC full-scale (clipping)."""
    data = _as_float(channel)
    if data.size == 0:
        return False
    return bool(np.max(np.abs(data)) > _RAIL_FRACTION * GANGLION_FULLSCALE_UV)


def signal_quality(channel: np.ndarray, sampling_rate: int) -> dict:
    """Lightweight per-channel quality metrics for electrode characterization."""
    data = _as_float(channel)
    if data.size < _MIN_FILTER_SAMPLES:
        return {"rms": 0.0, "ptp": 0.0, "railed": False, "line_ratio": 0.0}
    rms = float(np.sqrt(np.mean(np.square(data - np.mean(data)))))
    ptp = float(np.ptp(data))
    railed = is_railed(data)
    freqs, mag = compute_fft(channel, sampling_rate)
    line_ratio = 0.0
    if freqs.size:
        total = float(np.sum(mag) + 1e-12)
        for f0 in (50.0, 60.0):
            band = (freqs > f0 - 2) & (freqs < f0 + 2)
            line_ratio = max(line_ratio, float(np.sum(mag[band]) / total))
    return {"rms": rms, "ptp": ptp, "railed": railed, "line_ratio": line_ratio}


def dominant_frequency(channel: np.ndarray, sampling_rate: int,
                       fmin: float = 1.0, fmax: float = 70.0) -> float:
    """Frequency (Hz) of the strongest spectral component in [fmin, fmax]."""
    freqs, mag = compute_fft(channel, sampling_rate)
    if freqs.size == 0:
        return 0.0
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return 0.0
    sub_freqs = freqs[band]
    sub_mag = mag[band]
    return float(sub_freqs[int(np.argmax(sub_mag))])


def quality_label(stats: dict) -> str:
    """Map raw metrics to a contact-quality label: good / ok / bad."""
    if stats.get("railed"):
        return "bad"
    ptp = stats.get("ptp", 0.0)
    line = stats.get("line_ratio", 0.0)
    if ptp > 1000.0 or line > 0.5:
        return "bad"
    if ptp > 200.0 or line > 0.25:
        return "ok"
    return "good"


def channel_stats(channel: np.ndarray, sampling_rate: int) -> dict:
    """Rich per-channel statistics for the live stats panel."""
    data = _as_float(channel)
    base = signal_quality(data, sampling_rate)
    std = float(np.std(data)) if data.size else 0.0
    dom = dominant_frequency(data, sampling_rate) if data.size else 0.0
    base.update({"std": std, "dominant_hz": dom, "quality": quality_label(base)})
    return base
