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
)

FILTER_TYPES = {
    "Butterworth": FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
    "Chebyshev": FilterTypes.CHEBYSHEV_TYPE_1_ZERO_PHASE.value,
    "Bessel": FilterTypes.BESSEL_ZERO_PHASE.value,
}

# Causal (forward-only) variants. The live time-series uses these: a zero-phase
# filter runs the data forwards AND backwards, so a real transient (e.g. touching
# an electrode) rings *before* and *after* it and bends both window edges. A
# causal filter only responds after the event, like a true streaming filter --
# the trade-off is a small phase delay, which is fine for a live display.
CAUSAL_FILTER_TYPES = {
    "Butterworth": FilterTypes.BUTTERWORTH.value,
    "Chebyshev": FilterTypes.CHEBYSHEV_TYPE_1.value,
    "Bessel": FilterTypes.BESSEL.value,
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
_CHEBYSHEV_RIPPLE_DB = 0.5
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

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
    """Always return a fresh, contiguous float64 *copy*.

    The copy is not optional: BrainFlow's DataFilter routines write into the
    buffer they are given, and ``np.ascontiguousarray`` hands back the *same*
    buffer when the input is already contiguous float64 (which a row of the raw
    recording is). Filtering would then silently overwrite the caller's raw data
    - e.g. ticking "Display filter" in the review window used to corrupt the
    array that is later exported to .fif.
    """
    return np.array(arr, dtype=np.float64, order="C", copy=True)


def finite_segments(data: np.ndarray) -> List[slice]:
    """Contiguous measured runs; 2-D input requires every channel to be finite."""
    measured = np.isfinite(data)
    if measured.ndim == 2:
        measured = measured.all(axis=0)
    boundaries = np.diff(np.concatenate(([False], measured, [False])).astype(np.int8))
    return [slice(int(start), int(stop)) for start, stop in
            zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1))]


def _integer_sampling_rate(sampling_rate: float) -> Optional[int]:
    """Return an exact BrainFlow-compatible rate; never round a file's rate."""
    rate = float(sampling_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("Sampling rate must be a positive, finite number.")
    return int(rate) if rate.is_integer() and rate <= np.iinfo(np.int32).max else None


def _scipy_band_filter(data: np.ndarray, sampling_rate: float, low: float,
                       high: float, order: int, kind: str, causal: bool,
                       filter_type: str) -> np.ndarray:
    # BrainFlow requires integer rates; its 5.22.2 native Chebyshev band filters
    # can also return NaNs for valid settings. SOS supports both cases safely.
    kwargs = dict(btype=kind, fs=sampling_rate, output="sos")
    if filter_type == "Chebyshev":
        sos = sp_signal.cheby1(order, _CHEBYSHEV_RIPPLE_DB, (low, high), **kwargs)
    elif filter_type == "Bessel":
        sos = sp_signal.bessel(order, (low, high), **kwargs)
    else:
        sos = sp_signal.butter(order, (low, high), **kwargs)
    filter_func = sp_signal.sosfilt if causal else sp_signal.sosfiltfilt
    return filter_func(sos, data)


def apply_filters(channel: np.ndarray, sampling_rate: float,
                  settings: FilterSettings, causal: bool = False) -> np.ndarray:
    """Return a filtered copy of a single-channel signal.

    ``causal=True`` uses forward-only filters (for the live display); the default
    is zero-phase (for offline/analysis where phase fidelity matters).
    Fractional sampling rates use SciPy without rounding to BrainFlow's integer
    rate API. Missing samples remain missing; each finite run is filtered alone.
    """
    data = _as_float(channel)
    if data.size < _MIN_FILTER_SAMPLES:
        return data
    # A timed impedance check leaves unmeasured EEG, not zero-valued EEG. Never
    # fill those gaps or let either filter direction carry signal across them.
    if not np.all(np.isfinite(data)):
        for run in finite_segments(data):
            data[run] = apply_filters(data[run], sampling_rate, settings, causal=causal)
        return data
    type_map = CAUSAL_FILTER_TYPES if causal else FILTER_TYPES
    default = (FilterTypes.BUTTERWORTH.value if causal
              else FilterTypes.BUTTERWORTH_ZERO_PHASE.value)
    ftype = type_map.get(settings.filter_type, default)
    integer_rate = _integer_sampling_rate(sampling_rate)
    nyq = sampling_rate / 2.0
    try:
        if settings.detrend:
            DataFilter.detrend(data, DetrendOperations.LINEAR.value)
        if settings.bandpass_enabled:
            low = max(0.1, settings.bp_low)
            high = min(settings.bp_high, nyq - 1.0)
            if high > low:
                if integer_rate is None or settings.filter_type == "Chebyshev":
                    data = _scipy_band_filter(data, sampling_rate, low, high,
                                              settings.order, "bandpass", causal,
                                              settings.filter_type)
                else:
                    DataFilter.perform_bandpass(
                        data, integer_rate, low, high, settings.order, ftype, 0.0
                    )
        if settings.notch_enabled:
            f = settings.notch_freq
            half = settings.notch_width / 2.0
            if f + half < nyq:
                if integer_rate is None or settings.filter_type == "Chebyshev":
                    data = _scipy_band_filter(data, sampling_rate, f - half,
                                              f + half, settings.order, "bandstop", causal,
                                              settings.filter_type)
                else:
                    DataFilter.perform_bandstop(
                        data, integer_rate, f - half, f + half, settings.order, ftype, 0.0
                    )
    except Exception:
        # Bad parameter combination (e.g. order too high for sample count);
        # fall back to the unfiltered copy rather than crashing the UI.
        return _as_float(channel)
    return data


# Seconds of extra history fed to the filter as warm-up before the visible
# window. Zero-phase IIR filters ring at both edges of whatever they are given;
# giving them real past context (and reflecting the "now" edge) pushes those
# transients outside the part we actually show.
FILTER_WARMUP_SEC = 2.0


def apply_filters_windowed(channel: np.ndarray, sampling_rate: float,
                           settings: FilterSettings, visible: int) -> np.ndarray:
    """Filter ``channel`` but return only its last ``visible`` samples, clean.

    ``channel`` should include extra past samples before the visible window. The
    filter "warms up" over that history, so its left-edge transient lands in the
    discarded warm-up region instead of bending the start of the visible window
    (the artifact you get from filtering each short window in isolation). The
    right edge is "now" -- there is no future data, and the zero-phase filter's
    own internal padding already handles it well, so we leave it untouched.
    """
    data = _as_float(channel)
    visible = max(0, min(visible, data.size))
    if visible == 0:
        return data[:0]
    filtered = apply_filters(data, sampling_rate, settings, causal=True)
    return filtered[-visible:]


def interpolate_gaps(channel: np.ndarray, bad_mask: np.ndarray) -> np.ndarray:
    """Linearly interpolate the ``True`` positions of ``bad_mask`` from the
    surrounding good samples. Returns a copy; a no-op if nothing (or everything)
    is masked, or if the shapes disagree.

    Repairs BLE packet-loss seams *before* filtering: the decoder holds the last
    value flat on a dropped packet, which is a sharp DC step the band-pass rings
    on. Replacing those held samples with a smooth ramp between the good samples on
    either side removes the step, so the filter no longer rings. Gaps of any width
    (consecutive drops) are bridged; positions with no good sample to one side fall
    back to the nearest good value (``np.interp`` clamps at the edges).
    """
    data = np.array(channel, dtype=np.float64)  # copy (callers pass ring slices)
    bad = np.asarray(bad_mask, dtype=bool)
    if bad.shape != data.shape or not bad.any() or bad.all():
        return data
    if not np.isfinite(data).all():
        # BLE interpolation repairs held samples only inside measured runs.
        # It must not fill an unmeasured interval or bridge either side of one.
        for run in finite_segments(data):
            data[run] = interpolate_gaps(data[run], bad[run])
        return data
    good = ~bad
    x = np.arange(data.size)
    data[bad] = np.interp(x[bad], x[good], data[good])
    return data


def compute_psd(channel: np.ndarray, sampling_rate: float
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Power spectral density via Welch's method, in input units squared / Hz.

    PSD = how much signal power sits at each frequency. Welch averages the
    spectrum over overlapping windows for a smoother, more stable estimate.
    SciPy retains fractional rates and corrects for Hann window energy, which
    BrainFlow 5.22.2's Welch output omits. Missing runs are excluded: each finite
    section is analyzed separately and weighted by its complete Welch windows.
    Sections shorter than the common window do not contribute. Returns
    (freqs, amplitudes), or empty arrays when no usable measured section exists.
    """
    data = _as_float(channel)
    if data.size < 16:
        return np.array([]), np.array([])
    sections = [data[run] for run in finite_segments(data) if run.stop - run.start >= 16]
    if not sections:
        return np.array([]), np.array([])
    _integer_sampling_rate(sampling_rate)  # validate, without quantizing the rate
    # Approximately one-second power-of-two windows, retaining the original
    # window length choice while using the exact rate for frequency/PSD scaling.
    longest = max(section.size for section in sections)
    target = max(1.0, min(float(sampling_rate), longest))
    lower = 2 ** int(np.floor(np.log2(target)))
    nfft = lower if target - lower < 2 * lower - target else 2 * lower
    while nfft > longest:
        nfft //= 2
    if nfft < 8:
        return np.array([]), np.array([])
    spectra, weights = [], []
    for section in sections:
        if section.size < nfft:
            continue
        freqs, psd = sp_signal.welch(sp_signal.detrend(section), fs=sampling_rate,
                                    window="hann", nperseg=nfft, noverlap=nfft // 2,
                                    detrend=False, scaling="density")
        spectra.append(psd)
        weights.append(1 + (section.size - nfft) // (nfft - nfft // 2))
    return freqs, np.average(spectra, axis=0, weights=weights)


def compute_fft(channel: np.ndarray, sampling_rate: float
                ) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum of the longest contiguous measured run.

    No phase relationship is invented across missing EEG intervals. Returns
    (freqs, magnitude), or empty arrays when every measured run is too short.
    """
    data = _as_float(channel)
    if not np.isfinite(data).all():
        runs = finite_segments(data)
        if not runs:
            return np.array([]), np.array([])
        data = data[max(runs, key=lambda run: run.stop - run.start)]
    n = data.size
    if n < 16:
        return np.array([]), np.array([])
    data = data - np.mean(data)
    window = np.hanning(n)
    spectrum = np.fft.rfft(data * window)
    mag = np.abs(spectrum) * (2.0 / np.sum(window))
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    return freqs, mag


def compute_spectrogram(channel: np.ndarray, sampling_rate: float,
                        nperseg: Optional[int] = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (freqs, times, Sxx_dB), leaving gap-overlapping windows missing."""
    data = _as_float(channel)
    if data.size < 64:
        return np.array([]), np.array([]), np.zeros((0, 0))
    if nperseg is None:
        nperseg = min(256, data.size // 4 or 1)
    nperseg = max(16, nperseg)
    nperseg = min(nperseg, data.size)
    noverlap = nperseg // 2
    measured = np.isfinite(data)
    # SciPy handles every window separately. A temporary fill permits safe
    # computation; every window touching that fill is then explicitly masked.
    clean = np.where(measured, data, 0.0)
    freqs, times, sxx = sp_signal.spectrogram(
        clean, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap,
        window="hann", scaling="density",
    )
    sxx_db = 10.0 * np.log10(sxx + 1e-12)
    missing = np.concatenate(([0], np.cumsum(~measured)))
    starts = np.arange(times.size) * (nperseg - noverlap)
    invalid = missing[starts + nperseg] != missing[starts]
    sxx_db[:, invalid] = np.nan
    return freqs, times, sxx_db


def integrate_band_power(freqs: np.ndarray, psd: np.ndarray,
                         low: float, high: float) -> float:
    """Integrate a PSD up to exact band edges, clipped to available frequencies.

    Linear interpolation includes the portions between an edge and its nearest
    FFT bin, even if the entire band lies between two bins.
    """
    if freqs.size < 2:
        return 0.0
    low = max(low, float(freqs[0]))
    high = min(high, float(freqs[-1]))
    if high <= low:
        return 0.0
    interior = (freqs > low) & (freqs < high)
    band_freqs = np.concatenate(([low], freqs[interior], [high]))
    band_psd = np.interp(band_freqs, freqs, psd)
    return float(_trapezoid(band_psd, band_freqs))


def compute_band_powers(eeg: np.ndarray, sampling_rate: float
                        ) -> Tuple[List[str], np.ndarray]:
    """Relative power in the canonical EEG bands, averaged across channels.

    A compact summary of rhythm content -- how strong the slow vs. fast brain
    rhythms are. Average absolute channel powers first, then normalize by their
    sum. Welch detrends each channel; no hidden high-pass removes low delta.
    Returns (band_names, values), with values summing to one for nonzero power.
    """
    names = [b[0] for b in EEG_BANDS]
    if eeg.ndim != 2 or eeg.shape[1] < sampling_rate:
        return names, np.zeros(len(EEG_BANDS))
    data = _as_float(eeg)
    # BrainFlow's get_avg_band_powers uses a different delta band (2–4 Hz) and
    # rounds integration edges to bins. Share the report's exact boundaries.
    powers = []
    for channel in data:
        freqs, psd = compute_psd(channel, sampling_rate)
        powers.append([integrate_band_power(freqs, psd, lo, hi)
                       for _name, lo, hi in EEG_BANDS])
    mean = np.mean(powers, axis=0) if powers else np.zeros(len(EEG_BANDS))
    total = float(np.sum(mean))
    return names, mean / total if total > 0 else np.zeros(len(EEG_BANDS))


def is_railed(channel: np.ndarray) -> bool:
    """True if the signal pins near the Ganglion ADC full-scale (clipping)."""
    data = _as_float(channel)
    if data.size == 0:
        return False
    return bool(np.max(np.abs(data)) > _RAIL_FRACTION * GANGLION_FULLSCALE_UV)


def signal_quality(channel: np.ndarray, sampling_rate: float) -> dict:
    """Lightweight per-channel quality metrics for electrode characterization."""
    data = _as_float(channel)
    if data.size < _MIN_FILTER_SAMPLES:
        return {"rms": 0.0, "ptp": 0.0, "railed": False, "line_ratio": 0.0}
    rms = float(np.sqrt(np.mean(np.square(data - np.mean(data)))))
    ptp = float(np.ptp(data))
    railed = is_railed(data)
    # line_ratio: fraction of spectral magnitude sitting at mains hum (50/60 Hz).
    # High values flag electrical interference or poor electrode contact.
    freqs, mag = compute_fft(channel, sampling_rate)
    line_ratio = 0.0
    if freqs.size:
        total = float(np.sum(mag) + 1e-12)
        for f0 in (50.0, 60.0):
            band = (freqs > f0 - 2) & (freqs < f0 + 2)
            line_ratio = max(line_ratio, float(np.sum(mag[band]) / total))
    return {"rms": rms, "ptp": ptp, "railed": railed, "line_ratio": line_ratio}


def dominant_frequency(channel: np.ndarray, sampling_rate: float,
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


def channel_stats(channel: np.ndarray, sampling_rate: float) -> dict:
    """Rich per-channel statistics for the live stats panel."""
    data = _as_float(channel)
    base = signal_quality(data, sampling_rate)
    std = float(np.std(data)) if data.size else 0.0
    dom = dominant_frequency(data, sampling_rate) if data.size else 0.0
    base.update({"std": std, "dominant_hz": dom, "quality": quality_label(base)})
    return base
