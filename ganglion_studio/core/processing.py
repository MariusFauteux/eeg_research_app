"""Preprocessing/processing pipeline for the Processing Lab.

Steps (applied in this fixed order, each independently toggleable):

    re-reference (CAR) -> detrend -> filters -> wavelet denoise -> ASR
    -> ECG removal via R-peak-locked median AAS (NeuroKit2)

Light steps use BrainFlow/numpy; ASR uses asrpy on a temporary MNE Raw; AAS uses
NeuroKit2 for R-peak detection plus numpy epoching. Every step is guarded so a
failure degrades gracefully and reports a message instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from brainflow.data_filter import (
    DataFilter,
    DetrendOperations,
    ThresholdTypes,
    WaveletDenoisingTypes,
    WaveletTypes,
)

from .dsp import FilterSettings, apply_filters

WAVELETS = ["DB4", "DB6", "DB8", "SYM4", "SYM5", "COIF3", "BIOR3_9", "HAAR"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class FilterStepConfig:
    enabled: bool = True
    detrend: bool = False  # handled by its own step; kept False here
    bandpass_enabled: bool = True
    bp_low: float = 1.0
    bp_high: float = 45.0
    order: int = 4
    filter_type: str = "Butterworth"
    notch_enabled: bool = True
    notch_freq: int = 50

    def to_filter_settings(self) -> FilterSettings:
        return FilterSettings(
            detrend=False,
            bandpass_enabled=self.bandpass_enabled,
            bp_low=self.bp_low,
            bp_high=self.bp_high,
            order=self.order,
            filter_type=self.filter_type,
            notch_enabled=self.notch_enabled,
            notch_freq=self.notch_freq,
        )


@dataclass
class WaveletStepConfig:
    enabled: bool = False
    wavelet: str = "DB4"
    level: int = 3
    denoising: str = "SURESHRINK"  # or VISUSHRINK
    threshold: str = "SOFT"  # or HARD


@dataclass
class AsrStepConfig:
    enabled: bool = False
    cutoff: float = 20.0


@dataclass
class AasStepConfig:
    enabled: bool = False
    ref_channel: int = 0  # index into the loaded channels
    pre_ms: float = 250.0
    post_ms: float = 450.0
    aggregation: str = "median"  # or "mean"


@dataclass
class ProcessingConfig:
    reref_car: bool = False
    detrend: str = "none"  # none / constant / linear
    filters: FilterStepConfig = field(default_factory=FilterStepConfig)
    wavelet: WaveletStepConfig = field(default_factory=WaveletStepConfig)
    asr: AsrStepConfig = field(default_factory=AsrStepConfig)
    aas: AasStepConfig = field(default_factory=AasStepConfig)


# --------------------------------------------------------------------------- #
# Backend availability
# --------------------------------------------------------------------------- #
def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def available_methods() -> dict:
    return {
        "wavelet": True,  # BrainFlow, always present
        "asr": _have("meegkit"),
        "aas": _have("neurokit2"),
    }


_asr_compat_done = False


def _ensure_asr_compat() -> None:
    """Patch meegkit's fit_eeg_distribution for numpy>=2 compatibility.

    Under numpy 2.x the function returns mu/sig as 1-element arrays, which
    breaks scalar assignment in asr_calibrate. We coerce them back to floats.
    """
    global _asr_compat_done
    if _asr_compat_done:
        return
    try:
        import numpy as _np
        import meegkit.asr as _a
        import meegkit.utils.asr as _u

        _orig = _u.fit_eeg_distribution

        def _patched(*args, **kwargs):
            mu, sig, alpha, beta = _orig(*args, **kwargs)
            return float(_np.ravel(mu)[0]), float(_np.ravel(sig)[0]), alpha, beta

        _u.fit_eeg_distribution = _patched
        _a.fit_eeg_distribution = _patched
        _asr_compat_done = True
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# R-peak detection (NeuroKit2)
# --------------------------------------------------------------------------- #
def detect_rpeaks(ref_channel: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Return R-peak sample indices using NeuroKit2."""
    import neurokit2 as nk

    sig = np.ascontiguousarray(ref_channel, dtype=np.float64)
    if sig.size < sampling_rate:
        return np.array([], dtype=int)
    cleaned = nk.ecg_clean(sig, sampling_rate=sampling_rate)
    _signals, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
    peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    return peaks


# --------------------------------------------------------------------------- #
# AAS - R-peak-locked median template subtraction
# --------------------------------------------------------------------------- #
def apply_aas(eeg: np.ndarray, sampling_rate: int, cfg: AasStepConfig,
              ) -> Tuple[np.ndarray, str]:
    """Subtract a median (or mean) cardiac artifact template locked to R-peaks."""
    out = eeg.copy()
    ref_idx = cfg.ref_channel
    if ref_idx < 0 or ref_idx >= eeg.shape[0]:
        return out, "AAS skipped: invalid reference channel"
    try:
        peaks = detect_rpeaks(eeg[ref_idx], sampling_rate)
    except Exception as exc:
        return out, f"AAS skipped: R-peak detection failed ({exc})"
    if peaks.size < 3:
        return out, "AAS skipped: too few R-peaks detected"

    pre = int(round(cfg.pre_ms * sampling_rate / 1000.0))
    post = int(round(cfg.post_ms * sampling_rate / 1000.0))
    win = pre + post
    if win < 4:
        return out, "AAS skipped: window too small"
    n = eeg.shape[1]
    # Keep only peaks whose full window fits inside the recording.
    valid = peaks[(peaks - pre >= 0) & (peaks + post <= n)]
    if valid.size < 3:
        return out, "AAS skipped: not enough complete beats"

    agg = np.median if cfg.aggregation == "median" else np.mean
    for ch in range(eeg.shape[0]):
        epochs = np.stack([eeg[ch, p - pre:p + post] for p in valid], axis=0)
        template = agg(epochs, axis=0)
        for p in valid:
            out[ch, p - pre:p + post] -= template
    return out, f"AAS: removed cardiac template over {valid.size} beats (ref {ref_idx})"


# --------------------------------------------------------------------------- #
# ASR (asrpy + MNE)
# --------------------------------------------------------------------------- #
def apply_asr(eeg: np.ndarray, sampling_rate: int, ch_names: List[str],
              cfg: AsrStepConfig) -> Tuple[np.ndarray, str]:
    try:
        _ensure_asr_compat()
        from meegkit.asr import ASR
    except Exception:
        return eeg, "ASR skipped: meegkit not installed"
    if eeg.shape[0] < 2:
        return eeg, "ASR skipped: needs >= 2 channels"
    try:
        data = np.ascontiguousarray(eeg, dtype=np.float64)
        asr = ASR(sfreq=sampling_rate, cutoff=cfg.cutoff)
        asr.fit(data)
        cleaned = asr.transform(data)
        return np.ascontiguousarray(cleaned), f"ASR: cutoff {cfg.cutoff}"
    except Exception as exc:
        return eeg, f"ASR skipped: {exc}"


# --------------------------------------------------------------------------- #
# Wavelet (BrainFlow)
# --------------------------------------------------------------------------- #
def apply_wavelet(eeg: np.ndarray, cfg: WaveletStepConfig) -> Tuple[np.ndarray, str]:
    wt = getattr(WaveletTypes, cfg.wavelet, WaveletTypes.DB4)
    den = getattr(WaveletDenoisingTypes, cfg.denoising, WaveletDenoisingTypes.SURESHRINK)
    thr = getattr(ThresholdTypes, cfg.threshold, ThresholdTypes.SOFT)
    out = eeg.copy()
    try:
        for ch in range(out.shape[0]):
            DataFilter.perform_wavelet_denoising(
                out[ch], wt, cfg.level, wavelet_denoising=den, threshold=thr
            )
        return out, f"Wavelet: {cfg.wavelet} L{cfg.level} {cfg.denoising}/{cfg.threshold}"
    except Exception as exc:
        return eeg, f"Wavelet skipped: {exc}"


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def apply_pipeline(eeg: np.ndarray, sampling_rate: int, ch_names: List[str],
                   config: ProcessingConfig) -> Tuple[np.ndarray, List[str]]:
    """Apply the enabled steps in order. Returns (processed_eeg, messages)."""
    messages: List[str] = []
    if eeg is None or eeg.ndim != 2 or eeg.shape[1] == 0:
        return eeg, ["No data to process."]
    data = np.ascontiguousarray(eeg, dtype=np.float64).copy()

    if config.reref_car and data.shape[0] > 1:
        data = data - np.mean(data, axis=0, keepdims=True)
        messages.append("Re-reference: common average (CAR)")

    if config.detrend in ("constant", "linear"):
        op = DetrendOperations.CONSTANT.value if config.detrend == "constant" else DetrendOperations.LINEAR.value
        for ch in range(data.shape[0]):
            DataFilter.detrend(data[ch], op)
        messages.append(f"Detrend: {config.detrend}")

    if config.filters.enabled:
        fs = config.filters.to_filter_settings()
        for ch in range(data.shape[0]):
            data[ch] = apply_filters(data[ch], sampling_rate, fs)
        messages.append(
            f"Filter: {'BP %.1f-%.1f' % (fs.bp_low, fs.bp_high) if fs.bandpass_enabled else 'no BP'}"
            f"{' + notch %d' % fs.notch_freq if fs.notch_enabled else ''} ({fs.filter_type} o{fs.order})"
        )

    if config.wavelet.enabled:
        data, msg = apply_wavelet(data, config.wavelet)
        messages.append(msg)

    if config.asr.enabled:
        data, msg = apply_asr(data, sampling_rate, ch_names, config.asr)
        messages.append(msg)

    if config.aas.enabled:
        data, msg = apply_aas(data, sampling_rate, config.aas)
        messages.append(msg)

    return data, messages
