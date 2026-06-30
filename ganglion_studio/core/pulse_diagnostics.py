"""Characterize a periodic "pulse" artifact in an EEG channel.

Shared, side-effect-free analysis used by BOTH the live Diagnostics tab
(``ui/plots/pulse_diagnostics_widget.py``) and the offline
``tools/diagnose_artifact.py``, so the two never drift apart.

The question this answers: a recurring once-per-second "pulse" in the trace --
is it a *systematic* artifact (locked to the clock / to BLE packet-loss seams) or
a biological / random signal? A real heartbeat wanders (HRV) and is rarely
exactly 1.000 s apart; a pipeline glitch is phase-locked with sub-sample jitter
and often lands exactly on a dropped-packet seam.

Everything here is plain NumPy/SciPy on a 1-D float array, so it is trivially
unit-testable with synthetic spike trains (see ``tests/test_pulse_diagnostics.py``).
Thresholds mirror the values already validated in the offline tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

# Verdict keys returned by :func:`classify`. The UI maps these to colours/wording.
VERDICT_NO_PULSE = "no_pulse"          # nothing repeating found
VERDICT_DIGITAL = "digital"            # pulses sit on BLE packet-loss seams
VERDICT_FRAME_LOCKED = "frame_locked"  # phase-locked but NOT on loss seams
VERDICT_BIOLOGICAL = "biological"      # spacing wanders -> looks like a real signal
VERDICT_INCONCLUSIVE = "inconclusive"


def detect_pulses(signal: np.ndarray, sr: int, *, sigma: float = 8.0,
                  min_gap_s: float = 0.8) -> np.ndarray:
    """Indices of large, isolated deflections ("pulses") in ``signal``.

    Robust against the EEG's own wander: the threshold is built from the
    median-absolute-deviation (MAD), not the mean/std, so a handful of big pulses
    can't inflate the very threshold meant to catch them. ``1.4826 * MAD`` is the
    MAD-to-std conversion for a normal distribution, so ``sigma`` reads as
    "sigmas". ``min_gap_s`` enforces a minimum spacing (a ~1 Hz pulse can't have
    two peaks 10 ms apart).
    """
    x = np.asarray(signal, dtype=float)
    if x.size < 4:
        return np.empty(0, dtype=int)
    x = x - np.median(x)
    mad = np.median(np.abs(x - np.median(x))) + 1e-9
    thresh = sigma * 1.4826 * mad
    distance = max(1, int(min_gap_s * sr))
    peaks, _ = find_peaks(np.abs(x), height=thresh, distance=distance)
    return peaks.astype(int)


def spacing_stats(peaks: np.ndarray, sr: int) -> Dict[str, float]:
    """Inter-pulse spacing summary.

    ``jitter_ms`` is the std of the spacing: a clock-locked artifact has a few ms;
    a heartbeat (HRV) has tens of ms. ``rate_hz`` is 1 / median-spacing.
    """
    peaks = np.asarray(peaks)
    if peaks.size < 3 or sr <= 0:
        return {"rate_hz": 0.0, "median_s": 0.0, "mean_s": 0.0, "jitter_ms": 0.0,
                "n": int(peaks.size)}
    spacing = np.diff(peaks) / sr
    median_s = float(np.median(spacing))
    return {
        "rate_hz": (1.0 / median_s) if median_s > 0 else 0.0,
        "median_s": median_s,
        "mean_s": float(spacing.mean()),
        "jitter_ms": float(spacing.std() * 1000.0),
        "n": int(peaks.size),
    }


def phase_concentration(peaks: np.ndarray, sr: int) -> Dict[str, float]:
    """How tightly the pulses lock to a constant phase within the 1 s frame.

    ``r``=1.0 -> every pulse at the same sample-of-second (frame-locked, the
    fingerprint of a digital/clock event); ``r``~0 -> phases spread out.
    """
    peaks = np.asarray(peaks)
    if peaks.size < 3 or sr <= 0:
        return {"r": 0.0, "median_phase_sample": 0}
    phase = peaks % sr
    ang = 2 * np.pi * phase / sr
    r = float(np.abs(np.mean(np.exp(1j * ang))))
    return {"r": r, "median_phase_sample": int(np.median(phase))}


def loss_coincidence(peaks: np.ndarray, anomaly_flags: np.ndarray, *,
                     tol: int = 3) -> Dict[str, float]:
    """How often pulses land within ``tol`` samples of a flagged anomaly.

    ``anomaly_flags`` is a per-sample array that is non-zero at each event of
    interest -- live, the BLE packet-loss flags from ``recent_loss``; offline, a
    1 placed at each timestamp gap/dup. ``fraction`` observed far above ``chance``
    means the pulse IS the seam (a stream/decode artifact, fixable in code) rather
    than a coupled analog event.
    """
    peaks = np.asarray(peaks)
    flags = np.asarray(anomaly_flags)
    n = flags.size
    anomalies = np.flatnonzero(flags != 0) if n else np.empty(0, dtype=int)
    if peaks.size == 0 or anomalies.size == 0:
        return {"fraction": 0.0, "chance": 0.0, "n_anomalies": int(anomalies.size)}
    near = sum(1 for p in peaks if np.min(np.abs(anomalies - p)) <= tol)
    fraction = near / peaks.size
    # Probability a randomly placed pulse lands within +/-tol of any anomaly.
    chance = min(1.0, anomalies.size * (2 * tol + 1) / n)
    return {"fraction": float(fraction), "chance": float(chance),
            "n_anomalies": int(anomalies.size)}


def pulse_triggered_average(signal: np.ndarray, peaks: np.ndarray, sr: int, *,
                            half_s: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Average waveform in a +/- ``half_s`` window around each pulse.

    The shape is itself diagnostic: a single-sample step/spike points to a
    decode/packet glitch; a smooth bump points to a coupled analog event. Each
    segment is de-medianed before averaging so they overlay regardless of DC.
    Returns ``(t_seconds, mean_waveform)``; empty arrays if no complete window.
    """
    x = np.asarray(signal, dtype=float)
    peaks = np.asarray(peaks)
    half = max(1, int(half_s * sr))
    segments = []
    for p in peaks:
        lo, hi = int(p) - half, int(p) + half + 1
        if lo >= 0 and hi <= x.size:
            seg = x[lo:hi]
            segments.append(seg - np.median(seg))
    if not segments:
        return np.empty(0), np.empty(0)
    avg = np.mean(np.vstack(segments), axis=0)
    t = (np.arange(avg.size) - half) / sr
    return t, avg


def classify(spacing: Dict[str, float], phase: Dict[str, float],
             coincidence: Dict[str, float]) -> Tuple[str, str]:
    """Map the numeric tells to a verdict key + a plain-language sentence."""
    n = int(spacing.get("n", 0))
    if n < 3:
        return VERDICT_NO_PULSE, "No repeating pulse detected in this window."
    rate = spacing["rate_hz"]
    jitter = spacing["jitter_ms"]
    r = phase["r"]
    frac = coincidence["fraction"]
    chance = coincidence["chance"]

    # Coincident with packet loss far above chance -> the held-flat seam itself.
    if coincidence["n_anomalies"] > 0 and frac >= 0.6 and frac > 2 * chance:
        return VERDICT_DIGITAL, (
            f"Pulses land on BLE packet-loss seams ({frac * 100:.0f}% vs "
            f"~{chance * 100:.0f}% by chance) -> a dropped-packet / delta-decode "
            "glitch. This is fixable in the stream (gap repair / wrap handling), "
            "not at the electrodes.")

    # Tightly phase- and clock-locked, but no loss to blame -> a coupled event.
    if jitter < 15.0 and r >= 0.8:
        return VERDICT_FRAME_LOCKED, (
            f"Phase-locked at {rate:.2f} Hz (jitter {jitter:.0f} ms, r={r:.2f}) "
            "but NOT on packet-loss seams -> a once-per-second coupled/board event "
            "(BLE/LED/charger), not packet loss. Try shielding, change the power "
            "setup, or A/B another connection backend.")

    # Spacing wanders -> consistent with a genuine (e.g. biological) rhythm.
    if jitter >= 40.0 or r < 0.5:
        return VERDICT_BIOLOGICAL, (
            f"Spacing wanders (jitter {jitter:.0f} ms, r={r:.2f}); looks like a "
            "real signal (e.g. a heartbeat with HRV), not a systematic 1 Hz "
            "artifact.")

    return VERDICT_INCONCLUSIVE, (
        f"Detected {n} pulses at {rate:.2f} Hz (jitter {jitter:.0f} ms, r={r:.2f}); "
        "the evidence is mixed -- mark an event and watch how the pulse responds.")


@dataclass
class DiagnosticResult:
    """Everything the UI / CLI needs for one channel, in one object."""

    n_pulses: int
    rate_hz: float
    jitter_ms: float
    phase_r: float
    loss_fraction: float
    loss_chance: float
    n_anomalies: int
    verdict: str
    message: str
    peaks: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))


def diagnose(signal: np.ndarray, sr: int,
             anomaly_flags: Optional[np.ndarray] = None, *,
             sigma: float = 8.0) -> DiagnosticResult:
    """Run the full pulse characterization on one channel.

    ``anomaly_flags`` (optional) is a per-sample non-zero-at-event array used for
    the loss-coincidence test -- the live BLE-loss flags, or offline timestamp
    anomalies. When omitted, the coincidence test is simply skipped.
    """
    peaks = detect_pulses(signal, sr, sigma=sigma)
    spacing = spacing_stats(peaks, sr)
    phase = phase_concentration(peaks, sr)
    if anomaly_flags is None:
        anomaly_flags = np.zeros(np.asarray(signal).size)
    coincidence = loss_coincidence(peaks, anomaly_flags)
    verdict, message = classify(spacing, phase, coincidence)
    return DiagnosticResult(
        n_pulses=spacing["n"],
        rate_hz=spacing["rate_hz"],
        jitter_ms=spacing["jitter_ms"],
        phase_r=phase["r"],
        loss_fraction=coincidence["fraction"],
        loss_chance=coincidence["chance"],
        n_anomalies=coincidence["n_anomalies"],
        verdict=verdict,
        message=message,
        peaks=peaks,
    )
