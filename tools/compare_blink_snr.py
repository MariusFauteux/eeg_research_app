"""Compare the blink SNR between two .fif recordings.

Standalone tool. Each recording is expected to contain a few "Movement"
markers, each announcing a short block of eye-blinks. The script applies light
preprocessing and reports two complementary signal-to-noise measures for each
file, then prints a side-by-side comparison.

Preprocessing (zero-phase, applied per active channel)
------------------------------------------------------
  1. linear detrend            -- removes DC offset and slow drift
  2. band-pass 1-60 Hz         -- 4th-order Butterworth, filtfilt
  3. notch 50 Hz               -- IIR notch (mains), filtfilt

Two SNR definitions are reported
--------------------------------
  Amplitude SNR  = median blink-peak amplitude / RMS of the quiet baseline.
                   "How tall is a blink compared to the resting noise floor."
  Spectral SNR   = blink-band power during blinks / blink-band power during the
                   quiet baseline.  "How much more low-frequency power a blink
                   produces over the same band at rest."

Both are reported as a linear ratio and in dB (10*log10 for power, 20*log10 for
amplitude). Higher = cleaner blinks.

Usage
-----
Edit the two file names in the CHOOSE FILES block below to pick which
recordings in ``mne_data_raw/`` to compare, then run:

    python tools/compare_blink_snr.py

Set PLOT_PATH to a filename to also save an overlay plot (or None to skip).

Needs: mne, numpy, scipy (matplotlib only when PLOT_PATH is set).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.signal import butter, detrend, filtfilt, find_peaks, iirnotch, welch

# --- CHOOSE FILES (edit these) ----------------------------------------------
# Folder holding the .fif recordings, relative to this script.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "mne_data_raw")

# The two recordings to compare (file names inside DATA_DIR).
FILE_A = "mc15_blinks_day1_raw.fif"
FILE_B = "mc15_blinks_day2_raw.fif"
# Commercial vs MC15 (for the film) vs semiwet (for the housing)

# Set to a path to save an overlay plot, or None to skip plotting.
PLOT_PATH = "./output/"
# ----------------------------------------------------------------------------

# --- analysis settings (tune here) ------------------------------------------
BAND = (1.0, 60.0)          # band-pass passband (Hz)
NOTCH_HZ = 50.0             # mains notch (Hz)
NOTCH_Q = 30.0              # notch quality factor
BLINK_BAND = (1.0, 5.0)     # where eye-blink energy lives (Hz)
PEAK_K = 5.0                # blink threshold = PEAK_K * robust std
MIN_BLINK_GAP_S = 0.25      # minimum spacing between blink peaks (s)
GUARD_S = 0.4               # samples within +/-GUARD_S of a peak are "blink"
EPOCH_PRE_S = 0.2           # blink epoch window before the peak (s)
EPOCH_POST_S = 0.4          # blink epoch window after the peak (s)
ACTIVE_STD_UV = 0.5         # a channel with std below this is treated as OFF


@dataclass
class Result:
    """Everything we measured for one file/channel."""

    path: str
    channel: str
    n_blinks: int
    blink_amp_uv: float       # median blink peak amplitude (uV)
    noise_rms_uv: float       # baseline RMS (uV)
    amp_snr: float            # linear amplitude SNR
    amp_snr_db: float
    spec_snr: float           # linear spectral SNR (power ratio)
    spec_snr_db: float


def preprocess(x: np.ndarray, fs: float) -> np.ndarray:
    """Detrend, band-pass 1-60 Hz, then notch 50 Hz (all zero-phase)."""
    x = detrend(x, type="linear")
    nyq = fs / 2.0
    b, a = butter(4, [BAND[0] / nyq, BAND[1] / nyq], btype="band")
    x = filtfilt(b, a, x)
    b, a = iirnotch(NOTCH_HZ, NOTCH_Q, fs)
    x = filtfilt(b, a, x)
    return x


def active_channels(data_uv: np.ndarray, names: List[str]) -> List[int]:
    """Indices of channels that actually carry signal (not switched OFF)."""
    return [i for i in range(data_uv.shape[0])
            if data_uv[i].std() > ACTIVE_STD_UV]


def detect_blinks(y: np.ndarray, fs: float) -> np.ndarray:
    """Find blink peaks on the preprocessed signal.

    Blinks are large *positive-going* deflections (standard EOG/blink polarity
    on this montage), so we look for positive peaks only -- this avoids
    mistaking the negative undershoot that follows each blink for a second
    blink. A robust (MAD-based) threshold is used so the few large blinks don't
    inflate the noise floor the way plain std would. Returns peak sample
    indices.
    """
    robust_std = 1.4826 * np.median(np.abs(y - np.median(y)))
    height = PEAK_K * robust_std
    peaks, _ = find_peaks(y, height=height,
                          distance=int(MIN_BLINK_GAP_S * fs))
    return peaks


def quiet_mask(y: np.ndarray, peaks: np.ndarray, fs: float) -> np.ndarray:
    """Boolean mask of samples that are *not* near any blink (the baseline)."""
    guard = int(GUARD_S * fs)
    mask = np.ones(len(y), dtype=bool)
    for p in peaks:
        mask[max(0, p - guard):min(len(y), p + guard + 1)] = False
    return mask


def band_power(freqs: np.ndarray, psd: np.ndarray, band) -> float:
    """Integrate a Welch PSD over a frequency band (trapezoid)."""
    sel = (freqs >= band[0]) & (freqs <= band[1])
    if sel.sum() < 2:
        return float(psd[sel].sum())
    return float(np.trapezoid(psd[sel], freqs[sel]))


def epochs_around(y: np.ndarray, peaks: np.ndarray, fs: float) -> np.ndarray:
    """Concatenate fixed windows centred on each blink peak."""
    pre, post = int(EPOCH_PRE_S * fs), int(EPOCH_POST_S * fs)
    segs = [y[p - pre:p + post] for p in peaks
            if p - pre >= 0 and p + post <= len(y)]
    return np.concatenate(segs) if segs else np.array([])


def analyse(path: str) -> List[Result]:
    """Load one .fif, preprocess each active channel, and measure SNR."""
    import mne
    mne.set_log_level("ERROR")

    raw = mne.io.read_raw_fif(path, preload=True)
    fs = float(raw.info["sfreq"])
    data_uv = raw.get_data() * 1e6  # V -> uV
    names = list(raw.ch_names)
    markers = [(a["onset"], str(a["description"])) for a in raw.annotations]

    active = active_channels(data_uv, names)
    print(f"\n# {path}")
    print(f"  sfreq={fs:g} Hz  duration={raw.times[-1]:.1f} s  "
          f"markers={len(markers)}  active channels={[names[i] for i in active] or 'NONE'}")
    if markers:
        mtxt = ", ".join(f"{o:.2f}s({d})" for o, d in markers)
        print(f"  marker onsets: {mtxt}")

    results: List[Result] = []
    for ci in active:
        y = preprocess(data_uv[ci], fs)
        peaks = detect_blinks(y, fs)

        # --- amplitude SNR ---------------------------------------------------
        blink_amp = float(np.median(np.abs(y[peaks]))) if len(peaks) else 0.0
        qmask = quiet_mask(y, peaks, fs)
        noise_rms = float(np.sqrt(np.mean(y[qmask] ** 2))) if qmask.any() else np.nan
        amp_snr = blink_amp / noise_rms if noise_rms else np.nan
        amp_snr_db = 20.0 * np.log10(amp_snr) if amp_snr and amp_snr > 0 else np.nan

        # --- spectral SNR ----------------------------------------------------
        blink_sig = epochs_around(y, peaks, fs)
        baseline = y[qmask]
        nperseg = min(256, len(blink_sig), len(baseline))
        if nperseg >= 32:
            fb, pb = welch(blink_sig, fs=fs, nperseg=nperseg)
            fn, pn = welch(baseline, fs=fs, nperseg=nperseg)
            p_sig = band_power(fb, pb, BLINK_BAND)
            p_noise = band_power(fn, pn, BLINK_BAND)
            spec_snr = p_sig / p_noise if p_noise else np.nan
            spec_snr_db = 10.0 * np.log10(spec_snr) if spec_snr and spec_snr > 0 else np.nan
        else:
            spec_snr = spec_snr_db = np.nan

        results.append(Result(path, names[ci], len(peaks), blink_amp, noise_rms,
                              amp_snr, amp_snr_db, spec_snr, spec_snr_db))
    return results


def print_report(all_results: List[Result]) -> None:
    print("\n" + "=" * 78)
    print("SNR COMPARISON")
    print("=" * 78)
    hdr = (f"{'file : channel':<42}{'blinks':>7}{'blink uV':>9}"
           f"{'noise uV':>9}{'amp dB':>8}{'spec dB':>8}")
    print(hdr)
    print("-" * 78)
    for r in all_results:
        tag = f"{r.path.split('/')[-1]} : {r.channel}"
        print(f"{tag:<42}{r.n_blinks:>7}{r.blink_amp_uv:>9.1f}"
              f"{r.noise_rms_uv:>9.2f}{r.amp_snr_db:>8.1f}{r.spec_snr_db:>8.1f}")
    print("-" * 78)

    # head-to-head when exactly two results (one channel each)
    if len(all_results) == 2:
        a, b = all_results
        d_amp = a.amp_snr_db - b.amp_snr_db
        d_spec = a.spec_snr_db - b.spec_snr_db
        better = a if d_amp >= 0 else b
        print(f"\nAmplitude SNR difference (A - B): {d_amp:+.1f} dB")
        print(f"Spectral  SNR difference (A - B): {d_spec:+.1f} dB")
        print(f"Higher amplitude SNR: {better.path.split('/')[-1]}")
    print()


def make_plot(paths, out_png) -> None:
    """Optional: overlay preprocessed traces with detected blinks marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    mne.set_log_level("ERROR")

    fig, axes = plt.subplots(len(paths), 1, figsize=(11, 3 * len(paths)),
                             sharex=False)
    if len(paths) == 1:
        axes = [axes]
    for ax, path in zip(axes, paths):
        raw = mne.io.read_raw_fif(path, preload=True)
        fs = float(raw.info["sfreq"])
        data_uv = raw.get_data() * 1e6
        active = active_channels(data_uv, raw.ch_names)
        for ci in active:
            y = preprocess(data_uv[ci], fs)
            t = np.arange(len(y)) / fs
            ax.plot(t, y, lw=0.7, label=raw.ch_names[ci])
            peaks = detect_blinks(y, fs)
            ax.plot(peaks / fs, y[peaks], "rx", ms=6)
        for onset, _ in [(a["onset"], a["description"]) for a in raw.annotations]:
            ax.axvline(onset, color="green", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(path.split("/")[-1])
        ax.set_ylabel("uV")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"Saved plot -> {out_png}")


def main() -> None:
    path_a = os.path.join(DATA_DIR, FILE_A)
    path_b = os.path.join(DATA_DIR, FILE_B)

    all_results: List[Result] = []
    for path in (path_a, path_b):
        all_results.extend(analyse(path))
    print_report(all_results)

    if PLOT_PATH:
        make_plot([path_a, path_b], PLOT_PATH)


if __name__ == "__main__":
    main()