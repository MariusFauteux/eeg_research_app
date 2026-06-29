"""Localize the once-per-second pulse artifact in a raw Ganglion recording.

Read-only. Works on this app's ``*_raw.csv`` (BrainFlow native format, which
includes the timestamp row). It does NOT filter or alter anything.

What it answers
---------------
1. Are the pulses really 1.000 s apart with sub-sample jitter? (clock-locked tell)
2. Do they land at a constant phase within the 1 s frame? (frame-locked tell)
3. Do the pulses coincide with timestamp gaps/duplicates?
   - YES  -> dropped/duplicated BLE packet + delta-decompression glitch (stream).
   - NO   -> timestamps regular at the pulses -> a real coupled board event
            (BLE/LED) rather than packet loss; address with shielding, not code.

Usage
-----
    .venv/bin/python tools/diagnose_artifact.py recordings/<session>/<session>_raw.csv
    .venv/bin/python tools/diagnose_artifact.py recordings/<session>        # folder ok
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim
from brainflow.data_filter import DataFilter
from scipy.signal import find_peaks


def _resolve_csv(path: str) -> str:
    if os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, "*_raw.csv")))
        if not hits:
            sys.exit(f"No *_raw.csv found in {path}")
        return hits[0]
    return path


def _layout(csv_path: str):
    """Return (sampling_rate, eeg_rows, timestamp_row) from the sibling meta or
    the Ganglion board descriptor."""
    meta_path = csv_path[:-len("_raw.csv")] + "_meta.json" if csv_path.endswith("_raw.csv") else ""
    board_id = BoardIds.GANGLION_NATIVE_BOARD.value
    sr, eeg_rows = None, None
    if meta_path and os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        sr = int(meta.get("sampling_rate", 0)) or None
        eeg_rows = list(meta.get("eeg_channels", [])) or None
        board_id = int(meta.get("board_id", board_id))
    descr = BoardShim.get_board_descr(board_id)
    sr = sr or int(descr["sampling_rate"])
    eeg_rows = eeg_rows or list(descr["eeg_channels"])
    return sr, eeg_rows, int(descr["timestamp_channel"])


def main(path: str) -> None:
    csv_path = _resolve_csv(path)
    data = DataFilter.read_file(csv_path)            # (rows, samples), as recorded
    sr, eeg_rows, ts_row = _layout(csv_path)
    n = data.shape[1]
    print(f"file: {csv_path}")
    print(f"{data.shape[0]} rows x {n} samples | sr={sr} Hz | {n / sr:.1f} s "
          f"| eeg rows={eeg_rows} | timestamp row={ts_row}\n")

    # --- detect pulses on the channel with the largest outliers --------------
    ch = max(eeg_rows, key=lambda r: np.ptp(data[r]) if r < data.shape[0] else 0)
    x = data[ch].astype(float)
    x = x - np.median(x)
    mad = np.median(np.abs(x - np.median(x))) + 1e-9
    thresh = 8.0 * 1.4826 * mad                      # ~8 sigma, robust
    peaks, _ = find_peaks(np.abs(x), height=thresh, distance=int(0.8 * sr))
    print(f"detected {len(peaks)} pulses on row {ch} (>|{thresh:.0f}| uV, robust)")
    if len(peaks) < 3:
        print("Too few pulses to characterize. Lower the threshold or pick a "
              "channel with the artifact.")
        return

    # --- 1) spacing: clock-locked? -------------------------------------------
    spacing_s = np.diff(peaks) / sr
    print(f"\n[1] spacing: median={np.median(spacing_s):.4f} s  "
          f"mean={spacing_s.mean():.4f} s  std={spacing_s.std() * 1000:.1f} ms  "
          f"(jitter; <~15 ms = clock-locked, not HRV)")

    # --- 2) phase within the 1 s frame: frame-locked? ------------------------
    phase = peaks % sr
    # circular spread of the phase (0 = perfectly frame-locked)
    ang = 2 * np.pi * phase / sr
    r = np.abs(np.mean(np.exp(1j * ang)))
    print(f"[2] phase within 1 s frame: median sample={int(np.median(phase))}/{sr} "
          f"| concentration r={r:.3f} (1.0 = perfectly frame-locked)")

    # --- 3) timestamp regularity at the pulses -------------------------------
    if ts_row < data.shape[0]:
        ts = data[ts_row].astype(float)
        dt = np.diff(ts)
        exp = 1.0 / sr
        med_dt = np.median(dt)
        gaps = np.where(dt > 1.5 * exp)[0]      # >= ~1 missing-sample interval
        dups = np.where(dt < 0.5 * exp)[0]      # duplicated / bursted samples
        anom = np.union1d(gaps, dups)
        secs = n / sr
        print(f"\n[3] timestamps: median dt={med_dt * 1000:.2f} ms "
              f"(expected {exp * 1000:.2f}), max dt={dt.max() * 1000:.1f} ms")
        print(f"    {len(gaps)} gaps (>1.5x), {len(dups)} dups (<0.5x) over "
              f"{secs:.0f} s = {len(anom) / secs:.2f} anomalies/s")
        near = sum(1 for p in peaks
                   if anom.size and np.min(np.abs(anom - p)) <= 3)
        frac = near / len(peaks)
        # chance a pulse falls within +/-3 samples of an anomaly at random
        chance = min(1.0, len(anom) * 7.0 / n)
        print(f"    pulses within +/-3 samples of an anomaly: {near}/{len(peaks)} "
              f"({frac * 100:.0f}%); expected by chance ~{chance * 100:.0f}%")

        print("\nVERDICT:")
        if frac >= 0.6 and frac > 2 * chance:
            print("  -> pulses coincide with timestamp gaps/dupes far above chance:")
            print("     dropped/duplicated BLE packets + delta-decompression glitch.")
            print("     Inspect BrainFlow Ganglion packet handling and BLE link.")
        elif len(anom) <= 1:
            print("  -> timestamps are essentially perfectly regular. Either no")
            print("     packet loss, OR BrainFlow back-fills a constant rate and")
            print("     hides it. INCONCLUSIVE from timestamps alone.")
        else:
            print("  -> anomalies exist but do NOT line up with the pulses.")
            print("     Leans toward a real coupled board event, not packet loss.")
        print("  Decisive next step: record the SAME montage in the official")
        print("  OpenBCI GUI. Present there too -> it's the stream/board (not")
        print("  this app). Absent there -> compare acquisition settings.")
        print("  Hardware control: repeat the bench test with ~1 MOhm in series.")
    else:
        print("\n[3] no timestamp row in file; cannot test packet loss.")

    print("\n(Reminder: this app records RAW and never per-chunk filters, so the "
          "pulse is not introduced by Ganglion EEG Studio's processing.)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
