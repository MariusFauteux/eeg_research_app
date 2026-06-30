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

# This script lives in tools/ but reuses the app's shared analysis so the offline
# verdict and the live Diagnostics tab can never drift apart. Put the repo root
# (parent of tools/) on sys.path so `import ganglion_studio` works when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ganglion_studio.core import pulse_diagnostics as pdiag


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

    # Channel with the largest outliers (most likely to carry the pulse).
    ch = max(eeg_rows, key=lambda r: np.ptp(data[r]) if r < data.shape[0] else 0)
    x = data[ch].astype(float)

    # Per-sample anomaly flags for the coincidence test. Offline we infer packet
    # loss from the timestamp row: an interval much longer/shorter than 1/sr is a
    # gap/dup. (Live, the app supplies exact BLE-loss flags instead.)
    has_ts = ts_row < data.shape[0]
    gaps = dups = np.empty(0, dtype=int)
    anomaly_flags = None
    if has_ts:
        dt = np.diff(data[ts_row].astype(float))
        exp = 1.0 / sr
        gaps = np.where(dt > 1.5 * exp)[0]      # >= ~1 missing-sample interval
        dups = np.where(dt < 0.5 * exp)[0]      # duplicated / bursted samples
        anomaly_flags = np.zeros(n)
        anomaly_flags[np.union1d(gaps, dups)] = 1.0

    # The actual analysis lives in core.pulse_diagnostics (shared with the live tab).
    res = pdiag.diagnose(x, sr, anomaly_flags)
    print(f"detected {res.n_pulses} pulses on row {ch} (robust ~8 sigma)")
    if res.n_pulses < 3:
        print("Too few pulses to characterize. Lower the threshold or pick a "
              "channel with the artifact.")
        return

    ph = pdiag.phase_concentration(res.peaks, sr)
    print(f"\n[1] spacing: median rate={res.rate_hz:.3f} Hz  "
          f"jitter={res.jitter_ms:.1f} ms  (<~15 ms = clock-locked, not HRV)")
    print(f"[2] phase within 1 s frame: median sample={ph['median_phase_sample']}/{sr} "
          f"| concentration r={res.phase_r:.3f} (1.0 = perfectly frame-locked)")

    if has_ts:
        dt = np.diff(data[ts_row].astype(float))
        secs = n / sr
        print(f"\n[3] timestamps: median dt={np.median(dt) * 1000:.2f} ms "
              f"(expected {1000.0 / sr:.2f}), max dt={dt.max() * 1000:.1f} ms")
        print(f"    {len(gaps)} gaps (>1.5x), {len(dups)} dups (<0.5x) over "
              f"{secs:.0f} s = {res.n_anomalies / secs:.2f} anomalies/s")
        print(f"    pulses within +/-3 samples of an anomaly: "
              f"{res.loss_fraction * 100:.0f}%; expected by chance "
              f"~{res.loss_chance * 100:.0f}%")
    else:
        print("\n[3] no timestamp row in file; cannot test packet loss.")

    print("\nVERDICT:")
    print(f"  -> {res.message}")
    print("  Decisive next step: record the SAME montage in the official OpenBCI")
    print("  GUI. Present there too -> it's the stream/board (not this app).")
    print("  Hardware control: repeat the bench test with ~1 MOhm in series.")

    print("\n(Reminder: this app records RAW and never per-chunk filters, so the "
          "pulse is not introduced by Ganglion EEG Studio's processing.)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
