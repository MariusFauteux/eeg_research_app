"""Capture raw Ganglion EEG over the custom native-BLE driver (no BrainFlow link).

A/B tool: stream N seconds straight from :class:`NativeGanglionClient`, save a
raw CSV + meta in the exact format ``diagnose_artifact.py`` expects, so you can
confirm the once-per-second pulse is gone *before* switching the app over.

Usage
-----
    .venv/bin/python native_stream_check.py <ble-address-or-uuid> [--seconds 60]
    .venv/bin/python native_stream_check.py <addr> --out recordings/native_check/run1

Get <ble-address-or-uuid> from the app's Scan list (on macOS it is a
CoreBluetooth UUID, on Linux/Windows a MAC). Then run the diagnosis the script
prints at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim
from brainflow.data_filter import DataFilter

from ganglion_studio.core.native_ganglion import NUM_EEG, NativeGanglionClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Native-BLE Ganglion capture (custom driver)")
    parser.add_argument("address", help="BLE address/UUID of the Ganglion (from the app's Scan)")
    parser.add_argument("--seconds", type=float, default=60.0, help="capture duration (default 60)")
    parser.add_argument("--out", default="", help="output prefix (default recordings/native_check/<stamp>)")
    args = parser.parse_args()

    # Static descriptor only (no connection): gives the row layout BrainFlow uses,
    # so the saved CSV is byte-for-byte compatible with diagnose_artifact.py.
    board_id = BoardIds.GANGLION_NATIVE_BOARD.value
    descr = BoardShim.get_board_descr(board_id)
    sr = int(descr["sampling_rate"])
    eeg_channels = list(descr["eeg_channels"])[:NUM_EEG]

    out_prefix = args.out
    if not out_prefix:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_prefix = os.path.join("recordings", "native_check", f"native_{stamp}")
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    client = NativeGanglionClient(
        address=args.address,
        num_rows=int(descr["num_rows"]),
        eeg_channels=eeg_channels,
        timestamp_channel=int(descr["timestamp_channel"]),
        marker_channel=int(descr["marker_channel"]),
    )

    print(f"Connecting to {args.address} ...")
    client.prepare_session()
    print("Connected. Streaming (accel off, 19-bit)...")
    client.start_stream(450000)
    client.config_board("N")  # disable accelerometer -> 19-bit EEG, matches the app

    chunks = []
    deadline = time.time() + args.seconds
    try:
        while time.time() < deadline:
            time.sleep(0.1)
            data = client.get_board_data()
            if data.shape[1]:
                chunks.append(data)
    finally:
        client.stop_stream()
        client.release_session()

    if not chunks:
        raise SystemExit("No data captured -- check the address and that the board is on.")

    raw = np.ascontiguousarray(np.concatenate(chunks, axis=1))
    n = raw.shape[1]
    print(f"Captured {n} samples ({n / sr:.1f} s).")

    csv_path = f"{out_prefix}_raw.csv"
    meta_path = f"{out_prefix}_meta.json"
    DataFilter.write_file(raw, csv_path, "w")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"sampling_rate": sr, "eeg_channels": eeg_channels, "board_id": board_id,
             "n_samples": n, "source": "native_stream_check"},
            fh, indent=2,
        )

    print(f"\nWrote:\n  {csv_path}\n  {meta_path}")
    print(f"\nNow diagnose the pulse:\n  .venv/bin/python diagnose_artifact.py {csv_path}")


if __name__ == "__main__":
    main()
