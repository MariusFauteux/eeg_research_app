"""Session configuration, recording, and export.

Recording always stores *raw* unfiltered data (display filters never touch what
is written to disk). On stop we write:

* ``<session>_raw.csv``  - BrainFlow native format (DataFilter.write_file)
* ``<session>_meta.json`` - session metadata + channel map
* ``<session>_markers.csv`` - annotation log
* ``<session>_packet_loss.csv`` - sample indices where native BLE dropped a packet
* ``<session>_raw.edf``  - optional, only if MNE is installed
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

import numpy as np

from brainflow.data_filter import DataFilter


@dataclass
class SessionConfig:
    name: str = "session"
    demo: bool = False
    mac_address: str = ""
    serial_number: str = ""
    serial_port: str = ""  # set -> connect via the BLED112 dongle
    firmware: str = "3"
    notch_freq: int = 50
    # Native BLE only: use the custom bleak driver (True) vs BrainFlow's native
    # backend (False). Ignored for demo/dongle. See core/native_ganglion.py.
    use_custom_native: bool = True
    # Native BLE sample encoding: "delta" (firmware <= 2.x) or "msb" (firmware
    # 3.0.2+ sends absolute MSB-truncated samples). Custom native only.
    decode_mode: str = "delta"

    def safe_name(self) -> str:
        keep = "-_ "
        cleaned = "".join(c for c in self.name if c.isalnum() or c in keep).strip()
        return cleaned.replace(" ", "_") or "session"


@dataclass
class MarkerEvent:
    timestamp: float
    code: int
    label: str

    @property
    def time_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S.%f")[:-3]


@dataclass
class SessionRecorder:
    config: SessionConfig
    base_dir: str = "recordings"
    markers: List[MarkerEvent] = field(default_factory=list)
    started_at: Optional[datetime] = None
    out_dir: str = ""
    file_prefix: str = ""

    def begin(self) -> str:
        self.started_at = datetime.now()
        stamp = self.started_at.strftime("%Y%m%d_%H%M%S")
        self.file_prefix = f"{stamp}_{self.config.safe_name()}"
        self.out_dir = os.path.join(self.base_dir, self.file_prefix)
        os.makedirs(self.out_dir, exist_ok=True)
        return self.out_dir

    def add_marker(self, event: MarkerEvent) -> None:
        self.markers.append(event)

    # ------------------------------------------------------------- writing
    def _path(self, suffix: str) -> str:
        return os.path.join(self.out_dir, f"{self.file_prefix}{suffix}")

    def save(self, raw_data: np.ndarray, meta: dict,
             loss_samples: Optional[List[int]] = None) -> List[str]:
        """Persist the recording. Returns the list of written file paths.

        ``loss_samples`` are sample indices (within this recording) where the
        native BLE link dropped a packet; they are written to a sidecar CSV so the
        gaps can be inspected or excluded during offline processing.
        """
        if not self.out_dir:
            self.begin()
        written: List[str] = []
        loss_samples = list(loss_samples or [])

        if raw_data is not None and raw_data.size:
            csv_path = self._path("_raw.csv")
            to_write = self._with_acquisition_clock(raw_data, meta)
            DataFilter.write_file(np.ascontiguousarray(to_write), csv_path, "w")
            written.append(csv_path)

        meta_path = self._path("_meta.json")
        full_meta = {
            "config": asdict(self.config),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": datetime.now().isoformat(),
            "n_samples": int(raw_data.shape[1]) if raw_data is not None and raw_data.ndim == 2 else 0,
            "n_packet_losses": len(loss_samples),
            **meta,
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(full_meta, fh, indent=2)
        written.append(meta_path)

        marker_path = self._path("_markers.csv")
        with open(marker_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp", "time", "code", "label"])
            for ev in self.markers:
                writer.writerow([f"{ev.timestamp:.3f}", ev.time_str, ev.code, ev.label])
        written.append(marker_path)

        # Packet-loss sidecar: sample index + seconds from recording start.
        sr = max(1, int(meta.get("sampling_rate", 200)))
        loss_path = self._path("_packet_loss.csv")
        with open(loss_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["sample_index", "time_s"])
            for s in loss_samples:
                writer.writerow([int(s), f"{int(s) / sr:.4f}"])
        written.append(loss_path)

        edf = self._try_export_edf(raw_data, meta)
        if edf:
            written.append(edf)
        return written

    def _with_acquisition_clock(self, raw_data: np.ndarray, meta: dict) -> np.ndarray:
        """Return a copy of ``raw_data`` with a clean, evenly spaced time column.

        The native-BLE driver stamps every sample in a packet with the host's
        *arrival time* of that packet (``time.time()`` once per packet). Because
        each Ganglion packet carries two samples, both got the identical
        timestamp, and because the OS delivers BLE notifications in bursts the
        spacing jumped around (two samples at the same instant, then a gap). That
        column is therefore useless as a time axis.

        The board samples at a fixed rate, so we rebuild the timestamp row as a
        monotonic acquisition clock: ``t0 + sample_index / sampling_rate``. We
        anchor ``t0`` on the first real arrival timestamp (best wall-clock guess
        for sample 0), falling back to the recording start time. Only the
        timestamp row is touched; EEG values are copied through unchanged.
        """
        ts_row = int(meta.get("timestamp_channel", -1))
        sr = int(meta.get("sampling_rate", 200))
        if ts_row < 0 or ts_row >= raw_data.shape[0] or sr <= 0:
            return raw_data
        original = raw_data[ts_row]
        valid = original[np.isfinite(original) & (original > 0)]
        if valid.size:
            t0 = float(valid[0])
        elif self.started_at is not None:
            t0 = self.started_at.timestamp()
        else:
            t0 = 0.0
        out = raw_data.copy()
        out[ts_row] = t0 + np.arange(raw_data.shape[1], dtype=np.float64) / float(sr)
        return out

    def _try_export_edf(self, raw_data: np.ndarray, meta: dict) -> Optional[str]:
        if raw_data is None or raw_data.ndim != 2 or raw_data.shape[1] == 0:
            return None
        from .exporter import export, mne_available
        if not mne_available():
            return None
        try:
            # Reuse the central exporter (handles uV->V scaling + EEG channel
            # selection in one place). Auto-EDF is marker-free: markers are still
            # written to _markers.csv and the raw CSV marker channel, and
            # sample-aligned annotations are written by the Review window's
            # explicit export path.
            return export(self._path("_raw.edf"), "edf", raw_data, meta)
        except Exception:
            return None
