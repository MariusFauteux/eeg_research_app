"""Session configuration, recording, and export.

Recording always stores *raw* unfiltered data (display filters never touch what
is written to disk). On stop we write:

* ``<session>_raw.csv``  - BrainFlow native format (DataFilter.write_file)
* ``<session>_meta.json`` - session metadata + channel map
* ``<session>_markers.csv`` - annotation log
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
    firmware: str = "3"
    notch_freq: int = 50

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

    def save(self, raw_data: np.ndarray, meta: dict) -> List[str]:
        """Persist the recording. Returns the list of written file paths."""
        if not self.out_dir:
            self.begin()
        written: List[str] = []

        if raw_data is not None and raw_data.size:
            csv_path = self._path("_raw.csv")
            DataFilter.write_file(np.ascontiguousarray(raw_data), csv_path, "w")
            written.append(csv_path)

        meta_path = self._path("_meta.json")
        full_meta = {
            "config": asdict(self.config),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": datetime.now().isoformat(),
            "n_samples": int(raw_data.shape[1]) if raw_data is not None and raw_data.ndim == 2 else 0,
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

        edf = self._try_export_edf(raw_data, meta)
        if edf:
            written.append(edf)
        return written

    def _try_export_edf(self, raw_data: np.ndarray, meta: dict) -> Optional[str]:
        if raw_data is None or raw_data.ndim != 2 or raw_data.shape[1] == 0:
            return None
        try:
            import mne  # type: ignore
        except Exception:
            return None
        try:
            eeg_channels = meta.get("eeg_channels", [])
            ch_names = meta.get("channel_names", [f"Ch{i+1}" for i in range(len(eeg_channels))])
            sfreq = meta.get("sampling_rate", 200)
            eeg = raw_data[eeg_channels, :] * 1e-6  # uV -> V
            info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")
            mne_raw = mne.io.RawArray(eeg, info, verbose="ERROR")
            edf_path = self._path("_raw.edf")
            mne.export.export_raw(edf_path, mne_raw, fmt="edf", overwrite=True, verbose="ERROR")
            return edf_path
        except Exception:
            return None
