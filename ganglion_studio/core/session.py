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
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

import numpy as np

from brainflow.data_filter import DataFilter

from .eeg_gaps import normalise_eeg_gaps


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
    # Impedance is a separate hardware acquisition mode; it does not contain EEG.
    acquisition_mode: str = "eeg"
    # Current published firmware sends integer kOhm; older firmware used Ohm.
    # Keep this explicit because the wire payload carries no unit information.
    impedance_units: str = "kohms"

    def safe_name(self) -> str:
        keep = "-_ "
        cleaned = "".join(c for c in self.name if c.isalnum() or c in keep).strip()
        return cleaned.replace(" ", "_") or "session"


@dataclass
class MarkerEvent:
    timestamp: float
    code: int
    label: str
    sample_index: Optional[int] = None

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
    _markers_aligned: bool = field(default=False, init=False, repr=False)
    _impedance_recorder: Optional[object] = field(default=None, init=False, repr=False)

    def begin(self) -> str:
        self.started_at = datetime.now()
        stamp = self.started_at.strftime("%Y%m%d_%H%M%S")
        prefix = f"{stamp}_{self.config.safe_name()}"
        sequence = 1
        while True:
            candidate = prefix if sequence == 1 else f"{prefix}_{sequence}"
            out_dir = os.path.join(self.base_dir, candidate)
            try:
                # Reserve a new directory atomically: two recordings can begin
                # within the same second, and must never overwrite one another.
                os.makedirs(out_dir, exist_ok=False)
            except FileExistsError:
                sequence += 1
                continue
            self.file_prefix = candidate
            self.out_dir = out_dir
            self._impedance_recorder = None
            break
        return self.out_dir

    def add_marker(self, event: MarkerEvent) -> None:
        self.markers.append(event)

    def align_markers(self, events, code_labels: Optional[dict] = None):
        """Attach acquisition sample positions without losing colliding events.

        ``events`` contains (sample or None, code, insertion timestamp). The
        unrounded timestamp is shared with the UI log, so each event retains
        its own label even if several events have the same code or sample.
        """
        from .exporter import ReviewMarker

        self._markers_aligned = True
        pending = defaultdict(deque)
        for event in self.markers:
            pending[(event.timestamp, event.code)].append(event)
        labels = code_labels or {}
        aligned = []
        for sample, code, timestamp in events:
            code = int(code)
            matches = pending[(timestamp, code)]
            if matches:
                event = matches.popleft()
            else:
                event = MarkerEvent(timestamp, code, labels.get(code, f"code {code}"))
                self.markers.append(event)
            event.sample_index = sample
            if sample is not None:
                aligned.append(ReviewMarker(sample, code, event.label))
        return aligned

    # ------------------------------------------------------------- writing
    def _path(self, suffix: str) -> str:
        return os.path.join(self.out_dir, f"{self.file_prefix}{suffix}")

    def save(self, raw_data: np.ndarray, meta: dict,
             loss_samples: Optional[List[int]] = None,
             impedance_readings: Optional[list] = None) -> List[str]:
        """Persist the recording. Returns the list of written file paths.

        ``loss_samples`` are sample indices (within this recording) where the
        native BLE link dropped a packet; they are written to a sidecar CSV so the
        gaps can be inspected or excluded during offline processing.
        """
        if not self.out_dir:
            self.begin()
        written: List[str] = []
        loss_samples = list(loss_samples or [])
        meta = dict(meta)
        n_samples = int(raw_data.shape[1]) if raw_data is not None and raw_data.ndim == 2 else 0
        eeg_gaps = normalise_eeg_gaps(meta.get("eeg_gaps", []), n_samples,
                                      meta.get("sampling_rate"))
        if eeg_gaps or "eeg_gaps" in meta:
            meta["eeg_gaps"] = eeg_gaps
        if impedance_readings is not None:
            from .impedance_recording import ImpedanceRecorder

            if self._impedance_recorder is None:
                # The EEG session already reserved this directory. A retained
                # writer shares its start clock and retries at stable paths.
                self._impedance_recorder = ImpedanceRecorder(
                    self.config, self.base_dir,
                    started_at=self.started_at.astimezone(), out_dir=self.out_dir,
                    file_prefix=self.file_prefix + "_impedance",
                )
            impedance_readings = list(impedance_readings)
            impedance_names = list(meta.get("channel_names", []))[:4]
            impedance_names.extend(f"Ch{i + 1}" for i in range(len(impedance_names), 4))
            imp_paths = self._impedance_recorder.save(
                impedance_readings, impedance_names + ["REF"],
                metadata={"context": "timed_impedance_checks",
                          "eeg_file": os.path.basename(self._path("_raw.csv")),
                          "eeg_metadata_file": os.path.basename(self._path("_meta.json")),
                          "eeg_gaps": eeg_gaps,
                          "impedance_raw_unit_basis": self.config.impedance_units},
            )
            written.extend(imp_paths)
            meta["impedance_file"] = os.path.basename(imp_paths[0])
            meta["impedance_metadata_file"] = os.path.basename(imp_paths[1])
            meta["n_impedance_readings"] = len(impedance_readings)

        if raw_data is not None and raw_data.size:
            csv_path = self._path("_raw.csv")
            to_write = self._with_acquisition_clock(raw_data, meta)
            if eeg_gaps:
                to_write = to_write.copy()
                for gap in eeg_gaps:
                    for row in meta.get("eeg_channels", []):
                        to_write[row, gap["start_sample"]:gap["start_sample"] + gap["length"]] = np.nan
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
            aligned = self._markers_aligned or any(ev.sample_index is not None for ev in self.markers)
            header = ["timestamp", "time", "code", "label"]
            writer.writerow(header + (["sample_index"] if aligned else []))
            for ev in self.markers:
                row = [f"{ev.timestamp:.9f}", ev.time_str, ev.code, ev.label]
                if aligned:
                    row.append(ev.sample_index if ev.sample_index is not None else "")
                writer.writerow(row)
        written.append(marker_path)

        # Packet-loss sidecar: sample index + seconds from recording start.
        sr = float(meta.get("sampling_rate", 200))
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
        sr = float(meta.get("sampling_rate", 200))
        if ts_row < 0 or ts_row >= raw_data.shape[0] or not np.isfinite(sr) or sr <= 0:
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
            # selection in one place). Live sessions supply sample-aligned
            # marker_events in metadata, including collisions at one sample.
            return export(self._path("_raw.edf"), "edf", raw_data, meta)
        except Exception:
            return None
