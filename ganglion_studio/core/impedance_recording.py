"""Save and reopen timestamped impedance measurements, without an EEG clock.

Each CSV row is one actual channel update. ``elapsed_s`` is measured from
``begin()``; timestamps are never resampled and repeated values are retained.
The canonical stored resistance is ohms. The kOhm column is a readable copy.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from numbers import Integral
from typing import List, Optional

from .impedance import ImpedanceReading
from .session import SessionConfig

CSV_COLUMNS = ("timestamp", "elapsed_s", "channel_index", "channel_name",
               "impedance_ohm", "impedance_kohm")
DEFAULT_CHANNEL_NAMES = ["Ch1", "Ch2", "Ch3", "Ch4", "REF"]


class ImpedanceRecordingError(RuntimeError):
    """An impedance recording could not be saved or read safely."""


@dataclass
class LoadedImpedanceRecording:
    readings: List[ImpedanceReading]
    channel_names: List[str]
    started_at: float
    source_path: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.readings[-1].timestamp - self.started_at) if self.readings else 0.0


def _names(channel_names) -> List[str]:
    try:
        names = list(channel_names)
    except TypeError as exc:
        raise ImpedanceRecordingError("Channel names must be a list of names.") from exc
    if len(names) > 5 or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ImpedanceRecordingError("Provide up to five non-empty channel names (Ch1–Ch4 and REF).")
    return names + DEFAULT_CHANNEL_NAMES[len(names):]


def _finite(value, description: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ImpedanceRecordingError(f"{description} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ImpedanceRecordingError(f"{description} must be a finite number.")
    return number


def _reading(timestamp, channel, ohms, description: str) -> ImpedanceReading:
    timestamp = _finite(timestamp, f"{description} timestamp")
    ohms = _finite(ohms, f"{description} impedance")
    if isinstance(channel, bool) or not isinstance(channel, Integral) or not 0 <= channel <= 4:
        raise ImpedanceRecordingError(f"{description} channel index must be an integer from 0 to 4.")
    if ohms < 0:
        raise ImpedanceRecordingError(f"{description} impedance cannot be negative.")
    return ImpedanceReading(timestamp, int(channel), ohms)


def _cadence(readings) -> dict:
    channels = {}
    for channel in range(5):
        times = [reading.timestamp for reading in readings if reading.channel == channel]
        intervals = [right - left for left, right in zip(times, times[1:])]
        span = times[-1] - times[0] if times else 0.0
        channels[str(channel)] = {
            "n_readings": len(times),
            "min_interval_s": min(intervals) if intervals else None,
            "max_interval_s": max(intervals) if intervals else None,
            "mean_interval_s": sum(intervals) / len(intervals) if intervals else None,
            "observed_update_rate_hz": (len(times) - 1) / span if span > 0 else None,
        }
    return {"basis": "actual per-channel host timestamps; updates may be asynchronous",
            "per_channel": channels}


@dataclass
class ImpedanceRecorder:
    config: SessionConfig
    base_dir: str = "recordings"
    started_at: Optional[datetime] = None
    out_dir: str = ""
    file_prefix: str = ""
    _saved_signature: Optional[str] = field(default=None, init=False, repr=False)

    @property
    def outdir(self) -> str:
        return self.out_dir

    def begin(self) -> str:
        started_at = datetime.now().astimezone()
        prefix = f"{started_at:%Y%m%d_%H%M%S}_{self.config.safe_name()}_impedance"
        sequence = 1
        while True:
            candidate = prefix if sequence == 1 else f"{prefix}_{sequence}"
            out_dir = os.path.join(self.base_dir, candidate)
            try:
                os.makedirs(out_dir, exist_ok=False)
                break
            except FileExistsError:
                sequence += 1
            except OSError as exc:
                raise ImpedanceRecordingError(f"Could not create impedance recording: {exc}") from exc
        self.started_at = started_at
        self.file_prefix = candidate
        self.out_dir = out_dir
        self._saved_signature = None
        return out_dir

    def save(self, readings: List[ImpedanceReading], channel_names: List[str],
             metadata: Optional[dict] = None) -> List[str]:
        """Write all updates chronologically; retry failures at the same paths.

        A successful save is immutable. Saving the same input again returns its
        paths; call ``begin()`` to start another recording before changing it.
        """
        names = _names(channel_names)
        if metadata is not None and not isinstance(metadata, dict):
            raise ImpedanceRecordingError("Additional recording metadata must be a dictionary.")
        clean = []
        for index, reading in enumerate(readings, 1):
            try:
                clean.append(_reading(reading.timestamp, reading.channel, reading.ohms,
                                      f"Reading {index}"))
            except AttributeError as exc:
                raise ImpedanceRecordingError(f"Reading {index} is not an impedance measurement.") from exc
        clean.sort(key=lambda reading: reading.timestamp)  # stable, including simultaneous updates
        try:
            signature_payload = json.dumps(
                {"readings": [asdict(reading) for reading in clean], "channel_names": names,
                 "metadata": metadata or {}, "config": asdict(self.config)},
                sort_keys=True, allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ImpedanceRecordingError(f"Recording metadata is not valid JSON: {exc}") from exc
        signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
        if self._saved_signature is not None and signature != self._saved_signature:
            raise ImpedanceRecordingError("This recording is already saved. Begin a new recording to save different data.")
        if not self.out_dir:
            self.begin()
        csv_path = os.path.join(self.out_dir, self.file_prefix + ".csv")
        meta_path = os.path.join(self.out_dir, self.file_prefix + "_meta.json")
        if self._saved_signature is not None:
            return [csv_path, meta_path]
        assert self.started_at is not None
        start = self.started_at.timestamp()
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(CSV_COLUMNS)
        for reading in clean:
            writer.writerow([repr(reading.timestamp), repr(reading.timestamp - start),
                             reading.channel, names[reading.channel], repr(reading.ohms),
                             repr(reading.ohms / 1000.0)])
        csv_bytes = output.getvalue().encode("utf-8")
        saved_at = datetime.now().astimezone()
        full_meta = {
            "schema_version": 1,
            "mode": "impedance",
            "units": "ohms",
            "payload_units": getattr(self.config, "impedance_units", "unknown"),
            "config": asdict(self.config),
            "started_at_unix": start,
            "started_at_utc": self.started_at.astimezone(timezone.utc).isoformat(),
            "started_at_local": self.started_at.isoformat(),
            "saved_at_utc": saved_at.astimezone(timezone.utc).isoformat(),
            "saved_at_local": saved_at.isoformat(),
            "elapsed_origin": "recording_start",
            "timestamp_units": "Unix seconds (host receipt time)",
            "channel_names": names,
            "channel_map": [{"index": i, "name": name} for i, name in enumerate(names)],
            "n_readings": len(clean),
            "cadence": _cadence(clean),
            "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
            "extra": metadata or {},
        }
        try:
            meta_bytes = json.dumps(full_meta, indent=2, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ImpedanceRecordingError(f"Could not encode impedance metadata: {exc}") from exc
        staged = []
        try:
            # Stage both complete files before publishing either one. A retry
            # can finish this recorder's partial write without touching another
            # session. The checksum detects an interrupted two-file publication.
            for payload in (csv_bytes, meta_bytes):
                with tempfile.NamedTemporaryFile(dir=self.out_dir, prefix=".impedance-",
                                                 delete=False) as fh:
                    staged.append(fh.name)
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
            os.replace(staged[0], csv_path)
            os.replace(staged[1], meta_path)
        except OSError as exc:
            raise ImpedanceRecordingError(f"Could not save impedance recording: {exc}") from exc
        finally:
            for path in staged:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        self._saved_signature = signature
        return [csv_path, meta_path]


def load_impedance_recording(path: str) -> LoadedImpedanceRecording:
    """Load the impedance CSV and its optional metadata; reject inconsistent rows."""
    path = os.fspath(path)
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    try:
        with open(path, "rb") as fh:
            csv_bytes = fh.read()
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if not isinstance(meta, dict) or meta.get("mode") != "impedance":
                raise ImpedanceRecordingError("Metadata does not describe an impedance recording.")
            if meta.get("schema_version") != 1 or meta.get("units") != "ohms":
                raise ImpedanceRecordingError("Unsupported impedance metadata schema or units.")
            checksum = meta.get("csv_sha256")
            if checksum and checksum != hashlib.sha256(csv_bytes).hexdigest():
                raise ImpedanceRecordingError("The impedance CSV does not match its metadata checksum.")
        names = _names(meta.get("channel_names", []))
        start = (_finite(meta["started_at_unix"], "Recording start")
                 if "started_at_unix" in meta else None)
        reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"), newline=""))
        if reader.fieldnames != list(CSV_COLUMNS):
            raise ImpedanceRecordingError("CSV is missing the impedance measurement columns.")
        readings = []
        seen_names = {}
        for row_number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ImpedanceRecordingError(f"CSV row {row_number} has missing or extra values.")
            try:
                channel = int(row["channel_index"])
            except ValueError as exc:
                raise ImpedanceRecordingError(f"CSV row {row_number} has an invalid channel index.") from exc
            reading = _reading(row["timestamp"], channel, row["impedance_ohm"], f"CSV row {row_number}")
            kohms = _finite(row["impedance_kohm"], f"CSV row {row_number} kOhm value")
            if not math.isclose(kohms * 1000, reading.ohms, rel_tol=1e-10, abs_tol=1e-9):
                raise ImpedanceRecordingError(f"CSV row {row_number} has inconsistent ohm/kOhm values.")
            elapsed = _finite(row["elapsed_s"], f"CSV row {row_number} elapsed time")
            if start is None:
                start = reading.timestamp - elapsed
            if not math.isclose(reading.timestamp - start, elapsed, rel_tol=1e-12, abs_tol=1e-6):
                raise ImpedanceRecordingError(f"CSV row {row_number} has an inconsistent elapsed time.")
            name = row["channel_name"]
            if not name.strip() or (meta and names[channel] != name) \
                    or (channel in seen_names and seen_names[channel] != name):
                raise ImpedanceRecordingError(f"CSV row {row_number} has an inconsistent channel name.")
            seen_names[channel] = name
            names[channel] = name
            if readings and reading.timestamp < readings[-1].timestamp:
                raise ImpedanceRecordingError(f"CSV row {row_number} is out of chronological order.")
            readings.append(reading)
        if "n_readings" in meta and meta["n_readings"] != len(readings):
            raise ImpedanceRecordingError("The impedance recording is incomplete: measurement count does not match metadata.")
        return LoadedImpedanceRecording(readings, names, start if start is not None else 0.0, path, meta)
    except ImpedanceRecordingError:
        raise
    except (OSError, ValueError, UnicodeError, TypeError) as exc:
        raise ImpedanceRecordingError(f"Could not read impedance recording: {exc}") from exc
