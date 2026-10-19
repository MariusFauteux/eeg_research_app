"""Load saved recordings into a uniform structure for the Processing Lab.

Supports the app's own BrainFlow CSV (``*_raw.csv`` + sibling ``*_meta.json``)
as well as MNE-readable formats (.fif / .edf / .set / .gdf).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from brainflow.board_shim import BoardIds, BoardShim
from brainflow.data_filter import DataFilter

from .exporter import ReviewMarker
from .packet_loss import ANNOTATION_DESCRIPTION, loss_spans
from .eeg_gaps import IMPEDANCE_CHECK_ANNOTATION, crop_eeg_gaps, normalise_eeg_gaps


class LoadError(RuntimeError):
    """Raised when a recording cannot be loaded."""


@dataclass
class LoadedRecording:
    eeg: np.ndarray  # (n_channels, n_samples), microvolts
    sampling_rate: float
    channel_names: List[str]
    markers: List[ReviewMarker] = field(default_factory=list)
    source_path: str = ""
    channel_types: List[str] = field(default_factory=list)
    electrodes: List[str] = field(default_factory=list)
    # Per-channel 10-20 placement (from meta.json). Empty when unknown.
    placements: List[str] = field(default_factory=list)
    # Per-channel on/off state at record time (from meta.json). Empty when the
    # recording predates this field; a False channel was switched off (flat zero).
    channel_active: List[bool] = field(default_factory=list)
    # Sample indices where native BLE dropped a packet (from _packet_loss.csv).
    loss_samples: List[int] = field(default_factory=list)
    # Parallel lengths; absent in older logs means two samples per lost packet.
    loss_lengths: List[int] = field(default_factory=list)
    # Explicitly unmeasured EEG (e.g. a timed impedance check), never BLE loss.
    eeg_gaps: List[dict] = field(default_factory=list)
    impedance_path: str = ""

    @property
    def n_channels(self) -> int:
        return self.eeg.shape[0] if self.eeg.ndim == 2 else 0

    @property
    def n_samples(self) -> int:
        return self.eeg.shape[1] if self.eeg.ndim == 2 else 0


MNE_EXTENSIONS = {".fif", ".edf", ".bdf", ".set", ".gdf", ".vhdr"}


def load_recording(path: str) -> LoadedRecording:
    if not os.path.exists(path):
        raise LoadError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _load_csv(path)
    if ext in MNE_EXTENSIONS:
        return _load_mne(path)
    raise LoadError(f"Unsupported file type: {ext}")


# --------------------------------------------------------------------------- #
def _meta_path_for(path: str) -> Optional[str]:
    if path.endswith("_raw.csv"):
        cand = path[: -len("_raw.csv")] + "_meta.json"
    else:
        cand = os.path.splitext(path)[0] + "_meta.json"
    return cand if os.path.exists(cand) else None


def _impedance_path(path: str, reference) -> str:
    if not isinstance(reference, str) or not reference:
        return ""
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), reference))


def _load_loss_samples(path: str) -> List[int]:
    """Read the sibling ``_packet_loss.csv`` (sample_index column), if present."""
    if path.endswith("_raw.csv"):
        cand = path[: -len("_raw.csv")] + "_packet_loss.csv"
    else:
        cand = os.path.splitext(path)[0] + "_packet_loss.csv"
    if not os.path.exists(cand):
        return []
    out: List[int] = []
    try:
        with open(cand, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # header: sample_index,time_s
            for row in reader:
                if row:
                    out.append(int(float(row[0])))
    except Exception:
        return []
    return out


def _load_marker_labels(path: str) -> Dict[int, str]:
    """Read the sibling ``_markers.csv`` into a {code: label} map, if present.

    The raw CSV's marker channel only stores the numeric code, so to recover the
    human-readable name the user assigned we read it from the marker sidecar and
    map code -> label (the same idea as the live view's marker types).
    """
    if path.endswith("_raw.csv"):
        cand = path[: -len("_raw.csv")] + "_markers.csv"
    else:
        cand = os.path.splitext(path)[0] + "_markers.csv"
    labels: Dict[int, str] = {}
    if not os.path.exists(cand):
        return labels
    try:
        with open(cand, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)  # header: timestamp,time,code,label
            for row in reader:
                label = (row.get("label") or "").strip()
                if label:
                    labels[int(float(row["code"]))] = label
    except Exception:
        return {}
    return labels


def _load_sample_markers(path: str, n_samples: int) -> Optional[List[ReviewMarker]]:
    """Read the newer event log, retaining multiple markers at one sample.

    None means an older log without sample positions; its raw marker row is
    still authoritative. A blank position is an event without recorded data.
    """
    base = path[:-len("_raw.csv")] if path.endswith("_raw.csv") else os.path.splitext(path)[0]
    try:
        with open(base + "_markers.csv", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if "sample_index" not in (reader.fieldnames or []):
                return None
            events = []
            for row in reader:
                if not row.get("sample_index"):
                    continue
                try:
                    sample = int(row["sample_index"])
                    code = int(float(row["code"]))
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
                if 0 <= sample < n_samples:
                    events.append(ReviewMarker(sample, code, row.get("label") or f"code {code}"))
            return events
    except (OSError, UnicodeError, csv.Error):
        return None


def _load_csv(path: str) -> LoadedRecording:
    try:
        data = DataFilter.read_file(path)
    except Exception as exc:
        raise LoadError(f"Could not read CSV: {exc}") from exc
    if data.ndim != 2 or data.size == 0:
        raise LoadError("CSV contains no data.")

    meta = {}
    meta_path = _meta_path_for(path)
    if meta_path:
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError) as exc:
            raise LoadError(f"Could not read recording metadata: {exc}") from exc
        eeg_rows = list(meta.get("eeg_channels", []))
        try:
            sr = float(meta.get("sampling_rate", 200))
        except (TypeError, ValueError) as exc:
            raise LoadError("Sampling rate must be a positive finite number.") from exc
        if not np.isfinite(sr) or sr <= 0:
            raise LoadError("Sampling rate must be a positive finite number.")
        names = list(meta.get("channel_names",
                              [f"Ch{i + 1}" for i in range(len(eeg_rows))]))
        marker_channel = int(meta.get("marker_channel", 0))
        ch_types = list(meta.get("channel_types", []))
        electrodes = list(meta.get("electrodes", []))
        placements = list(meta.get("placements", []))
        channel_active = list(meta.get("channel_active", []))
    else:
        # No metadata: assume a Ganglion layout.
        bid = BoardIds.GANGLION_NATIVE_BOARD.value
        descr = BoardShim.get_board_descr(bid)
        eeg_rows = list(descr["eeg_channels"])
        sr = int(descr["sampling_rate"])
        names = [f"Ch{i + 1}" for i in range(len(eeg_rows))]
        marker_channel = int(descr.get("marker_channel", 0))
        ch_types = []
        electrodes = []
        placements = []
        channel_active = []

    eeg_rows = [r for r in eeg_rows if r < data.shape[0]]
    eeg = np.ascontiguousarray(data[eeg_rows, :], dtype=np.float64)
    names = names[: len(eeg_rows)] or [f"Ch{i + 1}" for i in range(eeg.shape[0])]

    code_labels = _load_marker_labels(path)
    markers = _load_sample_markers(path, data.shape[1])
    if markers is None and "marker_events" in meta:
        markers = []
        for event in meta["marker_events"]:
            try:
                sample, code = int(event["sample"]), int(event["code"])
                label = str(event.get("label") or f"code {code}")
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if 0 <= sample < data.shape[1]:
                markers.append(ReviewMarker(sample, code, label))
    if markers is None:
        markers = []
        if 0 <= marker_channel < data.shape[0]:
            row = data[marker_channel, :]
            for idx in np.flatnonzero(row != 0):
                code = int(row[idx])
                markers.append(ReviewMarker(int(idx), code,
                                            code_labels.get(code, f"code {code}")))

    gaps = loss_spans(_load_loss_samples(path), n_samples=eeg.shape[1])
    try:
        eeg_gaps = normalise_eeg_gaps(meta.get("eeg_gaps", []), eeg.shape[1], sr)
    except ValueError as exc:
        raise LoadError(f"Invalid missing EEG intervals: {exc}") from exc
    for gap in eeg_gaps:
        eeg[:, gap["start_sample"]:gap["start_sample"] + gap["length"]] = np.nan
    return LoadedRecording(eeg, sr, names, markers, path,
                           channel_types=ch_types[: len(eeg_rows)],
                           electrodes=electrodes[: len(eeg_rows)],
                           placements=placements[: len(eeg_rows)],
                           channel_active=channel_active[: len(eeg_rows)],
                           loss_samples=[sample for sample, length in gaps],
                           loss_lengths=[length for sample, length in gaps],
                           eeg_gaps=eeg_gaps,
                           impedance_path=_impedance_path(path, meta.get("impedance_file")))


def _load_mne(path: str) -> LoadedRecording:
    try:
        import mne
    except Exception as exc:
        raise LoadError("MNE is required to load this format. pip install mne") from exc

    mne.set_log_level("ERROR")
    ext = os.path.splitext(path)[1].lower()
    readers = {
        ".fif": mne.io.read_raw_fif,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".set": mne.io.read_raw_eeglab,
        ".gdf": mne.io.read_raw_gdf,
        ".vhdr": mne.io.read_raw_brainvision,
    }
    try:
        raw = readers[ext](path, preload=True)
    except Exception as exc:
        raise LoadError(f"Could not read {ext}: {exc}") from exc

    # The Ganglion's input channels can also carry ECG/EMG or another signal.
    # Keep those channels when a file records their actual types, while still
    # excluding auxiliary channels such as stimulation/status counters.
    picks = mne.pick_types(raw.info, eeg=True, ecg=True, emg=True, misc=True,
                           exclude=[])
    if len(picks) == 0:
        picks = list(range(len(raw.ch_names)))
    names = [raw.ch_names[i] for i in picks]
    sr = float(raw.info["sfreq"])
    eeg = np.ascontiguousarray(raw.get_data(picks=picks) * 1e6, dtype=np.float64)  # V -> uV

    markers: List[ReviewMarker] = []
    gaps = []
    eeg_gaps = []
    try:
        file_metadata = json.loads(raw.info.get("description") or "{}")
        native_meta = file_metadata.get("ganglion_studio", {})
        original_gaps = normalise_eeg_gaps(native_meta.get("eeg_gaps", []))
        gap_origin = int(native_meta.get("eeg_gap_origin_first_samp", 0))
    except (TypeError, ValueError, AttributeError):
        native_meta, original_gaps, gap_origin = {}, [], 0
    for ann in raw.annotations:
        # Annotation onsets include first_time for cropped FIFs and other files
        # whose acquisition sample counter does not start at zero.
        sample = int(round((ann["onset"] - raw.first_time) * raw.info["sfreq"]))
        if ann["description"] == ANNOTATION_DESCRIPTION:
            length = max(1, int(round(ann["duration"] * sr)))
            gaps.extend(loss_spans([sample], [length], raw.n_times))
        elif ann["description"] == IMPEDANCE_CHECK_ANNOTATION:
            length = max(1, int(round(ann["duration"] * sr)))
            annotated = normalise_eeg_gaps([
                {"start_sample": sample, "length": length, "reason": "impedance_check"}],
                raw.n_times,
            )
            for gap in annotated:
                absolute = int(raw.first_samp) - gap_origin + gap["start_sample"]
                # Annotations determine retained samples after an external crop;
                # native metadata supplies real wall timestamps when available.
                for original in original_gaps:
                    clipped = crop_eeg_gaps([original], absolute, absolute + gap["length"])
                    if clipped and clipped[0]["length"] == gap["length"]:
                        gap["started_at"] = clipped[0]["started_at"]
                        gap["ended_at"] = clipped[0]["ended_at"]
                        break
                eeg_gaps.append(gap)
        else:
            markers.append(ReviewMarker(sample, 0, str(ann["description"])))

    # The editor's sidecar also stores electrode material and placement, plus
    # channel edits from older files whose FIF metadata was always marked EEG.
    # Preserve those edits when reopening the file.
    ch_types = [kind.upper() for kind in raw.get_channel_types(picks=picks)]
    electrodes: List[str] = []
    placements: List[str] = []
    channel_active: List[bool] = []
    meta = {}
    meta_path = _meta_path_for(path)
    if meta_path:
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            saved_names = list(meta.get("channel_names", []))
            if len(saved_names) == len(names):
                names = saved_names
            ch_types = list(meta.get("channel_types", ch_types))[: len(names)]
            electrodes = list(meta.get("electrodes", []))[: len(names)]
            placements = list(meta.get("placements", []))[: len(names)]
            channel_active = list(meta.get("channel_active", []))[: len(names)]
        except Exception:
            pass

    # Older FIF exports may have a companion log; native annotations take
    # precedence because they follow external crop/concatenation operations.
    native_loss_schema = native_meta.get("packet_loss_annotations") == 1
    if not gaps and not native_loss_schema:
        gaps = loss_spans(_load_loss_samples(path), n_samples=raw.n_times)
    gaps = sorted(set(gaps))
    if not eeg_gaps and native_meta.get("eeg_gap_annotations") != 1:
        try:
            eeg_gaps = normalise_eeg_gaps(meta.get("eeg_gaps", []), raw.n_times, sr)
        except ValueError as exc:
            raise LoadError(f"Invalid missing EEG intervals: {exc}") from exc
    eeg_gaps = normalise_eeg_gaps(eeg_gaps, raw.n_times, sr)
    for gap in eeg_gaps:
        eeg[:, gap["start_sample"]:gap["start_sample"] + gap["length"]] = np.nan
    impedance_file = native_meta.get("impedance_file", meta.get("impedance_file"))
    return LoadedRecording(eeg, sr, names, markers, path,
                           channel_types=ch_types,
                           electrodes=electrodes,
                           placements=placements,
                           channel_active=channel_active,
                           loss_samples=[sample for sample, length in gaps],
                           loss_lengths=[length for sample, length in gaps],
                           eeg_gaps=eeg_gaps,
                           impedance_path=_impedance_path(path, impedance_file))
