"""Load saved recordings into a uniform structure for the Processing Lab.

Supports the app's own BrainFlow CSV (``*_raw.csv`` + sibling ``*_meta.json``)
as well as MNE-readable formats (.fif / .edf / .set / .gdf).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from brainflow.board_shim import BoardIds, BoardShim
from brainflow.data_filter import DataFilter

from .exporter import ReviewMarker


class LoadError(RuntimeError):
    """Raised when a recording cannot be loaded."""


@dataclass
class LoadedRecording:
    eeg: np.ndarray  # (n_channels, n_samples), microvolts
    sampling_rate: int
    channel_names: List[str]
    markers: List[ReviewMarker] = field(default_factory=list)
    source_path: str = ""
    channel_types: List[str] = field(default_factory=list)
    electrodes: List[str] = field(default_factory=list)
    # Sample indices where native BLE dropped a packet (from _packet_loss.csv).
    loss_samples: List[int] = field(default_factory=list)

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


def _load_csv(path: str) -> LoadedRecording:
    try:
        data = DataFilter.read_file(path)
    except Exception as exc:
        raise LoadError(f"Could not read CSV: {exc}") from exc
    if data.ndim != 2 or data.size == 0:
        raise LoadError("CSV contains no data.")

    meta_path = _meta_path_for(path)
    if meta_path:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        eeg_rows = list(meta.get("eeg_channels", []))
        sr = int(meta.get("sampling_rate", 200))
        names = list(meta.get("channel_names",
                              [f"Ch{i + 1}" for i in range(len(eeg_rows))]))
        marker_channel = int(meta.get("marker_channel", 0))
        ch_types = list(meta.get("channel_types", []))
        electrodes = list(meta.get("electrodes", []))
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

    eeg_rows = [r for r in eeg_rows if r < data.shape[0]]
    eeg = np.ascontiguousarray(data[eeg_rows, :], dtype=np.float64)
    names = names[: len(eeg_rows)] or [f"Ch{i + 1}" for i in range(eeg.shape[0])]

    markers: List[ReviewMarker] = []
    if 0 <= marker_channel < data.shape[0]:
        row = data[marker_channel, :]
        for idx in np.flatnonzero(row != 0):
            code = int(row[idx])
            markers.append(ReviewMarker(int(idx), code, f"code {code}"))

    return LoadedRecording(eeg, sr, names, markers, path,
                           channel_types=ch_types[: len(eeg_rows)],
                           electrodes=electrodes[: len(eeg_rows)],
                           loss_samples=_load_loss_samples(path))


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

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        picks = list(range(len(raw.ch_names)))
    names = [raw.ch_names[i] for i in picks]
    sr = int(round(raw.info["sfreq"]))
    eeg = np.ascontiguousarray(raw.get_data(picks=picks) * 1e6, dtype=np.float64)  # V -> uV

    markers: List[ReviewMarker] = []
    for ann in raw.annotations:
        sample = int(round(ann["onset"] * sr))
        markers.append(ReviewMarker(sample, 0, str(ann["description"])))

    return LoadedRecording(eeg, sr, names, markers, path)
