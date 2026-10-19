"""Rebuild a .fif straight from a session's original ``_raw.csv``.

The CSV written during recording is the untouched source of truth: raw counts
converted to microvolts, nothing filtered. This module reads it back (together
with its ``_meta.json`` and ``_markers.csv`` siblings) and writes a fresh .fif,
so a .fif that was exported with a display filter accidentally baked in can be
regenerated without re-running the session.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from . import exporter
from .recording_loader import LoadError, load_recording


@dataclass
class RebuildResult:
    """What the rebuild produced - shown back to the user."""

    fif_path: str
    sidecar_path: Optional[str]
    n_channels: int
    n_samples: int
    sampling_rate: float
    n_markers: int
    used_meta: bool
    duration_s: float


def default_output_path(csv_path: str) -> str:
    """``<session>_raw.csv`` -> ``<session>_raw.fif`` in the same folder."""
    base = csv_path[: -len("_raw.csv")] if csv_path.endswith("_raw.csv") \
        else os.path.splitext(csv_path)[0]
    return base + "_raw.fif"


def meta_path_for_csv(csv_path: str) -> str:
    """The ``_meta.json`` that belongs to ``csv_path`` (may not exist)."""
    base = csv_path[: -len("_raw.csv")] if csv_path.endswith("_raw.csv") \
        else os.path.splitext(csv_path)[0]
    return base + "_meta.json"


def rebuild(csv_path: str, out_path: Optional[str] = None) -> RebuildResult:
    """Write a .fif built from ``csv_path``. Returns what was written.

    Raises :class:`~ganglion_studio.core.recording_loader.LoadError` if the CSV
    cannot be read and :class:`~ganglion_studio.core.exporter.ExportError` if the
    .fif cannot be written.
    """
    rec = load_recording(csv_path)
    if rec.n_samples == 0:
        raise LoadError("The CSV contains no samples.")

    target = out_path or default_output_path(csv_path)
    meta = {
        "sampling_rate": rec.sampling_rate,
        "eeg_channels": list(range(rec.n_channels)),
        "channel_names": list(rec.channel_names),
        "channel_types": list(rec.channel_types),
        "loss_samples": list(rec.loss_samples),
        "loss_lengths": list(rec.loss_lengths),
        "eeg_gaps": list(rec.eeg_gaps),
        "impedance_file": rec.impedance_path,
    }
    written = exporter.export(target, "fif", rec.eeg, meta, rec.markers)
    sidecar = _write_sidecar(written, csv_path, rec)

    return RebuildResult(
        fif_path=written,
        sidecar_path=sidecar,
        n_channels=rec.n_channels,
        n_samples=rec.n_samples,
        sampling_rate=rec.sampling_rate,
        n_markers=len(rec.markers),
        used_meta=os.path.exists(meta_path_for_csv(csv_path)),
        duration_s=rec.n_samples / rec.sampling_rate if rec.sampling_rate else 0.0,
    )


def _write_sidecar(fif_path: str, csv_path: str, rec) -> Optional[str]:
    """Copy the per-channel info .fif cannot store next to the new file.

    Uses the base-name convention the loader looks for, so opening the rebuilt
    .fif in the Recording Editor restores channel type / electrode / placement.
    Refuses to touch the session's own ``_meta.json`` (which holds more than we
    would write back), so rebuilding can never damage the source recording.
    """
    sidecar = os.path.splitext(fif_path)[0] + "_meta.json"
    if os.path.abspath(sidecar) == os.path.abspath(meta_path_for_csv(csv_path)):
        return None

    n = rec.n_channels
    meta = {
        "sampling_rate": rec.sampling_rate,
        "eeg_channels": list(range(n)),
        "channel_names": list(rec.channel_names),
        "channel_types": _padded(rec.channel_types, n, "EEG"),
        "electrodes": _padded(rec.electrodes, n, ""),
        "placements": _padded(rec.placements, n, "None"),
        "channel_active": _padded(rec.channel_active, n, True),
        "loss_samples": list(rec.loss_samples),
        "loss_lengths": list(rec.loss_lengths),
        "eeg_gaps": list(rec.eeg_gaps),
        "impedance_file": rec.impedance_path,
        "rebuilt_from": os.path.basename(csv_path),
    }
    try:
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    except Exception:  # pragma: no cover - non-fatal, the .fif is already written
        return None
    return sidecar


def _padded(values: List, n: int, fill) -> List:
    """``values`` trimmed/padded to exactly ``n`` entries."""
    out = list(values)[:n]
    out.extend([fill] * (n - len(out)))
    return out
