"""Export recordings to research formats (.fif / .set / .edf / .gdf).

Built on MNE. ``.fif`` is MNE-native and lossless; ``.set`` (EEGLAB) and
``.edf`` are written through MNE's export framework (``eeglabio`` / ``edfio``).

``.gdf`` is *not* writable by MNE or any standard pip library (MNE can only read
GDF). We attempt it through the optional BioSig toolkit if it is installed,
otherwise we raise :class:`ExportError` recommending EDF, which is the closest
widely-supported open standard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


class ExportError(RuntimeError):
    """Raised when an export cannot be completed."""


@dataclass
class ReviewMarker:
    """A marker positioned at a sample index within a recording."""

    sample: int
    code: int
    label: str

    def onset(self, sampling_rate: int) -> float:
        return self.sample / float(sampling_rate)


# Format key -> (description, file extension).
FORMATS: Dict[str, Tuple[str, str]] = {
    "fif": ("MNE native (lossless)", ".fif"),
    "set": ("EEGLAB", ".set"),
    "edf": ("European Data Format", ".edf"),
    "gdf": ("General Data Format (BioSig)", ".gdf"),
}


def mne_available() -> bool:
    try:
        import mne  # noqa: F401
        return True
    except Exception:
        return False


def biosig_available() -> bool:
    try:
        import biosig  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def available_formats() -> Dict[str, bool]:
    """Return which formats can currently be written on this machine."""
    has_mne = mne_available()
    return {
        "fif": has_mne,
        "set": has_mne and _module_available("eeglabio"),
        "edf": has_mne and _module_available("edfio"),
        "gdf": biosig_available(),
    }


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def build_raw(raw_data: np.ndarray, meta: dict,
              markers: Optional[List[ReviewMarker]] = None):
    """Construct an ``mne.io.RawArray`` (EEG only) with annotations attached."""
    if not mne_available():
        raise ExportError("MNE is not installed. Run: pip install mne eeglabio edfio")
    import mne

    if raw_data is None or raw_data.ndim != 2 or raw_data.shape[1] == 0:
        raise ExportError("Recording is empty - nothing to export.")

    sfreq = int(meta.get("sampling_rate", 200))
    eeg_channels = list(meta.get("eeg_channels", []))
    if not eeg_channels:
        raise ExportError("No EEG channels found in recording metadata.")
    ch_names = list(meta.get("channel_names",
                             [f"Ch{i + 1}" for i in range(len(eeg_channels))]))

    eeg = np.ascontiguousarray(raw_data[eeg_channels, :], dtype=np.float64) * 1e-6  # uV -> V
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg, info, verbose="ERROR")

    if markers:
        onsets = [m.onset(sfreq) for m in markers]
        durations = [0.0] * len(markers)
        descriptions = [m.label or f"code {m.code}" for m in markers]
        raw.set_annotations(
            mne.Annotations(onset=onsets, duration=durations, description=descriptions),
            verbose="ERROR",
        )
    return raw


def export(path: str, fmt: str, raw_data: np.ndarray, meta: dict,
           markers: Optional[List[ReviewMarker]] = None) -> str:
    """Export the recording to ``path`` in ``fmt``. Returns the written path."""
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ExportError(f"Unknown format: {fmt}")

    if fmt == "gdf":
        return _export_gdf(path, raw_data, meta, markers)

    raw = build_raw(raw_data, meta, markers)
    import mne

    if fmt == "fif":
        # MNE expects raw fif files to end with "_raw.fif" / "raw.fif".
        if not path.endswith("raw.fif"):
            path = path[:-4] + "_raw.fif" if path.endswith(".fif") else path + "_raw.fif"
        raw.save(path, overwrite=True, verbose="ERROR")
    elif fmt == "set":
        if not _module_available("eeglabio"):
            raise ExportError("EEGLAB export needs 'eeglabio'. Run: pip install eeglabio")
        mne.export.export_raw(path, raw, fmt="eeglab", overwrite=True, verbose="ERROR")
    elif fmt == "edf":
        if not _module_available("edfio"):
            raise ExportError("EDF export needs 'edfio'. Run: pip install edfio")
        mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose="ERROR")
    return path


def _export_gdf(path: str, raw_data: np.ndarray, meta: dict,
                markers: Optional[List[ReviewMarker]]) -> str:
    if not biosig_available():
        raise ExportError(
            "GDF writing is not supported by MNE/standard Python libraries.\n"
            "Install the optional BioSig toolkit (https://biosig.sourceforge.io) "
            "to enable it, or export to EDF instead (closest open standard)."
        )
    # BioSig's Python API is platform-specific; we delegate and surface any error.
    try:  # pragma: no cover - depends on optional native toolkit
        import biosig  # type: ignore

        sfreq = int(meta.get("sampling_rate", 200))
        eeg_channels = list(meta.get("eeg_channels", []))
        data = np.ascontiguousarray(raw_data[eeg_channels, :], dtype=np.float64)
        biosig.save(path, data.T, sfreq)  # type: ignore[attr-defined]
        return path
    except Exception as exc:  # pragma: no cover
        raise ExportError(f"GDF export via BioSig failed: {exc}") from exc
