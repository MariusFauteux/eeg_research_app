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
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from .packet_loss import ANNOTATION_DESCRIPTION, loss_spans
from .eeg_gaps import IMPEDANCE_CHECK_ANNOTATION, normalise_eeg_gaps


class ExportError(RuntimeError):
    """Raised when an export cannot be completed."""


@dataclass
class ReviewMarker:
    """A marker positioned at a sample index within a recording."""

    sample: int
    code: int
    label: str

    def onset(self, sampling_rate: float) -> float:
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
    """Construct an ``mne.io.RawArray`` from the recorded signal channels."""
    if not mne_available():
        raise ExportError("MNE is not installed. Run: pip install mne eeglabio edfio")
    import mne

    if raw_data is None or raw_data.ndim != 2 or raw_data.shape[1] == 0:
        raise ExportError("Recording is empty - nothing to export.")

    try:
        sfreq = float(meta.get("sampling_rate", 200))
    except (TypeError, ValueError) as exc:
        raise ExportError("Sampling rate must be a positive finite number.") from exc
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ExportError("Sampling rate must be a positive finite number.")
    eeg_channels = list(meta.get("eeg_channels", []))
    if not eeg_channels:
        raise ExportError("No EEG channels found in recording metadata.")
    ch_names = list(meta.get("channel_names",
                             [f"Ch{i + 1}" for i in range(len(eeg_channels))]))
    saved_types = list(meta.get("channel_types", []))
    ch_types = [str(saved_types[i]).lower() if i < len(saved_types) else "eeg"
                for i in range(len(eeg_channels))]

    eeg = np.ascontiguousarray(raw_data[eeg_channels, :], dtype=np.float64) * 1e-6  # uV -> V
    try:
        eeg_gaps = normalise_eeg_gaps(meta.get("eeg_gaps", []), eeg.shape[1], sfreq)
    except ValueError as exc:
        raise ExportError(f"Invalid missing EEG intervals: {exc}") from exc
    for gap in eeg_gaps:
        eeg[:, gap["start_sample"]:gap["start_sample"] + gap["length"]] = np.nan
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg, info, verbose="ERROR")
    # An explicit schema also represents zero gaps. It survives external crops,
    # so a removed last annotation cannot revive an obsolete companion log.
    raw.info["description"] = json.dumps({"ganglion_studio": {
        "packet_loss_annotations": 1,
        "eeg_gap_annotations": 1,
        "eeg_gap_origin_first_samp": int(raw.first_samp),
        "eeg_gaps": eeg_gaps,
        "impedance_file": meta.get("impedance_file", ""),
    }})

    events = (list(markers) if markers is not None else
              [ReviewMarker(**event) for event in meta.get("marker_events", [])])
    onsets = [m.onset(sfreq) for m in events]
    durations = [0.0] * len(events)
    descriptions = [m.label or f"code {m.code}" for m in events]
    for sample, length in loss_spans(meta.get("loss_samples", []),
                                    meta.get("loss_lengths"), raw.n_times):
        onsets.append(sample / sfreq)
        durations.append(length / sfreq)
        descriptions.append(ANNOTATION_DESCRIPTION)
    for gap in eeg_gaps:
        onsets.append(gap["start_sample"] / sfreq)
        durations.append(gap["length"] / sfreq)
        descriptions.append(IMPEDANCE_CHECK_ANNOTATION)
    if onsets:
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
        # EDF's integer signals cannot represent NaN. Only explicitly marked
        # missing intervals may be replaced by export placeholders; their BAD
        # duration annotations remain in the file and restore NaNs on import.
        allowed = np.zeros(raw.n_times, dtype=bool)
        for gap in normalise_eeg_gaps(meta.get("eeg_gaps", []), raw.n_times):
            allowed[gap["start_sample"]:gap["start_sample"] + gap["length"]] = True
        if not np.isfinite(raw._data[:, ~allowed]).all():
            raise ExportError("EDF cannot store non-finite EEG samples outside annotated missing intervals.")
        raw._data[:, allowed] = 0.0
        # All Ganglion inputs are voltages. MNE treats MISC as unitless and
        # writes its unscaled SI values under an EDF uV label; marking those
        # inputs as EEG for this export keeps their physical amplitudes intact.
        misc_channels = {name: "eeg" for name, kind in
                         zip(raw.ch_names, raw.get_channel_types()) if kind == "misc"}
        if misc_channels:
            raw.set_channel_types(misc_channels, on_unit_change="ignore")
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

        sfreq = float(meta.get("sampling_rate", 200))
        eeg_channels = list(meta.get("eeg_channels", []))
        data = np.ascontiguousarray(raw_data[eeg_channels, :], dtype=np.float64)
        biosig.save(path, data.T, sfreq)  # type: ignore[attr-defined]
        return path
    except Exception as exc:  # pragma: no cover
        raise ExportError(f"GDF export via BioSig failed: {exc}") from exc
