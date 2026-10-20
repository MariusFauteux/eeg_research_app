"""Explicit missing EEG intervals, separate from short BLE packet losses."""

from __future__ import annotations

import math
import operator


IMPEDANCE_CHECK_ANNOTATION = "BAD_impedance_check"


def _clip(gap: dict, start: int, end: int, offset: int = 0) -> dict:
    out = dict(gap)
    original_start, original_length = gap["start_sample"], gap["length"]
    lo, hi = max(start, original_start), min(end, original_start + original_length)
    out.update(start_sample=lo - offset, length=hi - lo)
    if gap["started_at"] is not None and gap["ended_at"] is not None:
        duration = gap["ended_at"] - gap["started_at"]
        out["started_at"] = gap["started_at"] + (lo - original_start) / original_length * duration
        out["ended_at"] = gap["started_at"] + (hi - original_start) / original_length * duration
    return out


def normalise_eeg_gaps(gaps, n_samples=None, sampling_rate=None) -> list[dict]:
    """Validate explicit gaps and clip them to available recording samples.

    Wall timestamps describe the retained portion of a gap. If a format only
    provides duration annotations, unknown wall timestamps remain ``None``.
    ``sampling_rate`` is accepted for callers sharing sample-clock metadata;
    known wall intervals are clipped proportionally without inventing a clock.
    """
    result = []
    for value in gaps or []:
        if not isinstance(value, dict):
            raise ValueError("EEG gaps must contain interval dictionaries")
        try:
            start = operator.index(value["start_sample"])
            length = operator.index(value["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("EEG gaps need integer start_sample and length") from exc
        if length <= 0 or isinstance(value["start_sample"], bool) or isinstance(value["length"], bool):
            raise ValueError("EEG gap length must be positive")
        reason = value.get("reason", "impedance_check")
        if reason != "impedance_check":
            raise ValueError(f"Unsupported EEG gap reason: {reason}")
        times = []
        for key in ("started_at", "ended_at"):
            timestamp = value.get(key)
            try:
                timestamp = float(timestamp) if timestamp is not None else None
            except (TypeError, ValueError) as exc:
                raise ValueError(f"EEG gap {key} must be a finite timestamp") from exc
            if timestamp is not None and not math.isfinite(timestamp):
                raise ValueError(f"EEG gap {key} must be a finite timestamp")
            times.append(timestamp)
        if times[0] is not None and times[1] is not None and times[1] < times[0]:
            raise ValueError("EEG gap end precedes its start")
        gap = {"start_sample": start, "length": length, "reason": reason,
               "started_at": times[0], "ended_at": times[1]}
        lo, hi = max(0, start), start + length
        if n_samples is not None:
            hi = min(n_samples, hi)
        if lo < hi:
            clipped = _clip(gap, lo, hi)
            if clipped not in result:
                result.append(clipped)
    return sorted(result, key=lambda gap: gap["start_sample"])


def crop_eeg_gaps(gaps, start: int, end: int, sampling_rate=None) -> list[dict]:
    """Intersect missing intervals with [start, end), shifting into the crop."""
    result = []
    for gap in normalise_eeg_gaps(gaps):
        if max(start, gap["start_sample"]) < min(end, gap["start_sample"] + gap["length"]):
            result.append(_clip(gap, start, end, offset=start))
    return result
