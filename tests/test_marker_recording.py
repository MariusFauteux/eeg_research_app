"""Every inserted event survives the log and export even at one sample."""

import csv

import numpy as np
import pytest

from ganglion_studio.core import exporter
from ganglion_studio.core.recording_loader import load_recording
from ganglion_studio.core.session import MarkerEvent, SessionConfig, SessionRecorder


def _recording(tmp_path, monkeypatch):
    recorder = SessionRecorder(SessionConfig(name="markers"), base_dir=str(tmp_path))
    recorder.begin()
    monkeypatch.setattr(recorder, "_try_export_edf", lambda *a: None)
    raw = np.zeros((6, 20), dtype=float)
    raw[1] = np.arange(20)
    raw[5, 10] = 1  # a compatibility marker row can hold just one of the events
    meta = {"sampling_rate": 200.5, "eeg_channels": [1],
            "channel_names": ["A"], "marker_channel": 5}
    return recorder, raw, meta


def test_sample_log_preserves_collisions_and_individual_labels(tmp_path, monkeypatch):
    recorder, raw, meta = _recording(tmp_path, monkeypatch)
    recorder.add_marker(MarkerEvent(100.0, 1, "First"))
    recorder.add_marker(MarkerEvent(100.0, 2, "Second"))
    recorder.add_marker(MarkerEvent(100.01, 1, "Renamed"))

    aligned = recorder.align_markers([(10, 1, 100.0), (10, 2, 100.0), (12, 1, 100.01)])
    paths = recorder.save(raw, meta)
    loaded = load_recording(next(p for p in paths if p.endswith("_raw.csv")))

    assert [(m.sample, m.code, m.label) for m in loaded.markers] == [
        (10, 1, "First"), (10, 2, "Second"), (12, 1, "Renamed")]
    assert loaded.markers == aligned
    with open(next(p for p in paths if p.endswith("_markers.csv")), newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["sample_index"] for row in rows] == ["10", "10", "12"]


def test_unmapped_events_stay_in_log_without_becoming_phantom_markers(tmp_path, monkeypatch):
    recorder, raw, meta = _recording(tmp_path, monkeypatch)
    recorder.add_marker(MarkerEvent(100.0, 2, "No samples"))
    recorder.align_markers([(None, 2, 100.0)])

    paths = recorder.save(raw, meta)
    loaded = load_recording(next(p for p in paths if p.endswith("_raw.csv")))

    assert loaded.markers == []
    with open(next(p for p in paths if p.endswith("_markers.csv")), newline="") as fh:
        row, = list(csv.DictReader(fh))
    assert row["label"] == "No samples"
    assert row["sample_index"] == ""


def test_fif_export_retains_coincident_events_from_metadata(tmp_path):
    pytest.importorskip("mne")
    meta = {"sampling_rate": 200.5, "eeg_channels": [0], "channel_names": ["A"],
            "marker_events": [{"sample": 10, "code": 1, "label": "First"},
                              {"sample": 10, "code": 2, "label": "Second"}]}
    path = exporter.export(str(tmp_path / "markers.fif"), "fif", np.zeros((1, 100)), meta)
    loaded = load_recording(path)
    assert [(m.sample, m.label) for m in loaded.markers] == [(10, "First"), (10, "Second")]


def test_explicit_empty_marker_list_overrides_original_events():
    pytest.importorskip("mne")
    meta = {"sampling_rate": 200, "eeg_channels": [0], "channel_names": ["A"],
            "marker_events": [{"sample": 10, "code": 1, "label": "Removed"}]}
    raw = exporter.build_raw(np.zeros((1, 100)), meta, markers=[])
    assert len(raw.annotations) == 0


@pytest.mark.parametrize("sidecar", [None, b"\xff\xfe"])
def test_metadata_preserves_events_when_marker_log_missing_or_corrupt(tmp_path, monkeypatch, sidecar):
    recorder, raw, meta = _recording(tmp_path, monkeypatch)
    meta["marker_events"] = [dict(sample=10, code=1, label="First"),
                             dict(sample=10, code=2, label="Second")]
    recorder.save(raw, meta)
    marker_path, = tmp_path.rglob("*_markers.csv")
    if sidecar is None:
        marker_path.unlink()
    else:
        marker_path.write_bytes(sidecar)
    csv_path, = tmp_path.rglob("*_raw.csv")

    loaded = load_recording(str(csv_path))

    assert [(m.sample, m.label) for m in loaded.markers] == [(10, "First"), (10, "Second")]
