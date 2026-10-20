"""Timed impedance checks remain explicit missing EEG through save and crop."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from brainflow.data_filter import DataFilter

from ganglion_studio.core import exporter, rebuild_fif
from ganglion_studio.core.eeg_gaps import (
    IMPEDANCE_CHECK_ANNOTATION, crop_eeg_gaps, normalise_eeg_gaps,
)
from ganglion_studio.core.impedance import ImpedanceReading
from ganglion_studio.core.impedance_recording import load_impedance_recording
from ganglion_studio.core.recording_loader import load_recording
from ganglion_studio.core.session import MarkerEvent, SessionConfig, SessionRecorder


GAP = {"start_sample": 100, "length": 100, "reason": "impedance_check",
       "started_at": 1001.0, "ended_at": 1002.0}


@pytest.fixture
def session(tmp_path, monkeypatch):
    recorder = SessionRecorder(SessionConfig(name="check", impedance_units="ohms"), str(tmp_path))
    recorder.begin()
    recorder.started_at = datetime.fromtimestamp(1000, timezone.utc)
    raw = np.zeros((8, 400))
    raw[1:5] = np.sin(np.arange(400) / 10.0)[None, :] * np.arange(1, 5)[:, None] * 10
    raw[1:5, 100:200] = np.nan
    raw[6] = 1000 + np.arange(400) / 100
    meta = {"sampling_rate": 100.0, "eeg_channels": [1, 2, 3, 4],
            "channel_names": ["Fp1", "Fp2", "O1", "O2"],
            "timestamp_channel": 6, "marker_channel": 7, "eeg_gaps": [GAP],
            "loss_samples": [300], "loss_lengths": [2]}
    readings = [ImpedanceReading(1001.25, 0, 1234.5), ImpedanceReading(1001.75, 4, 18000)]
    recorder.markers = [MarkerEvent(1001, 11, "EEG stopped", 100),
                        MarkerEvent(1002, 12, "EEG resumed", 200)]
    monkeypatch.setattr(recorder, "_try_export_edf", lambda *args: None)
    return recorder, raw, meta, readings


def _save(session):
    recorder, raw, meta, readings = session
    return recorder.save(raw, meta, loss_samples=[300], impedance_readings=readings)


def test_session_csv_links_real_impedance_updates_and_preserves_missing_eeg(session):
    paths = _save(session)
    recorder, raw, meta, readings = session
    csv_path = next(path for path in paths if path.endswith("_raw.csv"))
    impedance_path = next(path for path in paths if path.endswith("_impedance.csv"))

    eeg = load_recording(csv_path)
    impedance = load_impedance_recording(impedance_path)

    assert eeg.eeg_gaps == [GAP]
    assert eeg.loss_samples == [300]  # missing EEG is never reported as BLE loss
    assert np.isnan(eeg.eeg[:, 100:200]).all()
    np.testing.assert_allclose(eeg.eeg[:, :100], raw[1:5, :100], atol=1e-6)
    assert eeg.n_samples == 400
    assert eeg.impedance_path == impedance_path
    assert [(marker.sample, marker.label) for marker in eeg.markers] == [
        (100, "EEG stopped"), (200, "EEG resumed")]
    assert impedance.readings == readings
    assert impedance.started_at == 1000
    assert impedance.metadata["payload_units"] == "ohms"
    assert impedance.metadata["extra"]["eeg_file"] == Path(csv_path).name
    assert impedance.metadata["extra"]["eeg_gaps"] == [GAP]
    assert impedance.channel_names == ["Fp1", "Fp2", "O1", "O2", "REF"]
    stored = DataFilter.read_file(csv_path)
    assert np.isnan(stored[1:5, 100:200]).all()
    np.testing.assert_allclose(stored[6], 1000 + np.arange(400) / 100, atol=1e-6)


def test_no_impedance_updates_still_writes_linked_empty_check(session):
    recorder, raw, meta, readings = session
    paths = recorder.save(raw, meta, impedance_readings=[])
    impedance_path = next(path for path in paths if path.endswith("_impedance.csv"))
    assert load_impedance_recording(impedance_path).readings == []
    assert load_recording(next(path for path in paths if path.endswith("_raw.csv"))).eeg_gaps == [GAP]


def test_retry_after_eeg_write_failure_retains_already_written_impedance(session, monkeypatch):
    recorder, raw, meta, readings = session
    original_write = DataFilter.write_file
    writes = []

    def fail_once(data, path, mode):
        writes.append(path)
        if len(writes) == 1:
            raise OSError("disk full")
        return original_write(data, path, mode)

    monkeypatch.setattr(DataFilter, "write_file", fail_once)
    with pytest.raises(OSError, match="disk full"):
        _save(session)
    impedance_path = Path(recorder._path("_impedance.csv"))
    before = impedance_path.read_bytes()

    paths = _save(session)

    assert impedance_path.read_bytes() == before
    assert load_impedance_recording(str(impedance_path)).readings == readings
    assert load_recording(next(path for path in paths if path.endswith("_raw.csv"))).eeg_gaps == [GAP]


def test_gap_crop_clips_both_sample_and_wall_intervals():
    assert crop_eeg_gaps([GAP], 125, 175, 100) == [{
        "start_sample": 0, "length": 50, "reason": "impedance_check",
        "started_at": 1001.25, "ended_at": 1001.75}]
    assert crop_eeg_gaps([GAP], 200, 300) == []
    with pytest.raises(ValueError, match="positive"):
        normalise_eeg_gaps([GAP | {"length": 0}])


def test_csv_to_standalone_fif_preserves_gap_annotations_and_link(session, tmp_path):
    mne = pytest.importorskip("mne")
    paths = _save(session)
    csv_path = next(path for path in paths if path.endswith("_raw.csv"))
    result = rebuild_fif.rebuild(csv_path)
    standalone = tmp_path / "standalone_raw.fif"
    standalone.write_bytes(Path(result.fif_path).read_bytes())

    loaded = load_recording(str(standalone))
    raw = mne.io.read_raw_fif(str(standalone), verbose="ERROR")

    assert loaded.eeg_gaps == [GAP]
    assert np.isnan(loaded.eeg[:, 100:200]).all()
    assert loaded.n_samples == 400
    assert loaded.impedance_path == next(path for path in paths if path.endswith("_impedance.csv"))
    bad = [ann for ann in raw.annotations if ann["description"] == IMPEDANCE_CHECK_ANNOTATION]
    assert len(bad) == 1
    assert bad[0]["duration"] == pytest.approx(1.0)
    assert len(loaded.markers) == 2


def test_external_crop_preserves_partial_gap_and_wall_clock(session, tmp_path):
    pytest.importorskip("mne")
    recorder, data, meta, readings = session
    raw = exporter.build_raw(data, meta)
    raw.crop(tmin=1.5, tmax=2.49)
    path = str(tmp_path / "cropped_raw.fif")
    raw.save(path, overwrite=True, verbose="ERROR")

    loaded = load_recording(path)

    assert loaded.n_samples == 100
    assert loaded.eeg_gaps == [{"start_sample": 0, "length": 50,
                                "reason": "impedance_check",
                                "started_at": 1001.5, "ended_at": 1002.0}]
    assert np.isnan(loaded.eeg[:, :50]).all()
    assert np.isfinite(loaded.eeg[:, 50:]).all()


def test_external_crop_cannot_revive_gap_from_stale_metadata(session, tmp_path):
    pytest.importorskip("mne")
    recorder, data, meta, readings = session
    raw = exporter.build_raw(data, meta)
    raw.crop(tmin=2.5, tmax=2.99)
    path = str(tmp_path / "clean_raw.fif")
    raw.save(path, overwrite=True, verbose="ERROR")
    (tmp_path / "clean_raw_meta.json").write_text(json.dumps({
        "eeg_gaps": [GAP | {"start_sample": 10, "length": 20}]}))

    loaded = load_recording(path)

    assert loaded.eeg_gaps == []
    assert np.isfinite(loaded.eeg).all()


def test_edf_uses_bad_annotated_placeholders_and_restores_nan_on_load(session, tmp_path):
    pytest.importorskip("mne")
    pytest.importorskip("edfio")
    recorder, data, meta, readings = session
    path = exporter.export(str(tmp_path / "check.edf"), "edf", data, meta)

    loaded = load_recording(path)

    assert [(gap["start_sample"], gap["length"]) for gap in loaded.eeg_gaps] == [(100, 100)]
    assert np.isnan(loaded.eeg[:, 100:200]).all()
    np.testing.assert_allclose(loaded.eeg[:, :100], data[1:5, :100], atol=0.002)
    assert loaded.loss_samples == [300]


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_edf_rejects_nonfinite_data_outside_marked_gaps(session, tmp_path, value):
    pytest.importorskip("mne")
    pytest.importorskip("edfio")
    recorder, data, meta, readings = session
    data[1, 5] = value

    with pytest.raises(exporter.ExportError, match="outside annotated"):
        exporter.export(str(tmp_path / "invalid.edf"), "edf", data, meta)


def test_editor_crop_preserves_gap_and_link(session, tmp_path, monkeypatch):
    pytest.importorskip("mne")
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
    from ganglion_studio.ui.fif_editor_window import RecordingEditorWindow

    app = QApplication.instance() or QApplication([])
    paths = _save(session)
    csv_path = next(path for path in paths if path.endswith("_raw.csv"))
    result = rebuild_fif.rebuild(csv_path)
    window = RecordingEditorWindow(result.fif_path)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(tmp_path / "edited.fif"), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    try:
        window.keep_from.setValue(1.5)
        window.keep_to.setValue(2.5)
        window._on_save()
        loaded = load_recording(str(tmp_path / "edited_raw.fif"))
        assert loaded.eeg_gaps == [{"start_sample": 0, "length": 50,
                                    "reason": "impedance_check",
                                    "started_at": 1001.5, "ended_at": 1002.0}]
        assert np.isnan(loaded.eeg[:, :50]).all()
        assert loaded.impedance_path == next(path for path in paths if path.endswith("_impedance.csv"))
    finally:
        window.close()
    assert app is not None
