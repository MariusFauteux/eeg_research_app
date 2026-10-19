"""Packet-loss provenance and exact clocks survive native export and cropping."""

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from brainflow.data_filter import DataFilter

mne = pytest.importorskip("mne")

from ganglion_studio.core import exporter, rebuild_fif
from ganglion_studio.core.exporter import ReviewMarker
from ganglion_studio.core.packet_loss import crop_loss_spans
from ganglion_studio.core.recording_loader import load_recording


def _data(n=500):
    return np.vstack([np.arange(n, dtype=float), np.arange(n, dtype=float) * 2])


def _meta():
    return {"sampling_rate": 200.5, "eeg_channels": [0, 1],
            "channel_names": ["A", "B"], "loss_samples": [0, 50, 499],
            "loss_lengths": [2, 2, 1]}


def test_fif_preserves_fractional_clock_markers_and_loss_intervals(tmp_path):
    path = exporter.export(str(tmp_path / "recording.fif"), "fif", _data(), _meta(),
                           [ReviewMarker(78, 3, "Stimulus")])

    loaded = load_recording(path)

    assert loaded.sampling_rate == 200.5
    assert loaded.n_samples == 500
    assert [(m.sample, m.label) for m in loaded.markers] == [(78, "Stimulus")]
    assert loaded.loss_samples == [0, 50, 499]
    assert loaded.loss_lengths == [2, 2, 1]
    np.testing.assert_allclose(loaded.eeg, _data(), atol=1e-4)


def test_external_mne_crop_preserves_partial_packet_loss(tmp_path):
    meta = _meta() | {"loss_samples": [99, 150, 199], "loss_lengths": [2, 2, 2]}
    raw = exporter.build_raw(_data(), meta)
    raw.crop(tmin=100 / 200.5, tmax=199 / 200.5)
    path = str(tmp_path / "cropped_raw.fif")
    raw.save(path, overwrite=True, verbose="ERROR")

    loaded = load_recording(path)

    assert loaded.n_samples == 100
    assert loaded.loss_samples == [0, 50, 99]
    assert loaded.loss_lengths == [1, 2, 1]
    assert loaded.markers == []


def test_crop_loss_intervals_clip_and_shift_both_edges():
    assert crop_loss_spans([50, 99, 150, 199, 200], None, 100, 200) == [
        (0, 1), (50, 2), (99, 1)]


def test_csv_rebuild_preserves_clock_and_loss_without_companion_files(tmp_path):
    csv_path = tmp_path / "session_raw.csv"
    data = np.vstack([_data(500), np.zeros(500)])
    DataFilter.write_file(data, str(csv_path), "w")
    meta = _meta() | {"marker_channel": 2}
    (tmp_path / "session_meta.json").write_text(json.dumps(meta))
    (tmp_path / "session_packet_loss.csv").write_text(
        "sample_index,time_s\n0,0\n50,0.2493766\n499,2.4887781\n")

    result = rebuild_fif.rebuild(str(csv_path), str(tmp_path / "separate.fif"))
    # The native file must carry provenance even when copied without sidecars.
    standalone = tmp_path / "standalone_raw.fif"
    standalone.write_bytes(Path(result.fif_path).read_bytes())
    loaded = load_recording(str(standalone))

    assert result.sampling_rate == loaded.sampling_rate == 200.5
    assert loaded.loss_samples == [0, 50, 499]
    assert loaded.loss_lengths == [2, 2, 1]


def test_editor_crop_saves_partial_losses_and_exact_rate(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
    from ganglion_studio.ui.fif_editor_window import RecordingEditorWindow

    app = QApplication.instance() or QApplication([])
    meta = _meta() | {"loss_samples": [99, 150, 199]}
    meta.pop("loss_lengths")
    path = exporter.export(str(tmp_path / "input.fif"), "fif", _data(), meta)
    window = RecordingEditorWindow(path)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(tmp_path / "edited.fif"), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *a: None)
    try:
        assert window._sr == 200.5
        window.keep_from.setValue(100 / 200.5)
        window.keep_to.setValue(200 / 200.5)
        window._on_save()
        loaded = load_recording(str(tmp_path / "edited_raw.fif"))
        assert loaded.sampling_rate == 200.5
        assert loaded.n_samples == 100
        assert loaded.loss_samples == [0, 50, 99]
        assert loaded.loss_lengths == [1, 2, 1]
    finally:
        window.close()
    assert app is not None


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_export_rejects_invalid_sampling_rate(rate):
    with pytest.raises(exporter.ExportError, match="Sampling rate"):
        exporter.build_raw(_data(), _meta() | {"sampling_rate": rate})


def test_old_fif_without_loss_annotations_still_loads(tmp_path):
    path = exporter.export(str(tmp_path / "old.fif"), "fif", _data(),
                           {k: v for k, v in _meta().items() if not k.startswith("loss_")})
    loaded = load_recording(path)
    assert loaded.loss_samples == loaded.loss_lengths == []


def test_external_crop_does_not_resurrect_stale_loss_sidecar(tmp_path):
    raw = exporter.build_raw(_data(), _meta() | {"loss_samples": [10], "loss_lengths": [2]})
    raw.crop(tmin=100 / 200.5, tmax=199 / 200.5)
    assert len(raw.annotations) == 0
    path = str(tmp_path / "cropped_raw.fif")
    raw.save(path, overwrite=True, verbose="ERROR")
    (tmp_path / "cropped_raw_packet_loss.csv").write_text("sample_index,time_s\n10,0.05\n")

    loaded = load_recording(path)

    assert loaded.loss_samples == loaded.loss_lengths == []


def test_nonbinary_fractional_rate_preserves_native_format_precision(tmp_path):
    path = exporter.export(str(tmp_path / "fractional.fif"), "fif", _data(),
                           _meta() | {"sampling_rate": 199.8})
    loaded = load_recording(path)
    assert loaded.sampling_rate == pytest.approx(199.8, rel=5e-8)
    assert loaded.sampling_rate != round(loaded.sampling_rate)
