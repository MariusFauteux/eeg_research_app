"""Rebuilding a .fif from a session's original _raw.csv.

The point of the feature is recovery: a .fif exported while "Display filter" was
ticked used to contain filtered data, so these tests check the rebuilt file
matches the CSV sample-for-sample (within fif's float32 storage) and carries the
markers over.
"""

import csv
import json
import os

import numpy as np
import pytest

from brainflow.data_filter import DataFilter

from ganglion_studio.core import rebuild_fif
from ganglion_studio.core.exporter import mne_available
from ganglion_studio.core.recording_loader import load_recording

pytestmark = pytest.mark.skipif(not mne_available(), reason="MNE not installed")

SR = 200
EEG_ROWS = [1, 2, 3, 4]
MARKER_ROW = 5


def _write_session(folder, name="sess", n_samples=1000, markers=((100, 1, "Eyes Open"),)):
    """Write a minimal <name>_raw.csv + _meta.json + _markers.csv trio."""
    rng = np.random.default_rng(7)
    data = np.zeros((8, n_samples))
    data[0] = np.arange(n_samples)  # package counter
    for i, row in enumerate(EEG_ROWS):
        data[row] = rng.normal(0.0, 20.0, n_samples) + 10.0 * i  # + DC offset
    for sample, code, _label in markers:
        data[MARKER_ROW, sample] = code

    prefix = os.path.join(str(folder), name)
    DataFilter.write_file(np.ascontiguousarray(data), prefix + "_raw.csv", "w")
    with open(prefix + "_meta.json", "w", encoding="utf-8") as fh:
        json.dump({
            "sampling_rate": SR,
            "eeg_channels": EEG_ROWS,
            "marker_channel": MARKER_ROW,
            "channel_names": ["Fp1", "Fp2", "O1", "O2"],
            "channel_types": ["EEG"] * 4,
            "electrodes": ["Ag/AgCl"] * 4,
            "placements": ["Fp1", "Fp2", "O1", "O2"],
            "channel_active": [True, True, True, False],
        }, fh)
    with open(prefix + "_markers.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "time", "code", "label"])
        for sample, code, label in markers:
            writer.writerow([0.0, sample / SR, code, label])
    return prefix + "_raw.csv", data


def test_rebuilt_fif_matches_the_csv_samples(tmp_path):
    csv_path, data = _write_session(tmp_path)
    result = rebuild_fif.rebuild(csv_path)

    back = load_recording(result.fif_path)
    assert back.eeg.shape == (4, 1000)
    # fif stores single precision: compare in uV with a tolerance scaled to that.
    assert np.allclose(back.eeg, data[EEG_ROWS, :], atol=1e-3)
    assert back.sampling_rate == SR
    assert back.channel_names == ["Fp1", "Fp2", "O1", "O2"]


def test_rebuild_carries_markers_and_reports_them(tmp_path):
    csv_path, _ = _write_session(
        tmp_path, markers=((100, 1, "Eyes Open"), (600, 2, "Eyes Closed")))
    result = rebuild_fif.rebuild(csv_path)

    assert result.n_markers == 2
    assert result.used_meta is True
    assert result.duration_s == pytest.approx(5.0)

    back = load_recording(result.fif_path)
    assert [(m.sample, m.label) for m in back.markers] == [
        (100, "Eyes Open"), (600, "Eyes Closed")]


def test_default_output_sits_next_to_the_csv(tmp_path):
    csv_path, _ = _write_session(tmp_path)
    assert rebuild_fif.default_output_path(csv_path) == os.path.join(
        str(tmp_path), "sess_raw.fif")

    result = rebuild_fif.rebuild(csv_path)
    assert result.fif_path == os.path.join(str(tmp_path), "sess_raw.fif")
    assert os.path.exists(result.fif_path)


def test_sidecar_keeps_channel_info_and_spares_the_source_meta(tmp_path):
    csv_path, _ = _write_session(tmp_path)
    source_meta = os.path.join(str(tmp_path), "sess_meta.json")
    before = open(source_meta, encoding="utf-8").read()

    result = rebuild_fif.rebuild(csv_path)

    # The source recording's own metadata must never be rewritten.
    assert open(source_meta, encoding="utf-8").read() == before
    assert result.sidecar_path == os.path.join(str(tmp_path), "sess_raw_meta.json")
    sidecar = json.load(open(result.sidecar_path, encoding="utf-8"))
    assert sidecar["electrodes"] == ["Ag/AgCl"] * 4
    assert sidecar["placements"] == ["Fp1", "Fp2", "O1", "O2"]
    assert sidecar["channel_active"] == [True, True, True, False]
    assert sidecar["rebuilt_from"] == "sess_raw.csv"
    assert load_recording(result.fif_path).placements == sidecar["placements"]


def test_rebuild_preserves_signal_types_in_fif(tmp_path):
    import mne

    csv_path, data = _write_session(tmp_path)
    meta_path = rebuild_fif.meta_path_for_csv(csv_path)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["channel_types"] = ["EEG", "ECG", "EMG", "MISC"]
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)

    result = rebuild_fif.rebuild(csv_path)

    raw = mne.io.read_raw_fif(result.fif_path, verbose="ERROR")
    assert raw.get_channel_types() == ["eeg", "ecg", "emg", "misc"]
    back = load_recording(result.fif_path)
    assert back.channel_types == meta["channel_types"]
    assert np.allclose(back.eeg, data[EEG_ROWS], atol=1e-3)


def test_sidecar_skipped_when_it_would_overwrite_the_source_meta(tmp_path):
    """A session already named "..._raw" makes the two meta names collide.

    ``sess_raw_raw.csv`` has ``sess_raw_meta.json`` as its metadata, and saving
    the rebuild as ``sess_raw.fif`` would want that very same sidecar name. The
    rebuild must leave the source metadata alone instead.
    """
    csv_path, _ = _write_session(tmp_path, name="sess_raw")
    source_meta = os.path.join(str(tmp_path), "sess_raw_meta.json")
    before = open(source_meta, encoding="utf-8").read()

    result = rebuild_fif.rebuild(csv_path, os.path.join(str(tmp_path), "sess_raw.fif"))

    assert result.sidecar_path is None
    assert open(source_meta, encoding="utf-8").read() == before
    meta = json.load(open(source_meta, encoding="utf-8"))
    assert meta["marker_channel"] == MARKER_ROW  # untouched original
