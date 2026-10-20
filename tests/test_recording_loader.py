"""MNE imports preserve sample positions and recorded signal channels."""

from datetime import datetime, timezone
import json

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from ganglion_studio.core.recording_loader import load_recording


@pytest.mark.parametrize("with_measurement_date", [False, True])
def test_fif_markers_are_relative_to_first_loaded_sample(tmp_path, with_measurement_date):
    raw = mne.io.RawArray(
        np.zeros((1, 1000)), mne.create_info(["EEG"], 200, "eeg"),
        first_samp=1000, verbose="ERROR",
    )
    if with_measurement_date:
        raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations([1.0], [0.0], ["Stimulus"]))
    path = str(tmp_path / "cropped_raw.fif")
    raw.save(path, overwrite=True, verbose="ERROR")

    loaded = load_recording(path)

    assert [(m.sample, m.label) for m in loaded.markers] == [(200, "Stimulus")]


def test_fif_preserves_signal_types_and_excludes_stim_channel(tmp_path):
    names = ["Brain", "Heart", "Muscle", "Other", "Status"]
    types = ["eeg", "ecg", "emg", "misc", "stim"]
    values = np.arange(5 * 200).reshape(5, 200) * 1e-6
    raw = mne.io.RawArray(values, mne.create_info(names, 200, types), verbose="ERROR")
    path = str(tmp_path / "mixed_raw.fif")
    raw.save(path, overwrite=True, verbose="ERROR")

    loaded = load_recording(path)

    assert loaded.channel_names == names[:4]
    assert loaded.channel_types == ["EEG", "ECG", "EMG", "MISC"]
    assert np.allclose(loaded.eeg, values[:4] * 1e6, atol=1e-4)


def test_eeg_only_fif_still_loads_all_channels(tmp_path):
    raw = mne.io.RawArray(
        np.ones((2, 200)) * 1e-6, mne.create_info(["A", "B"], 200, "eeg"),
        verbose="ERROR",
    )
    path = str(tmp_path / "eeg_raw.fif")
    raw.save(path, overwrite=True, verbose="ERROR")

    loaded = load_recording(path)

    assert loaded.channel_names == ["A", "B"]
    assert loaded.channel_types == ["EEG", "EEG"]
    assert np.allclose(loaded.eeg, 1.0)


def test_sidecar_preserves_channel_edits_from_older_fif_exports(tmp_path):
    raw = mne.io.RawArray(
        np.zeros((2, 200)), mne.create_info(["A", "B"], 200, "eeg"),
        verbose="ERROR",
    )
    path = tmp_path / "edited_raw.fif"
    raw.save(str(path), overwrite=True, verbose="ERROR")
    sidecar = {
        "channel_names": ["Occipital", "Heart"],
        "channel_types": ["EEG", "ECG"],
        "placements": ["O1", "None"],
    }
    (tmp_path / "edited_raw_meta.json").write_text(json.dumps(sidecar), encoding="utf-8")

    loaded = load_recording(str(path))

    assert loaded.channel_names == sidecar["channel_names"]
    assert loaded.channel_types == sidecar["channel_types"]
    assert loaded.placements == sidecar["placements"]
