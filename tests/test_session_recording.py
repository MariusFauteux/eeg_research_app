"""Live recording paths must not reuse an existing recording directory."""

from datetime import datetime
from unittest.mock import patch

import numpy as np
from brainflow.data_filter import DataFilter

from ganglion_studio.core.session import SessionConfig, SessionRecorder


def test_recordings_started_in_same_second_keep_separate_data(tmp_path):
    config = SessionConfig(name="same session")
    first = SessionRecorder(config, base_dir=str(tmp_path))
    second = SessionRecorder(config, base_dir=str(tmp_path))
    with patch("ganglion_studio.core.session.datetime") as clock:
        clock.now.return_value = datetime(2026, 10, 19, 12, 0, 0)
        first_dir = first.begin()
        second_dir = second.begin()

    assert first_dir != second_dir
    assert first.file_prefix == "20261019_120000_same_session"
    assert second.file_prefix == "20261019_120000_same_session_2"

    # Exercise the real writer, including its overwrite mode: separate paths
    # must preserve the first recording when the second one is saved.
    first_data = np.ones((4, 10), dtype=np.float64)
    second_data = np.full((4, 10), 2.0, dtype=np.float64)
    with patch.object(SessionRecorder, "_try_export_edf", return_value=None):
        first_files = first.save(first_data, {})
        second_files = second.save(second_data, {})
    first_csv = next(p for p in first_files if p.endswith("_raw.csv"))
    second_csv = next(p for p in second_files if p.endswith("_raw.csv"))
    assert np.array_equal(DataFilter.read_file(first_csv), first_data)
    assert np.array_equal(DataFilter.read_file(second_csv), second_data)
