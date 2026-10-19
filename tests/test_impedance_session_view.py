"""Impedance capture renders measured times and retains unsaved event records."""

from datetime import datetime, timezone
import os
import time
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

from ganglion_studio.core.impedance import ImpedanceReading
from ganglion_studio.core.impedance_recording import ImpedanceRecorder, load_impedance_recording
from ganglion_studio.core.session import SessionConfig
from ganglion_studio.ui import impedance_session_view as ui


class FakeManager:
    def __init__(self):
        self.now = time.time()
        self.board_id = -1
        self.acquisition_mode = "impedance"
        self.impedance_channel_names = ["A", "B", "C", "D", "REF"]
        self.impedance_raw_unit_basis = "kohms"
        self.channel_types = ["EEG"] * 4
        self.electrodes = ["Ag/AgCl (wet)"] * 4
        self.placements = ["None"] * 4
        self.history = []
        self.recorded = []
        self.recording = False
        self.starts = 0
        self.stops = 0
        self.recording_stops = 0
        self.disconnected = False

    def start_acquisition(self):
        self.starts += 1

    def stop_acquisition(self, timeout=0):
        self.stops += 1

    def push(self, timestamp, channel, ohms):
        reading = ImpedanceReading(timestamp, channel, ohms)
        self.history.append(reading)
        if self.recording:
            self.recorded.append(reading)
        return reading

    def impedance_history(self, seconds=None):
        return [r for r in self.history if seconds is None or r.timestamp >= self.now - seconds]

    def latest_impedance_readings(self):
        latest = {reading.channel: reading for reading in self.history}
        return [latest[channel] for channel in sorted(latest)]

    def is_disconnected(self):
        return self.disconnected

    def start_recording(self):
        self.recording = True
        self.recorded = []

    def stop_recording(self):
        self.recording_stops += 1
        self.recording = False
        return list(self.recorded)

    def recorded_sample_count(self):
        return len(self.recorded)

    def set_channel_config(self, names, types, electrodes, placements):
        self.impedance_channel_names = list(names) + ["REF"]
        self.channel_types = list(types)
        self.electrodes = list(electrodes)
        self.placements = list(placements)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def view(qt_app, tmp_path, monkeypatch):
    manager = FakeManager()
    monkeypatch.setattr(ui.time, "time", lambda: manager.now)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    widget = ui.ImpedanceSessionView(manager, SessionConfig(
        name="impedance", demo=True, acquisition_mode="impedance"), base_dir=str(tmp_path))
    widget._show_review = Mock()
    yield widget
    widget._recorder = None
    widget.shutdown()
    widget.close()


def test_refresh_uses_only_received_measurements_at_their_original_times(view):
    manager = view._manager
    manager.push(manager.now - 5, 0, 12000)
    manager.push(manager.now - 3, 4, 7000)
    manager.push(manager.now - 1, 0, 12000)  # equal value, separate real update

    for _ in range(4):
        view._tick()

    x, y = view.time_series.curves[0].getData()
    np.testing.assert_allclose(x, [0, 4])
    np.testing.assert_allclose(y, [12, 12])
    ref_x, ref_y = view.time_series.curves[4].getData()
    np.testing.assert_allclose(ref_x, [2])
    np.testing.assert_allclose(ref_y, [7])
    assert len(manager.history) == 3
    assert view.current_table.item(1, 1).text() == "—"
    assert view.current_table.item(1, 3).text() == "Waiting"


def test_current_table_distinguishes_stale_missing_and_current(view):
    manager = view._manager
    manager.push(manager.now - 11, 1, 15000)
    manager.push(manager.now - .5, 0, 8000)

    view._tick()

    assert view.current_table.item(0, 1).text() == "8.00"
    assert view.current_table.item(0, 2).text() == "0.5 s ago"
    assert view.current_table.item(0, 3).text() == "Current"
    assert view.current_table.item(1, 1).text() == "15.00"
    assert view.current_table.item(1, 3).text() == "Stale"
    assert view.current_table.item(2, 3).text() == "Waiting"

    manager.disconnected = True
    view._tick()
    assert view.current_table.item(0, 3).text() == "Disconnected"
    assert "Disconnected" in view.status_label.text()


def test_recording_continues_with_rendering_stopped_and_all_traces_hidden(view):
    manager = view._manager
    view.record_btn.setChecked(True)
    view._timer.stop()
    for checkbox in view.time_series.channel_checks:
        checkbox.setChecked(False)
    manager.push(manager.now + .1, 0, 10000)
    manager.push(manager.now + 1.2, 4, 8000)
    manager.push(manager.now + 2.3, 0, 11000)

    assert manager.recorded_sample_count() == 3
    view.record_btn.setChecked(False)

    loaded = load_impedance_recording(view._last_saved_paths[0])
    assert loaded.readings == manager.recorded
    assert len(loaded.readings) == 3
    assert not manager.recording
    assert manager.recording_stops == 1
    assert loaded.metadata["extra"]["impedance_raw_unit_basis"] == "kohms"
    assert loaded.metadata["extra"]["electrodes"] == manager.electrodes
    assert loaded.metadata["extra"]["placements"] == manager.placements
    assert loaded.metadata["extra"]["channel_types"] == manager.channel_types
    view._show_review.assert_called_once_with(view._last_saved_paths[0])


def test_failed_save_freezes_recording_and_retry_keeps_the_same_measurements(view, monkeypatch):
    manager = view._manager
    view.record_btn.setChecked(True)
    first = manager.push(manager.now + .1, 0, 10000)
    recorder = view._recorder
    save = recorder.save
    monkeypatch.setattr(recorder, "save", Mock(side_effect=OSError("disk full")))

    view.record_btn.setChecked(False)

    assert not manager.recording
    assert view.record_btn.isChecked()
    assert view.record_btn.text() == "Retry save"
    assert view._pending_recording == [first]
    assert not view.setup_btn.isEnabled()
    manager.push(manager.now + .2, 4, 5000)
    view._tick()
    assert view.shutdown() is False
    assert view._timer.isActive()
    assert manager.recording_stops == 1

    monkeypatch.setattr(recorder, "save", save)
    view.record_btn.setChecked(False)

    assert load_impedance_recording(view._last_saved_paths[0]).readings == [first]
    assert view._pending_recording is None
    assert view._recorder is None
    assert not view.record_btn.isChecked()
    assert manager.recording_stops == 1
    assert view.setup_btn.isEnabled()


def test_channel_setup_relabels_curves_and_freezes_the_map_while_recording(view, monkeypatch):
    class Dialog:
        DialogCode = ui.ChannelSetupDialog.DialogCode

        def __init__(self, *args):
            pass

        def exec(self):
            return self.DialogCode.Accepted

        def names(self):
            return ["Fp1", "Fp2", "O1", "O2"]

        def types(self):
            return ["EEG"] * 4

        def electrodes(self):
            return ["PEDOT"] * 4

        def placements(self):
            return self.names()

    monkeypatch.setattr(ui, "ChannelSetupDialog", Dialog)
    view._open_channel_setup()

    assert view._channel_names == ["Fp1", "Fp2", "O1", "O2", "REF"]
    assert view.current_table.item(0, 0).text() == "Fp1"
    assert view.time_series.channel_checks[0].text() == "Fp1"
    assert view.time_series.legend.items[0][1].text == "Fp1"

    view.record_btn.setChecked(True)
    monkeypatch.setattr(ui, "ChannelSetupDialog", Mock(side_effect=AssertionError("map is frozen")))
    assert not view.setup_btn.isEnabled()
    view._open_channel_setup()
    view.record_btn.setChecked(False)
    assert view.setup_btn.isEnabled()


def test_shutdown_saves_the_final_worker_snapshot_and_opens_no_review(view, monkeypatch):
    manager = view._manager
    view.record_btn.setChecked(True)
    final = [ImpedanceReading(manager.now + .1, 4, 9000)]
    manager.recording = False  # the MainWindow stop worker already drained it
    monkeypatch.setattr(manager, "stop_recording", Mock(side_effect=AssertionError("already drained")))
    view.quiesce_for_shutdown()

    assert view.shutdown(raw_data=final) is True

    assert load_impedance_recording(view._last_saved_paths[0]).readings == final
    assert not view._timer.isActive()
    view._show_review.assert_not_called()


def test_failed_shutdown_can_resume_for_save_retry(view, monkeypatch):
    view.record_btn.setChecked(True)
    reading = ImpedanceReading(view._manager.now, 0, 12000)
    monkeypatch.setattr(view._recorder, "save", Mock(side_effect=OSError("disk full")))
    view.quiesce_for_shutdown()

    assert view.shutdown(raw_data=[reading]) is False
    view.resume_after_failed_shutdown()

    assert view.isEnabled()
    assert view._timer.isActive()
    assert view._manager.starts == 2
    assert view._pending_recording == [reading]


def test_start_failure_does_not_begin_recording(view, monkeypatch):
    monkeypatch.setattr(ImpedanceRecorder, "begin", Mock(side_effect=OSError("read only folder")))

    view.record_btn.setChecked(True)

    assert view._recorder is None
    assert not view._manager.recording
    assert not view.record_btn.isChecked()


def test_stop_failure_keeps_recording_controls_available_for_retry(view, monkeypatch):
    view.record_btn.setChecked(True)
    recorder = view._recorder
    stop = view._manager.stop_recording
    monkeypatch.setattr(view._manager, "stop_recording", Mock(side_effect=OSError("read failed")))

    view.record_btn.setChecked(False)

    assert view._recorder is recorder
    assert view._manager.recording
    assert view.record_btn.isChecked()
    assert view._pending_recording is None
    assert not view.setup_btn.isEnabled()
    monkeypatch.setattr(view._manager, "stop_recording", stop)
    view.record_btn.setChecked(False)
    assert view._recorder is None


def test_saved_csv_opens_with_actual_times_reference_channel_and_summary(qt_app, tmp_path):
    recorder = ImpedanceRecorder(SessionConfig(name="review"), base_dir=str(tmp_path))
    recorder.begin()
    recorder.started_at = datetime.fromtimestamp(1000, tz=timezone.utc)
    readings = [ImpedanceReading(1001, 0, 10000), ImpedanceReading(1002.5, 0, 20000),
                ImpedanceReading(1003, 4, 5000)]
    csv_path, _ = recorder.save(readings, ["A", "B", "C", "D", "REF"])

    window = ui.ImpedanceRecordingWindow(csv_path)
    try:
        x, y = window.time_series.curves[0].getData()
        np.testing.assert_allclose(x, [1, 2.5])
        np.testing.assert_allclose(y, [10, 20])
        assert window.summary_table.item(0, 1).text() == "2"
        assert window.summary_table.item(0, 3).text() == "15.00"
        assert window.summary_table.item(4, 0).text() == "REF"
        assert window.summary_table.item(4, 1).text() == "1"
        window.time_series.channel_checks[4].setChecked(False)
        ref_x, _ = window.time_series.curves[4].getData()
        assert ref_x is None or not len(ref_x)
    finally:
        window.close()
