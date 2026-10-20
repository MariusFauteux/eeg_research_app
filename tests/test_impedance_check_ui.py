"""Timed contact checks preserve recording state and never block Qt transitions."""

import json
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.impedance import ImpedanceReading
from ganglion_studio.core.impedance_recording import load_impedance_recording
from ganglion_studio.core.recording_loader import load_recording
from ganglion_studio.core.session import SessionConfig, SessionRecorder
from ganglion_studio.ui import session_view as ui
from ganglion_studio.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def until(app, condition):
    deadline = time.monotonic() + 5
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    assert condition()


class Board:
    def __init__(self, manager):
        self.manager = manager
        self.commands = []
        self.readings = []
        self.fail_resume = False
        self.block_mode = "unused"
        self.entered = threading.Event()
        self.proceed = threading.Event()

    def get_board_data(self):
        return np.zeros((self.manager.num_rows, 0))

    def get_impedance_data(self):
        readings, self.readings = self.readings, []
        return readings

    def switch_acquisition_mode(self, mode):
        self.commands.append(mode)
        if mode == self.block_mode:
            self.entered.set()
            assert self.proceed.wait(5)
        if mode == "eeg" and self.fail_resume:
            raise RuntimeError("resume failed")
        return time.time()

    def config_board(self, command):
        pass

    def insert_marker(self, value):
        self.commands.append(("marker", value))

    def stop_stream(self):
        pass

    def release_session(self):
        pass


@pytest.fixture
def view(app, tmp_path, monkeypatch):
    manager = BoardManager(demo=True)
    manager.board_id = -1
    manager._load_descriptor()
    manager.sampling_rate = 20
    manager._buffer_len = 200
    manager._ring = np.zeros((manager.num_rows, 200))
    manager._loss_ring = np.zeros(200)
    manager.demo = False  # Consume explicit impedance events from our fake board.
    manager.board = Board(manager)
    manager.streaming = True
    monkeypatch.setattr(manager, "start_acquisition", lambda: True)
    monkeypatch.setattr(ui, "SessionRecorder", lambda config: SessionRecorder(
        config, base_dir=str(tmp_path)))
    monkeypatch.setattr(SessionRecorder, "_try_export_edf", lambda *args: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    widget = ui.SessionView(manager, SessionConfig(name="timed", demo=True))
    widget._timer.stop()
    yield widget
    manager.board.proceed.set() if manager.board is not None else None
    until(app, lambda: widget._check_worker is None)
    widget.shutdown()
    manager.release()
    for review in widget._reviews:
        review.close()
    widget.close()


def start_check(app, view):
    view.record_btn.setChecked(True)
    view.impedance_btn.click()
    until(app, lambda: view._check_worker is None)
    assert view._manager.impedance_check_status == "measuring"


def test_duration_auto_resumes_with_four_markers_and_linked_measurements(app, view, tmp_path):
    assert not view.impedance_btn.isEnabled()
    view.record_btn.setChecked(True)
    assert view.impedance_btn.isEnabled()
    view.impedance_duration.setValue(1)
    view.impedance_btn.click()
    until(app, lambda: view._check_worker is None)
    manager = view._manager
    assert manager.recording and not view.record_btn.isEnabled()
    assert view.tabs.currentWidget() is view.impedance
    manager.board.readings = [ImpedanceReading(time.time(), 0, 12000),
                              ImpedanceReading(time.time(), 4, 9000)]
    manager.poll()

    until(app, lambda: not manager.impedance_check_active and view._check_worker is None)
    assert manager.recording and manager.streaming
    assert view.record_btn.isEnabled() and view.impedance_btn.isEnabled()
    assert [m.label for m in view._recorder.markers] == [
        "EEG_STOP", "IMPEDANCE_START", "IMPEDANCE_END", "EEG_RESUME"]
    view.record_btn.setChecked(False)

    raw_path, = tmp_path.rglob("*_raw.csv")
    loaded = load_recording(str(raw_path))
    assert loaded.eeg_gaps[0]["length"] >= 18
    assert np.isnan(loaded.eeg).any()
    assert len(loaded.markers) == 4
    readings = load_impedance_recording(loaded.impedance_path)
    assert [r.ohms for r in readings.readings] == [12000, 9000]
    assert readings.channel_names[-1] == "REF"
    review = view._reviews[-1]
    assert os.path.isabs(review._meta["impedance_file"])
    review._open_impedance()
    assert len(review._impedance_windows[-1].recording.readings) == 2
    review._impedance_windows[-1].close()
    assert not view.impedance_btn.isEnabled()


def test_early_resume_and_protocol_stop_defer_save_until_eeg_restored(app, view, tmp_path):
    start_check(app, view)
    QTest.qWait(60)
    view._protocol_started_recording = True
    view._on_protocol_stopped()
    assert view._manager.recording and view._save_after_check
    assert not list(tmp_path.rglob("*_raw.csv"))

    view.impedance_btn.click()
    until(app, lambda: view._check_worker is None and view._recorder is None)

    assert view._manager.streaming and not view._manager.recording
    path, = [p for p in tmp_path.rglob("*_meta.json")
             if not p.name.endswith("_impedance_meta.json")]
    meta = json.loads(path.read_text())
    assert [m["label"] for m in meta["marker_events"]] == [
        "EEG_STOP", "IMPEDANCE_START", "IMPEDANCE_END", "EEG_RESUME"]


def test_failed_resume_offers_retry_and_does_not_claim_eeg_resumed(app, view):
    start_check(app, view)
    view._manager.board.fail_resume = True
    view.impedance_btn.click()
    until(app, lambda: view._check_worker is None)
    assert view.impedance_btn.text() == "Retry EEG resume"
    assert view.impedance_btn.isEnabled()
    assert not view.record_btn.isEnabled()
    assert 93 not in [m.code for m in view._recorder.markers]

    view._manager.board.fail_resume = False
    view.impedance_btn.click()
    until(app, lambda: view._check_worker is None)
    assert view._manager.streaming and not view._manager.impedance_check_active
    assert [m.code for m in view._recorder.markers].count(93) == 1


@pytest.mark.parametrize("phase", ["begin", "measuring", "end"])
def test_close_waits_for_mode_switch_then_saves_without_racing_board(app, view, tmp_path, phase):
    manager = view._manager
    board = manager.board
    window = MainWindow()
    window.manager = manager
    window.session_view = view
    window.stack.addWidget(view)
    window.show()
    try:
        view.record_btn.setChecked(True)
        if phase == "begin":
            board.block_mode = "impedance"
        view.impedance_btn.click()
        if phase == "begin":
            until(app, board.entered.is_set)
            QTest.qWait(60)
        else:
            until(app, lambda: view._check_worker is None)
            QTest.qWait(60)
            if phase == "end":
                board.block_mode = "eeg"
                view.impedance_btn.click()
                until(app, board.entered.is_set)

        started = time.monotonic()
        assert window.close() is False
        assert time.monotonic() - started < .5
        if phase != "measuring":
            assert window._waiting_transition
            assert window._session_stop_worker is None
            # A protocol/manual event arriving during transition stays responsive.
            marker_time = time.time()
            view._on_marker(3, "Stimulus", marker_time)
            assert view._deferred_markers == [(3, marker_time)]
        ticked = []
        QTimer.singleShot(0, lambda: ticked.append(True))
        app.processEvents()
        assert ticked
        board.proceed.set()
        until(app, lambda: window.manager is None and not window.isVisible())

        path, = [p for p in tmp_path.rglob("*_meta.json")
                 if not p.name.endswith("_impedance_meta.json")]
        meta = json.loads(path.read_text())
        codes = [m["code"] for m in meta["marker_events"]]
        assert 90 in codes and 91 in codes and 92 in codes
        assert (93 in codes) == (phase == "end")
        assert meta["eeg_gaps"]
    finally:
        board.proceed.set()
        until(app, lambda: view._check_worker is None and window._session_stop_worker is None
              and window._release_worker is None)
        window.close()
        window.stack.removeWidget(view)
        view.setParent(None)
