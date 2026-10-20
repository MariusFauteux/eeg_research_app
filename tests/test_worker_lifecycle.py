"""Worker shutdown keeps Qt responsive and never releases an in-use board."""

import os
import threading
import time
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QWidget

from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.recording_loader import LoadedRecording
from ganglion_studio.core.session import SessionConfig
from ganglion_studio.ui import main_window as main_ui
from ganglion_studio.ui import processing_window as processing_ui


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _until(qt_app, condition):
    deadline = time.monotonic() + 5.0
    while not condition() and time.monotonic() < deadline:
        qt_app.processEvents()
        QTest.qWait(5)
    assert condition(), "Worker lifecycle did not finish"


def _blocked_acquisition():
    entered = threading.Event()
    proceed = threading.Event()
    calls = []

    class Board:
        def __init__(self):
            self.read_threads = []

        def get_board_data(self):
            self.read_threads.append(threading.get_ident())
            entered.set()
            assert proceed.wait(5.0)
            calls.append("poll returned")
            return np.zeros((1, 0))

        def stop_stream(self):
            calls.append("stop")

        def release_session(self):
            calls.append("release")

    manager = BoardManager()
    manager.board = Board()
    manager.streaming = True
    assert manager.start_acquisition()
    assert entered.wait(2.0)
    return manager, proceed, calls


def test_timed_out_stop_retains_producer_and_release_waits_for_poll():
    manager, proceed, calls = _blocked_acquisition()
    producer = manager._acq_thread
    releasing = threading.Event()
    original_stop = manager.stop_acquisition

    def stop(timeout=2.0):
        if timeout is None:
            releasing.set()
        return original_stop(timeout)

    manager.stop_acquisition = stop
    cleanup = threading.Thread(target=manager.release)
    try:
        assert manager.stop_acquisition(timeout=0.0) is False
        assert manager._acq_thread is producer
        assert not manager.start_acquisition()
        assert manager._acq_stop.is_set()
        cleanup.start()
        assert releasing.wait(2.0)
        assert cleanup.is_alive()
        assert not calls
        assert not manager.start_acquisition()
    finally:
        proceed.set()
        if cleanup.ident is not None:
            cleanup.join(5.0)
        manager.release()

    assert not cleanup.is_alive()
    assert manager._acq_thread is None
    assert manager.board is None
    assert calls == ["poll returned", "stop", "release"]


def test_repeated_start_does_not_start_backend_twice():
    manager = BoardManager()
    board = Mock()
    manager.board = board

    manager.start()
    manager.start()

    board.start_stream.assert_called_once_with(450000)
    manager.release()


def test_main_close_waits_for_acquisition_without_blocking_qt(qt_app):
    manager, proceed, calls = _blocked_acquisition()
    window = main_ui.MainWindow()
    window.manager = manager
    window.show()
    try:
        started = time.monotonic()
        assert window.close() is False
        assert time.monotonic() - started < 0.5
        assert window._release_worker is not None
        assert window.isVisible()
        qt_app.processEvents()
        assert "release" not in calls

        proceed.set()
        _until(qt_app, lambda: window._release_worker is None and not window.isVisible())
        assert calls == ["poll returned", "stop", "release"]
        assert window.manager is None
    finally:
        proceed.set()
        _until(qt_app, lambda: window._release_worker is None)
        window.close()


def test_backend_release_failure_keeps_manager_for_close_retry(qt_app, monkeypatch):
    manager = Mock()
    manager.release.side_effect = [RuntimeError("disconnect failed"), None]
    warnings = []
    monkeypatch.setattr(main_ui.QMessageBox, "warning", lambda *args: warnings.append(args[1:]))
    window = main_ui.MainWindow()
    window.manager = manager
    window.show()
    try:
        assert window.close() is False
        _until(qt_app, lambda: bool(warnings))
        assert window.manager is manager
        assert window.isVisible()
        assert window._release_worker is None

        assert window.close() is False
        _until(qt_app, lambda: window._release_worker is None and not window.isVisible())
        assert manager.release.call_count == 2
        assert window.manager is None
    finally:
        _until(qt_app, lambda: window._release_worker is None)
        window.close()


def test_active_recording_close_finalizes_backend_reads_off_gui(qt_app):
    manager, proceed, calls = _blocked_acquisition()
    board = manager.board
    manager.num_rows = 1
    manager.recording = True
    manager._record_chunks = [np.array([[10.0, 11.0]])]
    manager._record_count = 2
    snapshots = []

    class Session(QWidget):
        def quiesce_for_shutdown(self):
            self.setEnabled(False)

        def shutdown(self, raw_data=None):
            assert threading.get_ident() == gui_thread
            if raw_data is not None:
                assert not manager.recording
                snapshots.append(raw_data.copy())
            return True

    gui_thread = threading.get_ident()
    window = main_ui.MainWindow()
    view = Session()
    window.session_view = view
    window.manager = manager
    window.show()
    try:
        assert window.close() is False
        assert window._session_stop_worker is not None
        assert not view.isEnabled()
        assert snapshots == []
        tick = Mock()
        QTimer.singleShot(0, tick)
        qt_app.processEvents()
        tick.assert_called_once()
        assert "release" not in calls

        proceed.set()
        _until(qt_app, lambda: window._session_stop_worker is None and
               window._release_worker is None and not window.isVisible())

        assert len(snapshots) == 1
        np.testing.assert_array_equal(snapshots[0], [[10.0, 11.0]])
        assert len(board.read_threads) == 2  # blocked producer + final recording drain
        assert all(ident != gui_thread for ident in board.read_threads)
        assert calls == ["poll returned", "poll returned", "stop", "release"]
    finally:
        proceed.set()
        _until(qt_app, lambda: window._session_stop_worker is None and window._release_worker is None)
        window.close()
        view.close()


def test_final_drain_failure_retains_session_for_retry(qt_app, monkeypatch):
    manager, proceed, calls = _blocked_acquisition()
    manager.recording = True
    manager.stop_recording = Mock(side_effect=RuntimeError("drain failed"))
    warnings = []
    monkeypatch.setattr(main_ui.QMessageBox, "warning", lambda *args: warnings.append(args[1:]))

    class Session(QWidget):
        def quiesce_for_shutdown(self):
            self.setEnabled(False)

        def resume_after_failed_shutdown(self):
            self.setEnabled(True)

    window = main_ui.MainWindow()
    view = Session()
    window.session_view = view
    window.manager = manager
    window.show()
    try:
        assert window.close() is False
        proceed.set()
        _until(qt_app, lambda: bool(warnings))

        assert window._session_stop_worker is None
        assert window._release_worker is None
        assert window.manager is manager
        assert window.session_view is view
        assert manager.recording
        assert window.isVisible()
        assert view.isEnabled()
        assert "release" not in calls
        assert warnings == [("Could not finish recording", "drain failed")]
    finally:
        proceed.set()
        _until(qt_app, lambda: window._session_stop_worker is None)
        window.session_view = None
        window.close()
        _until(qt_app, lambda: window._release_worker is None)
        view.close()


def test_close_during_prepare_retains_worker_and_skips_start(qt_app, monkeypatch):
    entered = threading.Event()
    proceed = threading.Event()
    prepared = threading.Event()
    manager = Mock()

    def prepare():
        entered.set()
        assert proceed.wait(5.0)
        prepared.set()

    def release():
        assert prepared.is_set()

    manager.prepare.side_effect = prepare
    manager.release.side_effect = release
    factory = Mock(return_value=manager)
    monkeypatch.setattr(main_ui, "BoardManager", factory)
    monkeypatch.setattr(main_ui, "SessionView", Mock(side_effect=AssertionError("Late session view")))
    window = main_ui.MainWindow()
    window.show()
    try:
        window._on_start_session(SessionConfig(demo=True))
        assert entered.wait(2.0)
        worker = window._worker
        window._on_start_session(SessionConfig(demo=True))
        assert factory.call_count == 1

        started = time.monotonic()
        assert window.close() is False
        assert time.monotonic() - started < 0.5
        assert window._worker is worker
        assert worker.isRunning()
        manager.release.assert_not_called()
        manager.start.assert_not_called()

        proceed.set()
        _until(qt_app, lambda: window._worker is None and
               window._release_worker is None and not window.isVisible())
        manager.start.assert_not_called()
        manager.release.assert_called_once()
        assert window.session_view is None
    finally:
        proceed.set()
        _until(qt_app, lambda: window._worker is None and window._release_worker is None)
        window.close()


def test_processing_close_ignores_late_results_and_main_waits(qt_app, monkeypatch):
    entered = threading.Event()
    proceed = threading.Event()
    pipeline = Mock()

    def apply(eeg, *args):
        entered.set()
        assert proceed.wait(5.0)
        return eeg * 2, ["Late result"]

    pipeline.side_effect = apply
    monkeypatch.setattr(processing_ui.proc, "apply_pipeline", pipeline)
    monkeypatch.setattr(processing_ui.proc, "available_methods",
                        lambda: {"asr": False, "aas": False})
    recording = LoadedRecording(np.ones((2, 200)), 200, ["a", "b"])
    monkeypatch.setattr(processing_ui, "load_recording", lambda path: recording)
    main = main_ui.MainWindow()
    main.show()
    main._open_processing_lab()
    window = main._processing_windows[0]
    try:
        window._load("test")
        assert entered.wait(2.0)
        worker = window._worker
        window._schedule()
        assert window._debounce.isActive()

        assert main.close() is False
        assert main.isVisible()
        assert window._worker is worker
        assert window._closing
        assert not window._debounce.isActive()
        worker.done.emit(np.full((2, 200), 9.0), ["Incorrect late result"])
        worker.failed.emit("Incorrect late failure")
        window._schedule()
        window._recompute()
        assert not window._debounce.isActive()
        np.testing.assert_array_equal(window._processed, recording.eeg)
        assert window.status.text() == "Finishing processing before closing..."

        proceed.set()
        _until(qt_app, lambda: window._worker is None and window._closed and not main.isVisible())
        assert pipeline.call_count == 1
        np.testing.assert_array_equal(window._processed, recording.eeg)
    finally:
        proceed.set()
        _until(qt_app, lambda: window._worker is None)
        window.close()
        main.close()
