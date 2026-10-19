"""Closing a session must preserve samples, including when a save needs retry."""

import json
import os
import time
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from brainflow.data_filter import DataFilter
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core import protocol_config
from ganglion_studio.core.session import SessionConfig, SessionRecorder
from ganglion_studio.ui import session_view
from ganglion_studio.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def view(qt_app, tmp_path, monkeypatch):
    # Load the synthetic descriptor without connecting to a device. Samples are
    # injected into the real recording buffer so these tests need no timing sleeps.
    manager = BoardManager(demo=True)
    manager.board_id = -1
    manager._load_descriptor()
    monkeypatch.setattr(protocol_config, "PATH", str(tmp_path / "protocols.json"))
    monkeypatch.setattr(session_view, "SessionRecorder", lambda config: SessionRecorder(
        config, base_dir=str(tmp_path)))
    monkeypatch.setattr(SessionRecorder, "_try_export_edf", lambda *args: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    widget = session_view.SessionView(manager, SessionConfig(name="lifecycle", demo=True))
    yield widget
    widget._recorder = None
    widget.shutdown()
    manager.release()
    widget.close()
    widget.deleteLater()


def _record_samples(view):
    view.record_btn.setChecked(True)
    raw = np.zeros((view._manager.num_rows, 8))
    raw[view._manager.eeg_channels[0]] = np.arange(8) + 10.0
    with view._manager._lock:
        view._manager._record_chunks.append(raw)
        view._manager._record_count = raw.shape[1]
    return raw


def _assert_saved_samples(tmp_path, raw):
    csv_path, = tmp_path.rglob("*_raw.csv")
    saved = DataFilter.read_file(str(csv_path))
    np.testing.assert_array_equal(saved[1], raw[1])
    meta_path, = tmp_path.rglob("*_meta.json")
    assert json.loads(meta_path.read_text())["n_samples"] == raw.shape[1]


def test_shutdown_saves_active_manual_recording(view, tmp_path):
    raw = _record_samples(view)

    assert view.shutdown() is True

    _assert_saved_samples(tmp_path, raw)
    assert not view._manager.recording
    assert view._recorder is None
    assert not view._reviews  # app teardown must not open another window


def test_declining_exit_keeps_session_and_recording(view, monkeypatch):
    _record_samples(view)
    exited = Mock()
    view.exit_session.connect(exited)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)

    view._on_exit()

    exited.assert_not_called()
    assert view._manager.recording
    assert view._recorder is not None


def test_failed_save_retains_samples_and_blocks_shutdown_until_retry(view, tmp_path, monkeypatch):
    raw = _record_samples(view)
    recorder = view._recorder
    save = recorder.save
    monkeypatch.setattr(recorder, "save", Mock(side_effect=OSError("disk full")))

    view.record_btn.setChecked(False)

    assert view.record_btn.isChecked()
    assert view.record_btn.text() == "Retry save"
    assert not view._manager.recording
    assert not view.protocol_panel.run_btn.isEnabled()
    assert view.shutdown() is False
    np.testing.assert_array_equal(view._pending_recording[0], raw)
    assert view._timer.isActive()

    monkeypatch.setattr(recorder, "save", save)
    assert view.shutdown() is True
    _assert_saved_samples(tmp_path, raw)
    assert view._pending_recording is None
    assert view.protocol_panel.run_btn.isEnabled()


def test_pending_save_cannot_start_protocol_or_append_markers(view, monkeypatch):
    _record_samples(view)
    recorder = view._recorder
    monkeypatch.setattr(recorder, "save", Mock(side_effect=OSError("disk full")))
    view.record_btn.setChecked(False)

    view.protocol_panel._start_run()
    view._on_marker(3, "Stimulus", 123.0)

    assert not view.protocol_panel._running
    assert not view._manager.recording
    assert recorder.markers == []


def test_main_window_rejects_close_when_recording_cannot_save(qt_app):
    manager = Mock()
    window = MainWindow()
    window.manager = manager
    window.session_view = SimpleNamespace(shutdown=Mock(return_value=False),
                                          resume_after_failed_shutdown=Mock())
    window._session_shutdown_ready = True
    event = Mock()

    try:
        MainWindow.closeEvent(window, event)

        event.ignore.assert_called_once()
        manager.release.assert_not_called()
        window.session_view.resume_after_failed_shutdown.assert_called_once()
    finally:
        window.session_view = None
        window.manager = None
        window.close()


def test_close_save_failure_updates_retry_control(view, monkeypatch):
    _record_samples(view)
    monkeypatch.setattr(view._recorder, "save", Mock(side_effect=OSError("disk full")))

    assert view.shutdown() is False

    assert view.record_btn.isChecked()
    assert view.record_btn.text() == "Retry save"


def test_async_close_retains_snapshot_after_save_failure(view, qt_app, tmp_path, monkeypatch):
    raw = _record_samples(view)
    recorder = view._recorder
    save = recorder.save
    monkeypatch.setattr(recorder, "save", Mock(side_effect=OSError("disk full")))
    window = MainWindow()
    window.manager = view._manager
    window.session_view = view
    window.stack.addWidget(view)
    window.stack.setCurrentWidget(view)
    window.show()

    def until(condition):
        deadline = time.monotonic() + 5.0
        while not condition() and time.monotonic() < deadline:
            qt_app.processEvents()
            QTest.qWait(5)
        assert condition()

    try:
        assert window.close() is False
        until(lambda: view._pending_recording is not None and view.isEnabled())

        assert window.isVisible()
        assert window.session_view is view
        assert window.manager is view._manager
        assert not view._manager.recording
        assert view._timer.isActive()
        np.testing.assert_array_equal(view._pending_recording[0], raw)

        monkeypatch.setattr(recorder, "save", save)
        assert window.close() is False
        until(lambda: window._session_stop_worker is None and
              window._release_worker is None and not window.isVisible())
        _assert_saved_samples(tmp_path, raw)
        assert view._pending_recording is None
        assert window.manager is None
    finally:
        monkeypatch.setattr(recorder, "save", save)
        until(lambda: window._session_stop_worker is None and window._release_worker is None)
        window.session_view = None
        window.manager = None
        window.stack.removeWidget(view)
        view.setParent(None)
        window.close()
