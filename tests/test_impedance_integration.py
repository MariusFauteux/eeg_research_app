"""Dashboard selection and a complete demo capture use the impedance-only path."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from ganglion_studio.core.impedance_recording import load_impedance_recording
from ganglion_studio.core.session import SessionConfig
from ganglion_studio.ui.dashboard import Dashboard
from ganglion_studio.ui.impedance_session_view import ImpedanceSessionView
from ganglion_studio.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _until(app, condition):
    deadline = time.monotonic() + 5.0
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    assert condition(), "Impedance session did not finish the requested transition"


def test_dashboard_selects_mode_and_explicit_firmware_units(qt_app):
    dashboard = Dashboard()
    configs = []
    dashboard.start_session.connect(configs.append)
    try:
        dashboard.demo_check.setChecked(True)
        dashboard.acquisition_combo.setCurrentIndex(1)
        dashboard.start_btn.click()
        assert configs[-1].acquisition_mode == "impedance"
        assert configs[-1].impedance_units == "kohms"
        assert not dashboard.notch_combo.isEnabled()
        assert not dashboard.decode_combo.isEnabled()

        dashboard.demo_check.setChecked(False)
        dashboard.mac_edit.setText("test-device")
        dashboard.fw_combo.setCurrentIndex(1)
        dashboard._on_start()
        assert configs[-1].impedance_units == "ohms"
        dashboard.acquisition_combo.setCurrentIndex(0)
        assert dashboard.notch_combo.isEnabled()
        assert dashboard.decode_combo.isEnabled()
    finally:
        dashboard.close()


@pytest.mark.parametrize("stop_by_closing", [False, True])
def test_main_window_records_and_reopens_impedance_without_eeg(qt_app, tmp_path, monkeypatch,
                                                              stop_by_closing):
    monkeypatch.chdir(tmp_path)
    window = MainWindow()
    window.show()
    try:
        window._on_start_session(SessionConfig(name="impedance-demo", demo=True,
                                               acquisition_mode="impedance"))
        _until(qt_app, lambda: window.session_view is not None)
        view = window.session_view
        manager = window.manager
        assert isinstance(view, ImpedanceSessionView)
        assert window.manager.acquisition_mode == "impedance"
        assert window.manager.impedance_mode
        assert window.manager._filled == 0

        view.record_btn.setChecked(True)
        QTest.qWait(1100)
        if stop_by_closing:
            window.close()
            _until(qt_app, lambda: window.manager is None and not window.isVisible())
        else:
            view.record_btn.setChecked(False)
        path = view._last_saved_paths[0]
        loaded = load_impedance_recording(path)

        assert {reading.channel for reading in loaded.readings} == set(range(5))
        assert loaded.channel_names[-1] == "REF"
        assert loaded.metadata["mode"] == "impedance"
        assert loaded.metadata["payload_units"] == "kohms"
        assert not list(tmp_path.rglob("*_raw.csv"))
        assert not list(tmp_path.rglob("*.edf"))
        assert not list(tmp_path.rglob("*.fif"))
        assert manager._filled == 0
        before = len(manager.impedance_history())
        view._tick()
        view._tick()
        assert len(manager.impedance_history()) == before
        assert len(view._reviews) == (0 if stop_by_closing else 1)
        if not stop_by_closing:
            assert len(view._reviews[0].recording.readings) == len(loaded.readings)
    finally:
        window.close()
        _until(qt_app, lambda: window.manager is None and not window.isVisible())
