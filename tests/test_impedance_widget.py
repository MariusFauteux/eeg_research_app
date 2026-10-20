"""The EEG-session impedance tab is a consumer of actual measurement events."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from ganglion_studio.core.dsp import FilterSettings
from ganglion_studio.core.impedance import ImpedanceReading
from ganglion_studio.ui.plots import impedance_widget as ui


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _manager(readings):
    return SimpleNamespace(
        impedance_channel_names=["A", "B", "C", "D", "REF"],
        impedance_history=lambda: list(readings),
        latest_impedance_readings=lambda: list({r.channel: r for r in readings}.values()),
        impedance_mode=False, impedance_check_status="idle", impedance_check_started_at=None,
        start_impedance=Mock(), stop_impedance=Mock(),
    )


def test_tab_retains_last_check_and_never_fabricates_repeated_gui_samples(qt_app, monkeypatch):
    now = [1000.]
    monkeypatch.setattr(ui.time, "time", lambda: now[0])
    readings = [ImpedanceReading(995., 0, 12000.), ImpedanceReading(996., 4, 7000.),
                ImpedanceReading(998., 0, 12000.)]
    manager = _manager(readings)
    widget = ui.ImpedanceWidget(manager)
    try:
        for _ in range(4):
            widget.update_plot(FilterSettings(), [False] * 4)
        x, y = widget.time_series.curves[0].getData()
        np.testing.assert_array_equal(x, [0, 3])
        np.testing.assert_array_equal(y, [12, 12])
        ref_x, ref_y = widget.time_series.curves[4].getData()
        np.testing.assert_array_equal(ref_x, [1])
        np.testing.assert_array_equal(ref_y, [7])
        now[0] += 100
        widget.update_plot(FilterSettings(), [])
        assert len(widget.time_series.curves[0].getData()[0]) == 2
        assert "Last impedance measurements" in widget.status.text()
        assert widget.current_table.item(0, 3).text() == "Last check"
        manager.start_impedance.assert_not_called()
        manager.stop_impedance.assert_not_called()
    finally:
        widget.close()


def test_tab_distinguishes_current_previous_missing_and_stale_updates(qt_app, monkeypatch):
    monkeypatch.setattr(ui.time, "time", lambda: 1000.)
    readings = [ImpedanceReading(980., 0, 10000.), ImpedanceReading(989., 1, 15000.),
                ImpedanceReading(999.5, 4, 5000.)]
    manager = _manager(readings)
    manager.impedance_check_status = "measuring"
    manager.impedance_check_started_at = 985.
    manager.impedance_mode = True
    widget = ui.ImpedanceWidget(manager)
    try:
        assert widget.current_table.item(0, 3).text() == "Previous check"
        assert widget.current_table.item(1, 3).text() == "Stale"
        assert widget.current_table.item(2, 3).text() == "Waiting"
        assert widget.current_table.item(4, 3).text() == "Current"
        assert widget.current_table.item(4, 2).text() == "0.5 s ago"
        assert "check running" in widget.status.text()
        widget.set_channel_names(["Fp1", "Fp2", "O1", "O2"])
        assert widget.current_table.item(0, 0).text() == "Fp1"
        assert widget.time_series.channel_checks[0].text() == "Fp1"
        assert widget.current_table.item(4, 0).text() == "REF"
    finally:
        widget.close()


def test_empty_tab_directs_user_to_timed_session_control(qt_app):
    widget = ui.ImpedanceWidget(_manager([]))
    try:
        assert "Get impedance while recording" in widget.status.text()
        assert widget.current_table.item(0, 1).text() == "—"
    finally:
        widget.close()
