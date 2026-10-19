"""Live spectral and pulse views do not show stale or fabricated gap results."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from ganglion_studio.core.dsp import FilterSettings
from ganglion_studio.ui.plots import pulse_diagnostics_widget, spectrogram_widget
from ganglion_studio.ui.panels.stats_panel import StatsPanel


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_spectrogram_uses_finite_levels_and_clears_old_plots_when_all_missing(qt_app):
    channel = np.sin(2 * np.pi * 10 * np.arange(1000) / 200)
    manager = SimpleNamespace(eeg_channels=[0], channel_names=["A"], sampling_rate=200,
                              recent_eeg=lambda seconds: channel[None, :])
    widget = spectrogram_widget.SpectrogramWidget(manager)
    try:
        widget.update_plot(FilterSettings(), [True])
        assert widget._img.image is not None
        channel[400:600] = np.nan
        widget._level_frames = 0
        widget.update_plot(FilterSettings(), [True])
        assert np.isfinite(widget._levels).all()
        assert np.isnan(widget._img.image).any()
        assert np.isfinite(widget._fft_curve.getData()[1]).all()
        channel[:] = np.nan

        widget.update_plot(FilterSettings(), [True])

        assert widget._img.image is None
        x, y = widget._fft_curve.getData()
        assert x is None or x.size == 0
        assert y is None or y.size == 0
    finally:
        widget.close()


def test_pulse_verdict_is_suspended_instead_of_treating_a_gap_as_no_pulse(qt_app, monkeypatch):
    data = np.zeros((2, 2000))
    data[0, 900:1100] = np.nan
    manager = SimpleNamespace(sampling_rate=200, eeg_channels=[0], channel_names=["A"],
                              marker_channel=1, recent=lambda seconds: data,
                              recent_loss=lambda seconds: np.zeros(2000))
    diagnose = Mock(side_effect=AssertionError("a discontinuous window has no valid pulse verdict"))
    monkeypatch.setattr(pulse_diagnostics_widget.pdiag, "diagnose", diagnose)
    widget = pulse_diagnostics_widget.PulseDiagnosticsWidget(manager)
    try:
        widget.metrics.setText("old verdict")
        widget._shape_curve.setData([0, 1], [1, 0])

        widget.update_plot(FilterSettings(), [True])

        assert "EEG gap" in widget.verdict.text()
        assert widget.metrics.text() == ""
        x, _ = widget._shape_curve.getData()
        assert x is None or x.size == 0
        diagnose.assert_not_called()
    finally:
        widget.close()


def test_stats_panel_marks_missing_channels_without_nan_values_or_good_quality(qt_app):
    data = np.stack([np.sin(2 * np.pi * 10 * np.arange(400) / 200), np.full(400, np.nan)])
    manager = SimpleNamespace(eeg_channels=[0, 1], channel_names=["A", "B"], sampling_rate=200,
                              recording=True, _filled=400, recent_eeg=lambda seconds: data)
    widget = StatsPanel(manager)
    try:
        widget.update_stats(FilterSettings(), [True, True])

        assert widget.table.item(1, 6).text() == "EEG gap"
        assert all(widget.table.item(1, column).text() == "—" for column in range(1, 6))
        assert "nan" not in widget.general.text().lower()
        assert "EEG gap" in widget.general.text()
        assert widget.table.item(0, 6).text() in ("good", "ok", "bad")
        data[:] = np.nan
        widget.update_stats(FilterSettings(), [True, True])
        assert "mean RMS —" in widget.general.text()
        data[1] = 0
        widget.update_stats(FilterSettings(), [True, False])
        assert widget.table.item(1, 6).text() == "off"
    finally:
        widget.close()
