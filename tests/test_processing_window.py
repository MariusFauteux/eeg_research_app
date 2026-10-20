"""Processing Lab regressions exercised with an offscreen Qt window."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from ganglion_studio.core.recording_loader import LoadedRecording
from ganglion_studio.ui import processing_window as ui


def test_analysis_requires_at_least_one_enabled_channel(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui.proc, "available_methods", lambda: {"asr": False, "aas": False})
    warnings = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda *args: warnings.append(args[1:]))

    def unexpected_analysis(**kwargs):
        raise AssertionError("An analysis report must not be constructed with no channels")

    monkeypatch.setattr(ui, "AnalysisWindow", unexpected_analysis)
    window = ui.ProcessingWindow()
    try:
        recording = LoadedRecording(np.zeros((2, 200)), 200, ["a", "b"])
        window._rec = recording
        window._processed = recording.eeg.copy()
        window._rebuild_channel_meta(recording)
        for checkbox in window._enable_checks:
            checkbox.setChecked(False)

        window._open_analysis()

        assert warnings == [("No channels selected",
                             "Enable at least one channel before generating analysis plots.")]
        assert not window._analysis_windows
    finally:
        window._debounce.stop()
        window.close()
    # Keep the QApplication alive until every widget has been closed.
    assert app is not None


def test_previous_recording_worker_cannot_replace_current_results(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui.proc, "available_methods", lambda: {"asr": False, "aas": False})

    class ControlledWorker(QObject):
        done = pyqtSignal(object, list)
        failed = pyqtSignal(str)
        finished = pyqtSignal()

        def __init__(self, *args):
            super().__init__()
            self.running = False

        def start(self):
            self.running = True

        def isRunning(self):
            return self.running

        def requestInterruption(self):
            pass

    monkeypatch.setattr(ui, "ProcessingWorker", ControlledWorker)
    recordings = {
        "a": LoadedRecording(np.ones((2, 200)), 200, ["a1", "a2"]),
        "b": LoadedRecording(np.full((2, 400), 2.0), 200, ["b1", "b2"]),
    }
    monkeypatch.setattr(ui, "load_recording", recordings.__getitem__)
    window = ui.ProcessingWindow()
    try:
        window._load("a")
        previous_worker = window._worker
        window._load("b")
        assert window._recompute_pending

        previous_worker.done.emit(np.full((2, 200), -1.0), ["Old result"])
        previous_worker.failed.emit("Old failure")

        np.testing.assert_array_equal(window._processed, recordings["b"].eeg)
        assert "Old" not in window.status.text()
        assert not window.apply_btn.isEnabled()

        previous_worker.running = False
        previous_worker.finished.emit()
        window._recompute()
        current_worker = window._worker
        result = recordings["b"].eeg * 3
        current_worker.done.emit(result, ["Current result"])

        np.testing.assert_array_equal(window._processed, result)
        assert window.status.text() == "Current result"
        assert window.apply_btn.isEnabled()
    finally:
        if window._worker is not None:
            window._worker.running = False
            window._worker.finished.emit()
        window._debounce.stop()
        window.close()
    assert app is not None


def test_analysis_requires_continuous_eeg_when_recording_contains_gap(monkeypatch):
    app = QApplication.instance() or QApplication([])
    warnings = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda *args: warnings.append(args[1:]))
    window = ui.ProcessingWindow()
    try:
        eeg = np.ones((2, 400))
        eeg[:, 100:200] = np.nan
        window._rec = LoadedRecording(eeg, 200, ["a", "b"])
        window._rebuild_channel_meta(window._rec)

        window._open_analysis()

        assert warnings[0][0] == "Select continuous EEG"
        assert not window._analysis_windows
    finally:
        window._debounce.stop()
        window.close()
    assert app is not None
