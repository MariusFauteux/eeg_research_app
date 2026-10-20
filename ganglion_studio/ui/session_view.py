"""Session view: toolbar, control panels, plot tabs, refresh timer, recording."""

from __future__ import annotations

import time
import os
from typing import List, Optional, Protocol, runtime_checkable

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio import palette
from ganglion_studio.core import protocols
from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.dsp import FilterSettings
from ganglion_studio.core.impedance_check import CHECK_MARKERS
from ganglion_studio.core.session import MarkerEvent, SessionConfig, SessionRecorder
from ganglion_studio.ui import theme
from ganglion_studio.ui.channel_setup_dialog import ChannelSetupDialog
from ganglion_studio.ui.review_window import ReviewWindow
# Plot tabs (central tab area) live in ui/plots/; side panels in ui/panels/.
from ganglion_studio.ui.plots.band_power_widget import BandPowerWidget
from ganglion_studio.ui.plots.impedance_widget import ImpedanceWidget
from ganglion_studio.ui.plots.psd_widget import PSDWidget
from ganglion_studio.ui.plots.pulse_diagnostics_widget import PulseDiagnosticsWidget
from ganglion_studio.ui.plots.spectrogram_widget import SpectrogramWidget
from ganglion_studio.ui.plots.time_series_widget import TimeSeriesWidget
from ganglion_studio.ui.panels.channel_panel import ChannelPanel
from ganglion_studio.ui.panels.filter_panel import FilterPanel
from ganglion_studio.ui.panels.marker_panel import MarkerPanel, MarkerType
from ganglion_studio.ui.panels.protocol_panel import ProtocolPanel
from ganglion_studio.ui.panels.stats_panel import StatsPanel


@runtime_checkable
class PlotTab(Protocol):
    """Structural contract for the plot tabs hosted in the session view.

    A plot tab is just a ``QWidget`` that can redraw itself on demand. Tabs do
    NOT need to inherit this Protocol -- it documents the duck-typed interface
    that :meth:`SessionView._tick` relies on (and enables an optional
    ``isinstance`` check). See docs/EXTENDING.md for a walkthrough.

    Required
        update_plot(settings, active): redraw using the live display-filter
        ``settings`` and the per-channel ``active`` flags.

    Optional (accessed via getattr/hasattr, so safe to omit)
        refresh_hz (float): cap on redraws per second; omit for "every tick".
        set_channel_names(names): relabel traces when the montage changes.
    """

    def update_plot(self, settings: FilterSettings, active: List[bool]) -> None:
        ...


class ImpedanceCheckWorker(QThread):
    """Switch board modes without blocking the recording controls."""

    def __init__(self, manager: BoardManager, action: str, duration: int) -> None:
        super().__init__()
        self.manager = manager
        self.action = action
        self.duration = duration
        self.error: Optional[str] = None

    def run(self) -> None:
        try:
            if self.action == "begin":
                self.manager.begin_impedance_check(self.duration)
            else:
                self.manager.end_impedance_check()
        except Exception as exc:
            self.error = str(exc)


class SessionView(QWidget):
    exit_session = pyqtSignal()
    open_processing = pyqtSignal()
    shutdown_ready = pyqtSignal()

    def __init__(self, manager: BoardManager, config: SessionConfig) -> None:
        super().__init__()
        self._manager = manager
        self._config = config
        self._settings = FilterSettings(notch_freq=config.notch_freq)
        self._active: List[bool] = list(manager.channel_active)
        self._recorder: Optional[SessionRecorder] = None
        self._pending_recording = None  # retain samples if writing to disk fails
        self._pending_impedance_readings = None
        self._check_worker: Optional[ImpedanceCheckWorker] = None
        self._check_deadline: Optional[float] = None
        self._check_quiescing = False
        self._check_event_count = 0
        self._deferred_markers: list[tuple[int, float]] = []
        self._save_after_check = False
        self._reviews: List[ReviewWindow] = []  # keep refs so windows aren't GC'd
        self._disconnect_handled = False  # one-shot guard for the BLE-drop notice
        self._protocol_started_recording = False  # did the protocol auto-start it?

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addLayout(self._build_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_tabs())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 900, 320])

        # Acquisition runs on its own thread; the GUI timer only renders.
        self._manager.start_acquisition()
        self._last_status = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._set_refresh(30)
        self._check_timer = QTimer(self)
        self._check_timer.setInterval(100)
        self._check_timer.timeout.connect(self._check_timeout)
        self._update_check_controls()

    # ----------------------------------------------------------- build UI
    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        back_btn = QPushButton("\u25C0 Dashboard")
        back_btn.clicked.connect(self._on_exit)
        bar.addWidget(back_btn)

        title = QLabel(f"  {self._config.name}")
        title.setStyleSheet("font-weight:700; font-size:15px;")
        bar.addWidget(title)
        mode = "DEMO" if self._config.demo else "GANGLION"
        tag = QLabel(mode)
        tag.setStyleSheet(
            f"background:{palette.ACCENT}; color:{palette.WHITE}; border-radius:4px; padding:2px 6px; font-weight:600;"
            if not self._config.demo else
            f"background:{palette.OK}; color:#15171c; border-radius:4px; padding:2px 6px; font-weight:600;"
        )
        bar.addWidget(tag)
        bar.addStretch(1)

        self.setup_btn = QPushButton("Channel setup")
        self.setup_btn.setToolTip("Set channel type, electrode and 10-20 placement")
        self.setup_btn.clicked.connect(self._open_channel_setup)
        bar.addWidget(self.setup_btn)

        lab_btn = QPushButton("Processing Lab")
        lab_btn.setToolTip("Open the offline processing playground")
        lab_btn.clicked.connect(self.open_processing.emit)
        bar.addWidget(lab_btn)

        self.pause_btn = QPushButton("Pause stream")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._on_pause)
        bar.addWidget(self.pause_btn)

        self.record_btn = QPushButton("\u25CF Record")
        self.record_btn.setCheckable(True)
        self.record_btn.setStyleSheet("QPushButton:checked { background:#b03434; }")
        self.record_btn.toggled.connect(self._on_record)
        bar.addWidget(self.record_btn)

        self.impedance_btn = QPushButton("Get impedance")
        self.impedance_btn.setToolTip(
            "Pause EEG for a timed impedance check, then resume EEG automatically. "
            "The interruption and impedance readings are saved with the recording."
        )
        self.impedance_btn.clicked.connect(self._on_impedance_check)
        bar.addWidget(self.impedance_btn)
        self.impedance_duration = QSpinBox()
        self.impedance_duration.setRange(1, 3600)
        self.impedance_duration.setValue(10)
        self.impedance_duration.setSuffix(" s")
        self.impedance_duration.setAccessibleName("Impedance check duration")
        self.impedance_duration.setToolTip(
            "Time spent measuring impedance. Short checks may not cover every electrode."
        )
        bar.addWidget(self.impedance_duration)

        bar.addWidget(QLabel("Refresh"))
        self.refresh_spin = QSpinBox()
        self.refresh_spin.setRange(2, 60)
        self.refresh_spin.setValue(30)
        self.refresh_spin.setSuffix(" Hz")
        self.refresh_spin.valueChanged.connect(self._set_refresh)
        bar.addWidget(self.refresh_spin)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(theme.MUTED_QSS)
        bar.addWidget(self.status_label)
        return bar

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.channel_panel = ChannelPanel(self._manager)
        self.channel_panel.channels_changed.connect(self._on_channels_changed)
        layout.addWidget(self.channel_panel)

        self.filter_panel = FilterPanel(self._config.notch_freq)
        self.filter_panel.filters_changed.connect(self._on_filters_changed)
        layout.addWidget(self.filter_panel)

        self.stats_panel = StatsPanel(self._manager)
        layout.addWidget(self.stats_panel)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setFixedWidth(310)
        return scroll

    def _build_tabs(self) -> QTabWidget:
        # Every widget added here satisfies the PlotTab protocol (top of file).
        self.tabs = QTabWidget()
        self.time_series = TimeSeriesWidget(self._manager)
        self.psd = PSDWidget(self._manager)
        self.spectrogram = SpectrogramWidget(self._manager)
        self.impedance = ImpedanceWidget(self._manager)
        self.band_power = BandPowerWidget(self._manager)
        self.diagnostics = PulseDiagnosticsWidget(self._manager)

        self.tabs.addTab(self.time_series, "Time Series")
        self.tabs.addTab(self.psd, "PSD")
        self.tabs.addTab(self.spectrogram, "Spectrogram / FFT")
        self.tabs.addTab(self.impedance, "Impedance")
        self.tabs.addTab(self.band_power, "Band Power")
        self.tabs.addTab(self.diagnostics, "Diagnostics")
        return self.tabs

    def _build_right_panel(self) -> QWidget:
        # Right column stacks the guided Protocol helper above the manual Markers
        # panel, in a vertical splitter so the researcher can resize between them.
        self.protocol_panel = ProtocolPanel()
        self.protocol_panel.marker.connect(self._on_protocol_marker)
        self.protocol_panel.started.connect(self._on_protocol_started)
        self.protocol_panel.stopped.connect(self._on_protocol_stopped)

        self.marker_panel = MarkerPanel(self)
        self.marker_panel.marker_fired.connect(self._on_marker)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.protocol_panel)
        right.addWidget(self.marker_panel)
        right.setStretchFactor(0, 0)
        right.setStretchFactor(1, 1)
        right.setSizes([380, 360])
        right.setFixedWidth(340)
        return right

    # --------------------------------------------------------------- logic
    def _set_refresh(self, hz: int) -> None:
        self._timer.start(max(16, int(1000 / hz)))

    def _tick(self) -> None:
        # A native-BLE link can drop mid-session (e.g. electrical disturbance);
        # tell the user clearly instead of leaving a silently-frozen trace.
        if self._manager.is_disconnected() and not self._disconnect_handled:
            self._disconnect_handled = True
            self.status_label.setText("Bluetooth disconnected")
            QMessageBox.warning(
                self, "Bluetooth disconnected",
                "The Ganglion link dropped. Return to the Dashboard and start a "
                "new session to reconnect.",
            )
            return
        # Data acquisition happens on the background thread; here we only
        # render the visible tab, throttled to its own preferred rate.
        current: PlotTab = self.tabs.currentWidget()
        checking = self._check_worker is not None or self._manager.impedance_check_active
        if (not checking or current is self.impedance) and hasattr(current, "update_plot") and self._due(current):
            current.update_plot(self._settings, self._active)
        if self._check_worker is None and self._due(self.stats_panel):
            self.stats_panel.update_stats(self._settings, self._active)
        self._update_status()

    @staticmethod
    def _due(widget) -> bool:
        """Rate-limit a widget to its ``refresh_hz`` (default: every tick)."""
        hz = getattr(widget, "refresh_hz", None)
        if not hz:
            return True
        now = time.monotonic()
        last = getattr(widget, "_last_render", 0.0)
        if now - last >= (1.0 / hz) - 1e-3:
            widget._last_render = now
            return True
        return False

    def _update_status(self) -> None:
        now = time.monotonic()
        if now - self._last_status < 0.5:  # throttle status to ~2 Hz
            return
        self._last_status = now
        if self._check_worker is not None:
            self.status_label.setText("Starting impedance…" if self._check_worker.action == "begin"
                                      else "Resuming EEG…")
            return
        if self._manager.impedance_check_active:
            if self._manager.impedance_check_status == "restore_failed":
                self.status_label.setText("EEG stopped · resume required")
            else:
                left = max(0.0, (self._check_deadline or now) - now)
                self.status_label.setText(f"Impedance · {left:.1f} s remaining")
            return
        rec = ""
        if self._manager.recording:
            secs = self._manager.recorded_sample_count() / max(1, self._manager.sampling_rate)
            rec = f" | REC {secs:0.1f}s"
        self.status_label.setText(
            f"{self._manager.sampling_rate} Hz | {len(self._manager.eeg_channels)} ch{rec}"
        )

    def _on_filters_changed(self, settings: FilterSettings) -> None:
        self._settings = settings

    def _on_channels_changed(self, active: List[bool]) -> None:
        self._active = active

    def _open_channel_setup(self) -> None:
        dialog = ChannelSetupDialog(self._manager, self)
        if dialog.exec() != ChannelSetupDialog.DialogCode.Accepted:
            return
        names = dialog.names()
        self._manager.set_channel_config(
            names, dialog.types(), dialog.electrodes(), dialog.placements()
        )
        for widget in (self.time_series, self.psd, self.spectrogram, self.impedance,
                       self.stats_panel, self.channel_panel, self.diagnostics):
            if hasattr(widget, "set_channel_names"):
                widget.set_channel_names(names)

    def _on_pause(self, paused: bool) -> None:
        if paused:
            self._manager.stop()
            self.pause_btn.setText("Resume stream")
        else:
            self._manager.start()
            self.pause_btn.setText("Pause stream")
        # Scrolling/zooming the time-series view is only enabled while paused.
        self.time_series.set_paused(paused)
        self._update_check_controls()

    def has_pending_transition(self) -> bool:
        return self._check_worker is not None

    def _update_check_controls(self) -> None:
        busy = self._check_worker is not None
        active = self._manager.impedance_check_active
        retry = self._manager.impedance_check_status == "restore_failed"
        self.impedance_btn.setText("Retry EEG resume" if retry else
                                   "Resume EEG now" if active else "Get impedance")
        self.impedance_btn.setEnabled(not busy and not self._check_quiescing and
            self._pending_recording is None and self._manager.recording and
            (active or self._manager.streaming))
        self.impedance_duration.setEnabled(not active and not busy)
        self.pause_btn.setEnabled(not active and not busy)
        self.record_btn.setEnabled(not active and not busy)
        self.setup_btn.setEnabled(not active and not busy and not self._manager.recording)
        self.channel_panel.setEnabled(not active and not busy)

    def _on_impedance_check(self) -> None:
        if self._check_worker is not None or self._check_quiescing:
            return
        if self._manager.impedance_check_active:
            self._start_check_worker("end")
        elif self._manager.recording and self._manager.streaming and self._pending_recording is None:
            self.tabs.setCurrentWidget(self.impedance)
            self._start_check_worker("begin")

    def _start_check_worker(self, action: str) -> None:
        self._check_timer.stop()
        self._check_deadline = None
        worker = ImpedanceCheckWorker(self._manager, action, self.impedance_duration.value())
        self._check_worker = worker
        worker.finished.connect(self._on_check_finished)
        self._update_check_controls()
        worker.start()

    def _sync_check_events(self) -> None:
        events = self._manager.impedance_check_events()
        for event in events[self._check_event_count:]:
            ts, code, label = event["timestamp"], event["code"], event["label"]
            self.marker_panel._append_log(ts, MarkerType(label, code, palette.MARKER))
            if self._recorder is not None and self._pending_recording is None:
                self._recorder.add_marker(MarkerEvent(ts, code, label))
        self._check_event_count = len(events)

    def _on_check_finished(self) -> None:
        worker = self.sender()
        if worker is not self._check_worker:
            return
        self._check_worker = None
        worker.deleteLater()
        # Marker requests during the BLE transition retain their original times.
        # Inserting them after the worker releases its locks keeps Qt responsive.
        for code, ts in self._deferred_markers:
            self._manager.insert_marker(code, timestamp=ts)
        self._deferred_markers.clear()
        self._sync_check_events()
        self._last_status = 0.0
        if self._check_quiescing:
            self.shutdown_ready.emit()
            return
        if self._manager.impedance_check_status == "measuring":
            started = self._manager.impedance_check_started_at or time.time()
            remaining = max(0.0, worker.duration - (time.time() - started))
            self._check_deadline = time.monotonic() + remaining
            self._check_timer.start()
        self._update_check_controls()
        if worker.error is not None:
            message = ("EEG is still stopped. Use Retry EEG resume, or leave the "
                       "session to save the recording with its interruption."
                       if self._manager.impedance_check_active else "EEG has resumed.")
            QMessageBox.warning(self, "Impedance check failed", f"{message}\n\n{worker.error}")
        if self._save_after_check and not self._manager.impedance_check_active:
            self._save_after_check = False
            self._save_recording()

    def _check_timeout(self) -> None:
        if self._check_worker is None and self._check_deadline is not None \
                and time.monotonic() >= self._check_deadline:
            self._start_check_worker("end")

    def _on_record(self, recording: bool) -> None:
        if recording:
            self._recorder = SessionRecorder(self._config)
            try:
                self._recorder.begin()
            except OSError as exc:
                self._recorder = None
                self.record_btn.blockSignals(True)
                self.record_btn.setChecked(False)
                self.record_btn.blockSignals(False)
                QMessageBox.warning(self, "Recording failed", str(exc))
                return
            self._manager.start_recording()
            self._check_event_count = len(self._manager.impedance_check_events())
        else:
            if self._manager.impedance_check_active or self._check_worker is not None:
                self._save_after_check = True
                self.record_btn.blockSignals(True)
                self.record_btn.setChecked(True)
                self.record_btn.blockSignals(False)
                return
            self._save_recording()
        self._update_check_controls()

    def _save_recording(self, show_review: bool = True, raw_data=None) -> bool:
        if self._recorder is None:
            return True
        if self._pending_recording is None:
            raw = self._manager.stop_recording() if raw_data is None else raw_data
            self._sync_check_events()
            loss_samples = self._manager.recorded_loss_indices()
            code_labels = dict(protocols.PROTOCOL_MARKERS)
            code_labels.update({mt.code: mt.label for mt in self.marker_panel._types})
            code_labels.update(CHECK_MARKERS)
            markers = self._recorder.align_markers(
                self._manager.recorded_marker_events(), code_labels)
            n_eeg = len(self._manager.eeg_channels)
            meta = {
                "sampling_rate": self._manager.sampling_rate,
                "board_id": self._manager.board_id,
                "eeg_channels": self._manager.eeg_channels,
                "channel_names": list(self._manager.channel_names[:n_eeg]),
                # False means switched off, rather than an active channel that
                # died or flat-lined. Keep this distinction when reloading.
                "channel_active": list(self._manager.channel_active[:n_eeg]),
                "channel_types": list(self._manager.channel_types[:n_eeg]),
                "electrodes": list(self._manager.electrodes[:n_eeg]),
                "placements": list(self._manager.placements[:n_eeg]),
                "marker_channel": self._manager.marker_channel,
                "timestamp_channel": self._manager.timestamp_channel,
                "notch_freq": self._config.notch_freq,
                "loss_samples": loss_samples,
                "eeg_gaps": self._manager.recorded_eeg_gaps(),
                "impedance_channel_names": self._manager.impedance_channel_names,
                "impedance_raw_unit_basis": self._manager.impedance_raw_unit_basis,
                "marker_events": [dict(sample=m.sample, code=m.code, label=m.label)
                                  for m in markers],
            }
            self._pending_recording = (raw, meta, loss_samples)
            readings = self._manager.recorded_impedance_checks()
            self._pending_impedance_readings = readings if readings or meta["eeg_gaps"] else None
        raw, meta, loss_samples = self._pending_recording
        # Always keep the lossless backup (CSV + meta + marker + packet-loss log).
        try:
            paths = self._recorder.save(raw, meta, loss_samples=loss_samples,
                                        impedance_readings=self._pending_impedance_readings)
            # The review can export to another folder: retain a resolvable link
            # to the measurements stored beside this EEG recording.
            for path in paths:
                if path.endswith("_impedance.csv"):
                    meta["impedance_file"] = os.path.abspath(path)
                elif path.endswith("_impedance_meta.json"):
                    meta["impedance_metadata_file"] = os.path.abspath(path)
        except Exception as exc:
            if self.protocol_panel._running:
                self.protocol_panel._stop_run()
            self.protocol_panel.run_btn.setEnabled(False)
            self.record_btn.blockSignals(True)
            self.record_btn.setChecked(True)
            self.record_btn.blockSignals(False)
            self.record_btn.setText("Retry save")
            self._update_check_controls()
            QMessageBox.warning(
                self, "Save failed",
                f"The recording is still held in memory. Retry saving before "
                f"leaving this session.\n\n{exc}",
            )
            return False
        self._recorder = None
        self._pending_recording = None
        self._pending_impedance_readings = None
        self.record_btn.blockSignals(True)
        self.record_btn.setChecked(False)
        self.record_btn.blockSignals(False)
        self.record_btn.setText("\u25CF Record")
        self.protocol_panel.run_btn.setEnabled(True)
        self._update_check_controls()

        if raw is None or raw.ndim != 2 or raw.shape[1] == 0:
            if show_review:
                QMessageBox.information(self, "Recording", "Recording was empty.")
            return True
        if not show_review:
            return True

        # Open the review window so the user can browse, edit markers and export.
        # Protocol markers (codes 6+) aren't buttons on the manual panel, so seed the
        # code->label map from protocols.PROTOCOL_MARKERS first, then let the manual
        # panel's types override any shared codes (1..5).
        types = list(getattr(self.marker_panel, "_types", []))
        code_labels = dict(protocols.PROTOCOL_MARKERS)
        code_labels.update({mt.code: mt.label for mt in types})
        code_labels.update(CHECK_MARKERS)
        review = ReviewWindow(raw, meta, code_labels, types, title=self._config.name,
                              loss_samples=loss_samples)
        review.show()
        self._reviews.append(review)
        return True

    def _on_marker(self, code: int, label: str, ts: float) -> None:
        if self._check_worker is not None:
            self._deferred_markers.append((code, ts))
        else:
            self._manager.insert_marker(code, timestamp=ts)
        if self._recorder is not None and self._pending_recording is None:
            self._recorder.add_marker(MarkerEvent(timestamp=ts, code=code, label=label))

    def _on_protocol_started(self) -> None:
        if self._pending_recording is not None:
            self.protocol_panel._stop_run()
            return
        # Start recording if it isn't already, so the protocol's annotations are
        # captured. Remember that WE started it, so we can stop it again when the
        # protocol ends -- while leaving a recording the user started themselves
        # untouched.
        self._protocol_started_recording = not self._manager.recording
        if self._protocol_started_recording:
            self.record_btn.setChecked(True)  # triggers _on_record(True)
            if not self._manager.recording:
                self._protocol_started_recording = False
                self.protocol_panel._stop_run()

    def _on_protocol_stopped(self) -> None:
        # When the protocol ends (finished or stopped early), stop the recording
        # too -- but only the one the protocol itself started.
        if self._protocol_started_recording and self._manager.recording:
            self.record_btn.setChecked(False)  # triggers _on_record(False) -> save
        self._protocol_started_recording = False

    def _on_protocol_marker(self, code: int, label: str) -> None:
        # Route protocol annotations through the manual marker panel so they show
        # up in the event log AND flow down the normal marker path (sample-aligned
        # marker channel + _markers.csv), exactly like a hand-dropped marker.
        marker = next((m for m in self.marker_panel._types if m.code == code),
                      MarkerType(label, code, palette.MARKER))
        self.marker_panel.fire(marker)

    def _on_exit(self) -> None:
        if self._recorder is not None:
            res = QMessageBox.question(
                self, "Recording in progress",
                "A recording is active. Save and exit?",
            )
            if res != QMessageBox.StandardButton.Yes:
                return
        self.exit_session.emit()

    def quiesce_for_shutdown(self) -> None:
        """Stop GUI producers before an off-thread final acquisition drain."""
        self._check_quiescing = True
        self._check_timer.stop()
        self._save_after_check = False
        self.protocol_panel.shutdown()
        self._timer.stop()
        self.setEnabled(False)

    def resume_after_failed_shutdown(self) -> None:
        """Keep the view usable for retry when persistence prevents closing."""
        self.setEnabled(True)
        self._check_quiescing = False
        self._manager.start_acquisition()
        self._timer.start()
        self._update_check_controls()

    def shutdown(self, raw_data=None) -> bool:
        if self._check_worker is not None:
            self.quiesce_for_shutdown()
            return False
        self._check_timer.stop()
        self.protocol_panel.shutdown()  # stop the runner + close the cue window
        if not self._save_recording(show_review=False, raw_data=raw_data):
            return False
        self._timer.stop()
        self._manager.stop_acquisition(timeout=0.0)
        return True
