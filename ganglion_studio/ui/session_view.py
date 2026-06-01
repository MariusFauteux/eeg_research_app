"""Session view: toolbar, control panels, plot tabs, refresh timer, recording."""

from __future__ import annotations

import time
from typing import List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
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

from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.dsp import FilterSettings
from ganglion_studio.core.session import MarkerEvent, SessionConfig, SessionRecorder
from ganglion_studio.ui.channel_setup_dialog import ChannelSetupDialog
from ganglion_studio.ui.review_window import ReviewWindow
from ganglion_studio.ui.widgets.band_power_widget import BandPowerWidget
from ganglion_studio.ui.widgets.channel_panel import ChannelPanel
from ganglion_studio.ui.widgets.filter_panel import FilterPanel
from ganglion_studio.ui.widgets.impedance_widget import ImpedanceWidget
from ganglion_studio.ui.widgets.marker_panel import MarkerPanel
from ganglion_studio.ui.widgets.psd_widget import PSDWidget
from ganglion_studio.ui.widgets.spectrogram_widget import SpectrogramWidget
from ganglion_studio.ui.widgets.stats_panel import StatsPanel
from ganglion_studio.ui.widgets.time_series import TimeSeriesWidget


class SessionView(QWidget):
    exit_session = pyqtSignal()
    open_processing = pyqtSignal()

    def __init__(self, manager: BoardManager, config: SessionConfig) -> None:
        super().__init__()
        self._manager = manager
        self._config = config
        self._settings = FilterSettings(notch_freq=config.notch_freq)
        self._active: List[bool] = list(manager.channel_active)
        self._recorder: Optional[SessionRecorder] = None
        self._reviews: List[ReviewWindow] = []  # keep refs so windows aren't GC'd

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
            "background:#4f8ef7; color:#fff; border-radius:4px; padding:2px 6px; font-weight:600;"
            if not self._config.demo else
            "background:#e2c044; color:#15171c; border-radius:4px; padding:2px 6px; font-weight:600;"
        )
        bar.addWidget(tag)
        bar.addStretch(1)

        setup_btn = QPushButton("Channel setup")
        setup_btn.setToolTip("Set channel type, electrode and 10-20 placement")
        setup_btn.clicked.connect(self._open_channel_setup)
        bar.addWidget(setup_btn)

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

        bar.addWidget(QLabel("Refresh"))
        self.refresh_spin = QSpinBox()
        self.refresh_spin.setRange(2, 60)
        self.refresh_spin.setValue(30)
        self.refresh_spin.setSuffix(" Hz")
        self.refresh_spin.valueChanged.connect(self._set_refresh)
        bar.addWidget(self.refresh_spin)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color:#9aa0aa;")
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
        self.tabs = QTabWidget()
        self.time_series = TimeSeriesWidget(self._manager)
        self.psd = PSDWidget(self._manager)
        self.spectrogram = SpectrogramWidget(self._manager)
        self.impedance = ImpedanceWidget(self._manager)
        self.band_power = BandPowerWidget(self._manager)

        self.tabs.addTab(self.time_series, "Time Series")
        self.tabs.addTab(self.psd, "PSD")
        self.tabs.addTab(self.spectrogram, "Spectrogram / FFT")
        self.tabs.addTab(self.impedance, "Impedance")
        self.tabs.addTab(self.band_power, "Band Power")
        return self.tabs

    def _build_right_panel(self) -> QWidget:
        self.marker_panel = MarkerPanel(self)
        self.marker_panel.marker_fired.connect(self._on_marker)
        self.marker_panel.setFixedWidth(320)
        return self.marker_panel

    # --------------------------------------------------------------- logic
    def _set_refresh(self, hz: int) -> None:
        self._timer.start(max(16, int(1000 / hz)))

    def _tick(self) -> None:
        # Data acquisition happens on the background thread; here we only
        # render the visible tab, throttled to its own preferred rate.
        current = self.tabs.currentWidget()
        if hasattr(current, "update_plot") and self._due(current):
            current.update_plot(self._settings, self._active)
        if self._due(self.stats_panel):
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
                       self.stats_panel, self.channel_panel):
            if hasattr(widget, "set_channel_names"):
                widget.set_channel_names(names)

    def _on_pause(self, paused: bool) -> None:
        if paused:
            self._manager.stop()
            self.pause_btn.setText("Resume stream")
        else:
            self._manager.start()
            self.pause_btn.setText("Pause stream")

    def _on_record(self, recording: bool) -> None:
        if recording:
            self._recorder = SessionRecorder(self._config)
            self._recorder.begin()
            self._manager.start_recording()
        else:
            self._save_recording()

    def _save_recording(self) -> None:
        if self._recorder is None:
            return
        raw = self._manager.stop_recording()
        n_eeg = len(self._manager.eeg_channels)
        meta = {
            "sampling_rate": self._manager.sampling_rate,
            "board_id": self._manager.board_id,
            "eeg_channels": self._manager.eeg_channels,
            "channel_names": list(self._manager.channel_names[:n_eeg]),
            "channel_types": list(self._manager.channel_types[:n_eeg]),
            "electrodes": list(self._manager.electrodes[:n_eeg]),
            "placements": list(self._manager.placements[:n_eeg]),
            "marker_channel": self._manager.marker_channel,
            "notch_freq": self._config.notch_freq,
        }
        # Always keep the lossless backup (CSV + meta + marker log).
        self._recorder.save(raw, meta)
        self._recorder = None

        if raw is None or raw.ndim != 2 or raw.shape[1] == 0:
            QMessageBox.information(self, "Recording", "Recording was empty.")
            return

        # Open the review window so the user can browse, edit markers and export.
        types = list(getattr(self.marker_panel, "_types", []))
        code_labels = {mt.code: mt.label for mt in types}
        review = ReviewWindow(raw, meta, code_labels, types, title=self._config.name)
        review.show()
        self._reviews.append(review)

    def _on_marker(self, code: int, label: str, ts: float) -> None:
        self._manager.insert_marker(code)
        if self._recorder is not None:
            self._recorder.add_marker(MarkerEvent(timestamp=ts, code=code, label=label))

    def _on_exit(self) -> None:
        if self._manager.recording:
            res = QMessageBox.question(
                self, "Recording in progress",
                "A recording is active. Save and exit?",
            )
            if res == QMessageBox.StandardButton.Yes:
                self.record_btn.setChecked(False)
        self.exit_session.emit()

    def shutdown(self) -> None:
        self._timer.stop()
        self._manager.stop_acquisition()
        if self._manager.recording:
            self._manager.stop_recording()
