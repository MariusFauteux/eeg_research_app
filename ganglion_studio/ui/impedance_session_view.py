"""Impedance-only capture and review using timestamped board measurements."""

from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio import palette
from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.impedance import ImpedanceReading
from ganglion_studio.core.impedance_recording import (
    ImpedanceRecorder,
    LoadedImpedanceRecording,
    load_impedance_recording,
)
from ganglion_studio.core.session import SessionConfig
from ganglion_studio.ui import theme
from ganglion_studio.ui.channel_setup_dialog import ChannelSetupDialog


STALE_AFTER_SECONDS = 10.0


def _table(headers: List[str], rows: int) -> QTableWidget:
    table = QTableWidget(rows, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().hide()
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    return table


class ImpedanceTimeSeries(QWidget):
    """Plot actual irregular updates; repainting never creates measurements."""

    def __init__(self, channel_names: List[str], parent=None) -> None:
        super().__init__(parent)
        self._channel_names = list(channel_names)
        self._readings: List[ImpedanceReading] = []
        self._origin = 0.0
        self._now: Optional[float] = None
        self._window: Optional[float] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        visibility = QHBoxLayout()
        visibility.addWidget(QLabel("Show"))
        self.channel_checks: List[QCheckBox] = []
        for index, name in enumerate(channel_names):
            check = QCheckBox(name)
            check.setChecked(True)
            check.setStyleSheet(f"color: {self._color(index)};")
            check.toggled.connect(self._redraw)
            self.channel_checks.append(check)
            visibility.addWidget(check)
        visibility.addStretch(1)
        layout.addLayout(visibility)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "Impedance", units="kΩ")
        self.plot.setLabel("bottom", "Elapsed time", units="s")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.legend = self.plot.addLegend()
        self.curves = [self.plot.plot(
            pen=pg.mkPen(self._color(index), width=1.5), name=name,
            symbol="o", symbolSize=5, symbolBrush=self._color(index),
            symbolPen=None,
        ) for index, name in enumerate(channel_names)]
        layout.addWidget(self.plot, 1)

    def set_channel_names(self, names: List[str]) -> None:
        self._channel_names = list(names)
        for index, name in enumerate(names[:len(self.curves)]):
            self.channel_checks[index].setText(name)
            self.legend.removeItem(self.curves[index])
            self.legend.addItem(self.curves[index], name)

    @staticmethod
    def _color(index: int) -> str:
        return (palette.CHANNEL_COLORS + [palette.VIOLET])[index % 5]

    def set_readings(self, readings: List[ImpedanceReading], *, origin: float,
                     now: Optional[float] = None,
                     window_seconds: Optional[float] = None) -> None:
        self._readings = list(readings)
        self._origin, self._now, self._window = origin, now, window_seconds
        self._redraw()

    def _redraw(self, *_args) -> None:
        for channel, curve in enumerate(self.curves):
            if not self.channel_checks[channel].isChecked():
                curve.setData([], [])
                continue
            readings = [r for r in self._readings if r.channel == channel]
            curve.setData([r.timestamp - self._origin for r in readings],
                          [r.ohms / 1000.0 for r in readings])
        if self._now is not None and self._window is not None:
            end = max(1.0, self._now - self._origin)
            self.plot.setXRange(max(0.0, end - self._window), end, padding=0)


class ImpedanceRecordingWindow(QWidget):
    """Inspect an impedance CSV with elapsed times and per-channel statistics."""

    def __init__(self, recording: LoadedImpedanceRecording | str, parent=None) -> None:
        super().__init__(parent)
        self.recording = (load_impedance_recording(recording)
                          if isinstance(recording, (str, os.PathLike)) else recording)
        rec = self.recording
        self.setWindowTitle(f"Impedance recording — {os.path.basename(rec.source_path)}")
        self.resize(1100, 760)
        layout = QVBoxLayout(self)
        title = QLabel("Recorded impedance")
        title.setStyleSheet("font-size:17px; font-weight:700;")
        layout.addWidget(title)
        info = QLabel(f"{len(rec.readings)} measurements · {rec.duration_s:.1f} s · "
                      f"{rec.source_path}")
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.setWordWrap(True)
        layout.addWidget(info)
        hint = QLabel("Each point is one received measurement. Channels update at different times. "
                      "Drag or scroll the plot to inspect the recording.")
        hint.setStyleSheet(theme.MUTED_QSS)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.time_series = ImpedanceTimeSeries(rec.channel_names)
        self.time_series.set_readings(rec.readings, origin=rec.started_at)
        layout.addWidget(self.time_series, 1)
        self.summary_table = _table(
            ["Channel", "Measurements", "Minimum (kΩ)", "Median (kΩ)", "Maximum (kΩ)"],
            len(rec.channel_names),
        )
        self.summary_table.setMaximumHeight(205)
        for channel, name in enumerate(rec.channel_names):
            values = np.array([r.ohms / 1000.0 for r in rec.readings if r.channel == channel])
            cells = [name, str(values.size)] + (
                [f"{v:.2f}" for v in (values.min(), np.median(values), values.max())]
                if values.size else ["—", "—", "—"])
            for column, text in enumerate(cells):
                self.summary_table.setItem(channel, column, QTableWidgetItem(text))
        layout.addWidget(self.summary_table)


class ImpedanceSessionView(QWidget):
    """Capture resistance updates independently of rendering or plot visibility."""

    exit_session = pyqtSignal()
    open_processing = pyqtSignal()

    def __init__(self, manager: BoardManager, config: SessionConfig,
                 base_dir: str = "recordings") -> None:
        super().__init__()
        self._manager, self._config, self._base_dir = manager, config, base_dir
        self._channel_names = list(manager.impedance_channel_names)
        self._recorder: Optional[ImpedanceRecorder] = None
        self._pending_recording: Optional[List[ImpedanceReading]] = None
        self._pending_metadata: Optional[dict] = None
        self._history_origin: Optional[float] = None
        self._reviews: List[ImpedanceRecordingWindow] = []
        self._last_saved_paths: List[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        toolbar = QHBoxLayout()
        back = QPushButton("◀ Dashboard")
        back.clicked.connect(self.exit_session.emit)
        toolbar.addWidget(back)
        title = QLabel(config.name)
        title.setStyleSheet("font-weight:700; font-size:15px;")
        toolbar.addWidget(title)
        mode = QLabel("IMPEDANCE ONLY" + (" · DEMO" if config.demo else ""))
        mode.setStyleSheet(f"color:{palette.ACCENT}; font-weight:700;")
        toolbar.addWidget(mode)
        toolbar.addStretch(1)
        self.setup_btn = QPushButton("Channel setup")
        self.setup_btn.clicked.connect(self._open_channel_setup)
        toolbar.addWidget(self.setup_btn)
        open_csv = QPushButton("Open impedance CSV…")
        open_csv.clicked.connect(self._open_csv)
        toolbar.addWidget(open_csv)
        self.record_btn = QPushButton("● Record impedance")
        self.record_btn.setCheckable(True)
        self.record_btn.setStyleSheet("QPushButton:checked { background:#b03434; }")
        self.record_btn.toggled.connect(self._on_record)
        toolbar.addWidget(self.record_btn)
        layout.addLayout(toolbar)

        hint = QLabel("Each point is one impedance update received from the board. "
                      "Channels update at different times; no EEG is recorded in this session.")
        hint.setWordWrap(True)
        hint.setStyleSheet(theme.MUTED_QSS)
        layout.addWidget(hint)
        basis = getattr(manager, "impedance_raw_unit_basis", config.impedance_units)
        unit = "kΩ" if basis == "kohms" else "Ω"
        self.units_hint = QLabel(f"Board readout: {unit} (selected at session setup). "
                                "Charts show kΩ; CSV also includes Ω.")
        self.units_hint.setStyleSheet(theme.HINT_QSS)
        layout.addWidget(self.units_hint)
        self.save_status = QLabel()
        self.save_status.setWordWrap(True)
        self.save_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.save_status.hide()
        layout.addWidget(self.save_status)

        self.current_table = _table(
            ["Channel", "Latest (kΩ)", "Last update", "State"], len(self._channel_names))
        self.current_table.setMaximumHeight(205)
        layout.addWidget(self.current_table)
        controls = QHBoxLayout()
        self.status_label = QLabel("Waiting for impedance measurements…")
        controls.addWidget(self.status_label, 1)
        controls.addWidget(QLabel("Rolling window"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(10, 3600)
        self.window_spin.setValue(60)
        self.window_spin.setSuffix(" s")
        self.window_spin.valueChanged.connect(self._tick)
        controls.addWidget(self.window_spin)
        layout.addLayout(controls)
        self.time_series = ImpedanceTimeSeries(self._channel_names)
        layout.addWidget(self.time_series, 1)

        # The producer captures every update, including when this widget or its
        # channels are hidden. The timer only takes snapshots for presentation.
        manager.start_acquisition()
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._tick()

    def _tick(self, *_args) -> None:
        now = time.time()
        readings = self._manager.impedance_history(seconds=self.window_spin.value())
        if readings and self._history_origin is None:
            self._history_origin = min(reading.timestamp for reading in readings)
        self.time_series.set_readings(readings, origin=(self._history_origin
                                      if self._history_origin is not None else now),
                                      now=now, window_seconds=self.window_spin.value())
        latest = {r.channel: r for r in self._manager.latest_impedance_readings()}
        disconnected = self._manager.is_disconnected()
        for channel, name in enumerate(self._channel_names):
            reading = latest.get(channel)
            if reading is None:
                cells, color = [name, "—", "—", "Waiting"], palette.MUTED
            else:
                age = max(0.0, now - reading.timestamp)
                state = ("Disconnected" if disconnected else
                         "Stale" if age >= STALE_AFTER_SECONDS else "Current")
                cells = [name, f"{reading.ohms / 1000.0:.2f}", f"{age:.1f} s ago", state]
                color = palette.OK if state != "Current" else palette.FG
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setForeground(QColor(color))
                self.current_table.setItem(channel, column, item)
        if disconnected:
            self.status_label.setText("Disconnected — return to the Dashboard to reconnect.")
        elif self._pending_recording is not None:
            self.status_label.setText("Measurements continue · recording stopped · save needs retry")
        elif self._manager.recording:
            self.status_label.setText(
                f"Recording · {self._manager.recorded_sample_count()} measurements")
        else:
            self.status_label.setText(f"Measuring · {len(readings)} updates in the rolling window")

    def _set_record_button(self, checked: bool, text: str) -> None:
        self.record_btn.blockSignals(True)
        self.record_btn.setChecked(checked)
        self.record_btn.setText(text)
        self.record_btn.blockSignals(False)

    def _open_channel_setup(self) -> None:
        if self._recorder is not None or self._pending_recording is not None:
            return
        dialog = ChannelSetupDialog(self._manager, self)
        if dialog.exec() != ChannelSetupDialog.DialogCode.Accepted:
            return
        self._manager.set_channel_config(dialog.names(), dialog.types(),
                                         dialog.electrodes(), dialog.placements())
        self._channel_names = list(self._manager.impedance_channel_names)
        self.time_series.set_channel_names(self._channel_names)
        self._tick()

    def _on_record(self, recording: bool) -> None:
        if self._pending_recording is not None:
            self._save_recording()
            return
        if not recording:
            self._save_recording()
            return
        recorder = ImpedanceRecorder(self._config, base_dir=self._base_dir)
        try:
            recorder.begin()
            self._manager.start_recording()
        except Exception as exc:
            self._set_record_button(False, "● Record impedance")
            QMessageBox.warning(self, "Could not start recording", str(exc))
            return
        self._recorder = recorder
        self.setup_btn.setEnabled(False)
        self._set_record_button(True, "■ Stop and save")
        self.save_status.hide()
        self._tick()

    def _save_recording(self, show_review: bool = True, raw_data=None) -> bool:
        if self._recorder is None:
            return True
        if self._pending_recording is None:
            try:
                self._pending_recording = list(self._manager.stop_recording()
                                               if raw_data is None else raw_data)
            except Exception as exc:
                self._set_record_button(True, "■ Stop and save")
                QMessageBox.warning(self, "Could not finish recording",
                                    f"The recording is still retained. Try stopping again.\n\n{exc}")
                return False
            self._pending_metadata = {
                "acquisition_mode": "impedance",
                "board_id": self._manager.board_id,
                "demo": self._config.demo,
                "impedance_raw_unit_basis": getattr(self._manager, "impedance_raw_unit_basis",
                                                     self._config.impedance_units),
                "channel_types": list(getattr(self._manager, "channel_types", [])),
                "electrodes": list(getattr(self._manager, "electrodes", [])),
                "placements": list(getattr(self._manager, "placements", [])),
            }
        readings = self._pending_recording
        try:
            paths = self._recorder.save(readings, self._channel_names,
                                         metadata=self._pending_metadata)
        except Exception as exc:
            self._set_record_button(True, "Retry save")
            self.save_status.setText("Save failed. Measurements are held in memory; "
                                     "click Retry save before leaving this session.")
            self.save_status.setStyleSheet(f"color:{palette.BAD};")
            self.save_status.show()
            QMessageBox.warning(self, "Save failed",
                                f"The recording is retained for another save attempt.\n\n{exc}")
            return False
        self._last_saved_paths = list(paths)
        self._recorder = None
        self._pending_recording = None
        self._pending_metadata = None
        self.setup_btn.setEnabled(True)
        self._set_record_button(False, "● Record impedance")
        self.save_status.setText(f"Saved {len(readings)} measurements: {paths[0]}")
        self.save_status.setStyleSheet(theme.MUTED_QSS)
        self.save_status.show()
        self._tick()
        if show_review and readings:
            self._show_review(paths[0])
        return True

    def _show_review(self, path: str) -> None:
        try:
            review = ImpedanceRecordingWindow(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not open impedance recording", str(exc))
            return
        review.show()
        self._reviews.append(review)

    def _open_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open impedance recording", self._base_dir,
            "Impedance CSV (*_impedance*.csv);;CSV (*.csv)")
        if path:
            self._show_review(path)

    def quiesce_for_shutdown(self) -> None:
        self._timer.stop()
        self.setEnabled(False)

    def resume_after_failed_shutdown(self) -> None:
        self.setEnabled(True)
        self._manager.start_acquisition()
        self._timer.start()

    def shutdown(self, raw_data=None) -> bool:
        if not self._save_recording(show_review=False, raw_data=raw_data):
            return False
        self._timer.stop()
        self._manager.stop_acquisition(timeout=0.0)
        for review in self._reviews:
            review.close()
        return True
