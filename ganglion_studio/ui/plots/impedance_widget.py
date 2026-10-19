"""Actual impedance measurements from a timed check within an EEG session."""

from __future__ import annotations

import time
from typing import List, Optional

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio import palette
from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.dsp import FilterSettings
from ganglion_studio.ui import theme
from ganglion_studio.ui.impedance_session_view import ImpedanceTimeSeries, STALE_AFTER_SECONDS


class ImpedanceWidget(QWidget):
    """Read-only view; the session controls own timed hardware mode changes."""

    refresh_hz = 4.0

    def __init__(self, manager: BoardManager) -> None:
        super().__init__()
        self._manager = manager
        self._channel_names = list(manager.impedance_channel_names)
        self._origin: Optional[float] = None
        layout = QVBoxLayout(self)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet(theme.MUTED_QSS)
        layout.addWidget(self.status)
        hint = QLabel("Each point is one received measurement, including REF. "
                      "Channels update at different times; results remain visible after the check.")
        hint.setWordWrap(True)
        hint.setStyleSheet(theme.HINT_QSS)
        layout.addWidget(hint)
        self.current_table = QTableWidget(len(self._channel_names), 4)
        self.current_table.setHorizontalHeaderLabels(["Channel", "Latest (kΩ)", "Last update", "State"])
        self.current_table.verticalHeader().hide()
        self.current_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.current_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.current_table.setMaximumHeight(205)
        layout.addWidget(self.current_table)
        self.time_series = ImpedanceTimeSeries(self._channel_names)
        layout.addWidget(self.time_series, 1)
        self.update_plot(FilterSettings(), [])

    def set_channel_names(self, names: List[str]) -> None:
        self._channel_names = [(names[i] if i < len(names) else f"Ch{i + 1}")
                               for i in range(4)] + ["REF"]
        self.time_series.set_channel_names(self._channel_names)
        self.update_plot(FilterSettings(), [])

    def update_plot(self, settings: FilterSettings, active: List[bool]) -> None:
        now = time.time()
        readings = self._manager.impedance_history()
        latest = {reading.channel: reading for reading in self._manager.latest_impedance_readings()}
        if readings and self._origin is None:
            self._origin = min(reading.timestamp for reading in readings)
        self.time_series.set_readings(readings, origin=self._origin if self._origin is not None else now)
        state = getattr(self._manager, "impedance_check_status", "idle")
        measuring = state == "measuring" or self._manager.impedance_mode
        started = getattr(self._manager, "impedance_check_started_at", None)
        for channel, name in enumerate(self._channel_names):
            reading = latest.get(channel)
            if reading is None:
                cells, color = [name, "—", "—", "Waiting"], palette.MUTED
            else:
                age = max(0.0, now - reading.timestamp)
                if started is not None and reading.timestamp < started:
                    freshness = "Previous check"
                elif measuring:
                    freshness = "Stale" if age >= STALE_AFTER_SECONDS else "Current"
                else:
                    freshness = "Last check"
                cells = [name, f"{reading.ohms / 1000.0:.2f}", f"{age:.1f} s ago", freshness]
                color = palette.OK if freshness == "Stale" else palette.FG
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setForeground(QColor(color))
                self.current_table.setItem(channel, column, item)
        if state == "starting":
            text = "Starting impedance check… EEG is temporarily paused."
        elif state == "restoring":
            text = "Restoring EEG… Impedance measurements remain visible below."
        elif state == "restore_failed":
            text = "EEG has not resumed. Follow the message in the session controls."
        elif measuring:
            text = "Impedance check running. EEG resumes when the timed check ends."
        elif readings:
            text = "Last impedance measurements shown below. Use Get impedance while recording for another check."
        else:
            text = "Use Get impedance while recording to run a timed check. Results will appear here."
        self.status.setText(text)
