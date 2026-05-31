"""Live signal-statistics panel for the session view's left column."""

from __future__ import annotations

from typing import List

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ganglion_studio.core import board_config as cfg
from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.dsp import FilterSettings, apply_filters, channel_stats

_QUALITY_COLORS = {"good": "#5fd38d", "ok": "#e2c044", "bad": "#f7766f"}
_COLUMNS = ["Ch", "RMS", "P-P", "Std", "Dom", "Line", "Q"]


class StatsPanel(QGroupBox):
    # Statistics move slowly; a few updates per second is plenty.
    refresh_hz = 4.0

    def __init__(self, manager: BoardManager, window_seconds: float = 2.0) -> None:
        super().__init__("Signal statistics")
        self._manager = manager
        self._window = window_seconds
        self._n = len(manager.eeg_channels)

        root = QVBoxLayout(self)

        self.table = QTableWidget(self._n, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setToolTip(
            "RMS/P-P/Std in uV, Dom = dominant frequency (Hz), "
            "Line = mains-noise fraction, Q = contact quality"
        )
        for i in range(self._n):
            name = manager.channel_names[i] if i < len(manager.channel_names) else f"Ch{i+1}"
            item = QTableWidgetItem(name)
            item.setForeground(QColor(cfg.CHANNEL_COLORS[i % len(cfg.CHANNEL_COLORS)]))
            self.table.setItem(i, 0, item)
            for c in range(1, len(_COLUMNS)):
                self.table.setItem(i, c, QTableWidgetItem("-"))
        self.table.setFixedHeight(28 + 30 * self._n)
        root.addWidget(self.table)

        self.general = QLabel("Waiting for data...")
        self.general.setWordWrap(True)
        self.general.setStyleSheet("color:#9aa0aa; font-size:11px;")
        root.addWidget(self.general)

    def set_channel_names(self, names: List[str]) -> None:
        for i in range(min(self.table.rowCount(), len(names))):
            self.table.item(i, 0).setText(names[i])

    def update_stats(self, settings: FilterSettings, active: List[bool]) -> None:
        data = self._manager.recent_eeg(self._window)
        sr = self._manager.sampling_rate
        n = data.shape[1]
        if n == 0:
            return

        rms_values = []
        for i in range(self._n):
            is_active = active[i] if i < len(active) else True
            if not is_active:
                self._set_row(i, None)
                continue
            filtered = apply_filters(data[i], sr, settings)
            stats = channel_stats(filtered, sr)
            self._set_row(i, stats)
            rms_values.append(stats["rms"])

        mean_rms = float(np.mean(rms_values)) if rms_values else 0.0
        rec = "recording" if self._manager.recording else "idle"
        secs = n / sr if sr else 0.0
        self.general.setText(
            f"Fs {sr} Hz | window {secs:0.1f}s ({n} samp) | "
            f"buffer {self._manager._filled} samp | mean RMS {mean_rms:0.1f} uV | {rec}"
        )

    def _set_row(self, row: int, stats) -> None:
        if stats is None:
            for c in range(1, len(_COLUMNS)):
                self.table.item(row, c).setText("off")
                self.table.item(row, c).setForeground(QColor("#666"))
            return
        values = [
            f"{stats['rms']:.1f}",
            f"{stats['ptp']:.0f}",
            f"{stats['std']:.1f}",
            f"{stats['dominant_hz']:.1f}",
            f"{stats['line_ratio'] * 100:.0f}%",
            stats["quality"],
        ]
        for c, text in enumerate(values, start=1):
            item = self.table.item(row, c)
            item.setText(text)
            item.setForeground(QColor("#e6e6e6"))
        q_item = self.table.item(row, len(_COLUMNS) - 1)
        q_item.setForeground(QColor(_QUALITY_COLORS.get(stats["quality"], "#e6e6e6")))
