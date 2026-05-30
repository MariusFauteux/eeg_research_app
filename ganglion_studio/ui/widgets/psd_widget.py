"""Live power spectral density (Welch) plot with controls."""

from __future__ import annotations

from typing import List

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio.core import board_config as cfg
from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.dsp import FilterSettings, apply_filters, compute_psd


class PSDWidget(QWidget):
    # Spectral estimate is expensive and slow-moving; no need for 30 fps.
    refresh_hz = 5.0

    def __init__(self, manager: BoardManager) -> None:
        super().__init__()
        self._manager = manager
        self._n = len(manager.eeg_channels)

        root = QVBoxLayout(self)
        root.addLayout(self._build_controls())

        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("bottom", "Frequency", units="Hz")
        self._plot.setLabel("left", "PSD", units="uV^2/Hz")
        self._plot.addLegend()
        root.addWidget(self._plot, 1)

        self._curves: List[pg.PlotDataItem] = []
        for i in range(self._n):
            color = cfg.CHANNEL_COLORS[i % len(cfg.CHANNEL_COLORS)]
            self._curves.append(
                self._plot.plot(pen=pg.mkPen(color, width=1.5), name=cfg.DEFAULT_CHANNEL_NAMES[i])
            )
        self._apply_log()

    def _build_controls(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Window"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(1.0, 20.0)
        self.window_spin.setValue(4.0)
        self.window_spin.setSuffix(" s")
        bar.addWidget(self.window_spin)

        bar.addWidget(QLabel("Max freq"))
        self.fmax_spin = QDoubleSpinBox()
        self.fmax_spin.setRange(5.0, 100.0)
        self.fmax_spin.setValue(60.0)
        self.fmax_spin.setSuffix(" Hz")
        self.fmax_spin.valueChanged.connect(self._apply_range)
        bar.addWidget(self.fmax_spin)

        self.log_y = QCheckBox("Log Y")
        self.log_y.setChecked(True)
        self.log_y.toggled.connect(self._apply_log)
        bar.addWidget(self.log_y)
        bar.addStretch(1)
        return bar

    def _apply_log(self, *_args) -> None:
        self._plot.setLogMode(x=False, y=self.log_y.isChecked())

    def _apply_range(self, *_args) -> None:
        self._plot.setXRange(0, self.fmax_spin.value(), padding=0)

    def update_plot(self, settings: FilterSettings, active: List[bool]) -> None:
        seconds = self.window_spin.value()
        data = self._manager.recent_eeg(seconds)
        sr = self._manager.sampling_rate
        fmax = self.fmax_spin.value()
        for i in range(self._n):
            visible = active[i] if i < len(active) else True
            if not visible or data.shape[1] == 0:
                self._curves[i].clear()
                continue
            filtered = apply_filters(data[i], sr, settings)
            freqs, amps = compute_psd(filtered, sr)
            if freqs.size == 0:
                self._curves[i].clear()
                continue
            mask = freqs <= fmax
            self._curves[i].setData(freqs[mask], amps[mask])
