"""Live multi-channel time-series plot with per-plot controls."""

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
from ganglion_studio.core.dsp import FilterSettings, apply_filters


class TimeSeriesWidget(QWidget):
    def __init__(self, manager: BoardManager) -> None:
        super().__init__()
        self._manager = manager
        self._n = len(manager.eeg_channels)

        root = QVBoxLayout(self)
        root.addLayout(self._build_controls())

        self._glw = pg.GraphicsLayoutWidget()
        root.addWidget(self._glw, 1)

        self._plots: List[pg.PlotItem] = []
        self._curves: List[pg.PlotDataItem] = []
        for i in range(self._n):
            p = self._glw.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.2)
            p.setLabel("left", cfg.DEFAULT_CHANNEL_NAMES[i], units="uV")
            p.setMouseEnabled(x=False, y=True)
            if i < self._n - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            else:
                p.setLabel("bottom", "Time", units="s")
            color = cfg.CHANNEL_COLORS[i % len(cfg.CHANNEL_COLORS)]
            curve = p.plot(pen=pg.mkPen(color, width=1))
            self._plots.append(p)
            self._curves.append(curve)
        self._marker_lines: List[pg.InfiniteLine] = []

    def _build_controls(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Window"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(1.0, 30.0)
        self.window_spin.setValue(5.0)
        self.window_spin.setSuffix(" s")
        bar.addWidget(self.window_spin)

        bar.addWidget(QLabel("Amplitude"))
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(1.0, 100000.0)
        self.amp_spin.setValue(200.0)
        self.amp_spin.setSuffix(" uV")
        bar.addWidget(self.amp_spin)

        self.autoscale = QCheckBox("Auto-scale")
        bar.addWidget(self.autoscale)

        self.show_markers = QCheckBox("Markers")
        self.show_markers.setChecked(True)
        bar.addWidget(self.show_markers)
        bar.addStretch(1)
        return bar

    def update_plot(self, settings: FilterSettings, active: List[bool]) -> None:
        seconds = self.window_spin.value()
        data = self._manager.recent_eeg(seconds)
        sr = self._manager.sampling_rate
        amp = self.amp_spin.value()
        n = data.shape[1]
        t = np.arange(n) / sr if n else np.array([])

        for i in range(self._n):
            visible = active[i] if i < len(active) else True
            self._plots[i].setVisible(visible)
            if not visible or n == 0:
                self._curves[i].clear()
                continue
            filtered = apply_filters(data[i], sr, settings)
            self._curves[i].setData(t, filtered)
            if self.autoscale.isChecked():
                self._plots[i].enableAutoRange(axis="y")
            else:
                self._plots[i].setYRange(-amp, amp, padding=0)
            if n:
                self._plots[i].setXRange(t[0], t[-1], padding=0)

        self._draw_markers(seconds, t)

    def _draw_markers(self, seconds: float, t: np.ndarray) -> None:
        for line in self._marker_lines:
            for p in self._plots:
                p.removeItem(line)
        self._marker_lines.clear()
        if not self.show_markers.isChecked() or t.size == 0:
            return
        idx, _codes = self._manager.recent_markers(seconds)
        sr = self._manager.sampling_rate
        n = t.size
        for mi in idx:
            x = mi / sr
            for p in self._plots:
                if not p.isVisible():
                    continue
                line = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen("#f7766f", width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
                p.addItem(line)
                self._marker_lines.append(line)
