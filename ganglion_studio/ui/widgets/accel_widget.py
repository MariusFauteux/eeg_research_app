"""Accelerometer / motion trace for movement-artifact monitoring."""

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

from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.dsp import FilterSettings

_AXES = ["X", "Y", "Z"]
_AXIS_COLORS = ["#f7766f", "#5fd38d", "#4f8ef7"]


class AccelWidget(QWidget):
    def __init__(self, manager: BoardManager) -> None:
        super().__init__()
        self._manager = manager
        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.enable_chk = QCheckBox("Accelerometer enabled")
        self.enable_chk.setChecked(manager.accel_enabled)
        self.enable_chk.toggled.connect(manager.set_accel)
        bar.addWidget(self.enable_chk)
        bar.addWidget(QLabel("Window"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(2.0, 30.0)
        self.window_spin.setValue(10.0)
        self.window_spin.setSuffix(" s")
        bar.addWidget(self.window_spin)
        bar.addStretch(1)
        root.addLayout(bar)

        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setLabel("left", "Acceleration")
        self._plot.addLegend()
        root.addWidget(self._plot, 1)

        self._curves: List[pg.PlotDataItem] = []
        for i, axis in enumerate(_AXES):
            self._curves.append(self._plot.plot(pen=pg.mkPen(_AXIS_COLORS[i], width=1.3), name=axis))

    def update_plot(self, settings: FilterSettings, active: List[bool]) -> None:
        seconds = self.window_spin.value()
        rows = self._manager.accel_channels[:3]
        if not rows:
            return
        data = self._manager.recent(seconds)
        if data.shape[1] == 0:
            return
        sr = self._manager.sampling_rate
        t = np.arange(data.shape[1]) / sr
        for i, row in enumerate(rows):
            if row < data.shape[0]:
                self._curves[i].setData(t, data[row, :])
