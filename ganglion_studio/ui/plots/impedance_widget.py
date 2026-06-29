"""Live electrode impedance bars + history (electrode characterization)."""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio import palette
from ganglion_studio.core import board_config as cfg
from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.dsp import FilterSettings
from ganglion_studio.ui import theme


class ImpedanceWidget(QWidget):
    # Impedance changes slowly; refreshing a few times per second is plenty.
    refresh_hz = 8.0

    def __init__(self, manager: BoardManager) -> None:
        super().__init__()
        self._manager = manager
        self._n = len(manager.eeg_channels)
        self._t0 = time.time()
        self._hist_t: List[float] = []
        self._hist_vals: List[List[float]] = [[] for _ in range(self._n)]

        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.toggle_btn = QPushButton("Start impedance test")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.toggled.connect(self._on_toggle)
        bar.addWidget(self.toggle_btn)
        self.status = QLabel("Impedance test off. EEG streaming continues.")
        self.status.setStyleSheet(theme.MUTED_QSS)
        bar.addWidget(self.status)
        bar.addStretch(1)
        root.addLayout(bar)

        legend = QLabel(
            f"Good < {cfg.IMPEDANCE_GOOD_KOHM:.0f} k\u03A9   "
            f"OK < {cfg.IMPEDANCE_OK_KOHM:.0f} k\u03A9   Bad above"
        )
        legend.setStyleSheet(theme.MUTED_QSS)
        root.addWidget(legend)

        glw = pg.GraphicsLayoutWidget()
        root.addWidget(glw, 1)

        self._bar_plot = glw.addPlot(row=0, col=0)
        self._bar_plot.setLabel("left", "Impedance", units="k\u03A9")
        self._bar_plot.setLabel("bottom", "Channel")
        self._bar_plot.getAxis("bottom").setTicks(
            [list(enumerate(self._manager.channel_names[: self._n]))]
        )
        self._bar_item = pg.BarGraphItem(
            x=list(range(self._n)), height=[0] * self._n, width=0.6, brush=palette.GOOD
        )
        self._bar_plot.addItem(self._bar_item)
        self._value_labels: List[pg.TextItem] = []
        for i in range(self._n):
            txt = pg.TextItem("", anchor=(0.5, 1.0))
            self._bar_plot.addItem(txt)
            self._value_labels.append(txt)

        self._hist_plot = glw.addPlot(row=1, col=0)
        self._hist_plot.setLabel("left", "Impedance", units="k\u03A9")
        self._hist_plot.setLabel("bottom", "Time", units="s")
        self._hist_plot.showGrid(x=True, y=True, alpha=0.2)
        self._hist_legend = self._hist_plot.addLegend()
        self._hist_curves: List[pg.PlotDataItem] = []
        for i in range(self._n):
            color = cfg.CHANNEL_COLORS[i % len(cfg.CHANNEL_COLORS)]
            self._hist_curves.append(
                self._hist_plot.plot(pen=pg.mkPen(color, width=1.5), name=self._manager.channel_names[i])
            )

    def set_channel_names(self, names: List[str]) -> None:
        self._bar_plot.getAxis("bottom").setTicks([list(enumerate(names[: self._n]))])
        for i, curve in enumerate(self._hist_curves):
            if i < len(names):
                self._hist_legend.removeItem(curve)
                self._hist_legend.addItem(curve, names[i])

    def _on_toggle(self, checked: bool) -> None:
        if checked:
            self._manager.start_impedance()
            self.toggle_btn.setText("Stop impedance test")
            self.status.setText("Impedance test running (data stream active)...")
        else:
            self._manager.stop_impedance()
            self.toggle_btn.setText("Start impedance test")
            self.status.setText("Impedance test off. EEG streaming continues.")

    def update_plot(self, settings: FilterSettings, active: List[bool]) -> None:
        vals = self._manager.latest_impedance_kohm()
        heights = [max(0.0, v) if v >= 0 else 0.0 for v in vals]
        colors = [cfg.impedance_color(v) for v in vals]
        self._bar_item.setOpts(height=heights, brushes=colors)

        ymax = max(heights + [cfg.IMPEDANCE_OK_KOHM]) * 1.2
        for i in range(self._n):
            label = f"{vals[i]:.1f}" if vals[i] >= 0 else "n/a"
            self._value_labels[i].setText(label)
            self._value_labels[i].setPos(i, heights[i] + ymax * 0.02)

        now = time.time() - self._t0
        self._hist_t.append(now)
        for i in range(self._n):
            self._hist_vals[i].append(heights[i])
        if len(self._hist_t) > 600:
            self._hist_t = self._hist_t[-600:]
            self._hist_vals = [v[-600:] for v in self._hist_vals]
        for i in range(self._n):
            self._hist_curves[i].setData(self._hist_t, self._hist_vals[i])
