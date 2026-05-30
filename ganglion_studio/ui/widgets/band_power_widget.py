"""Average EEG band powers (delta/theta/alpha/beta/gamma) across active channels."""

from __future__ import annotations

from typing import List

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.dsp import FilterSettings, compute_band_powers

_BAND_COLORS = ["#b48ef7", "#4f8ef7", "#5fd38d", "#e2c044", "#f7766f"]


class BandPowerWidget(QWidget):
    refresh_hz = 4.0

    def __init__(self, manager: BoardManager) -> None:
        super().__init__()
        self._manager = manager
        root = QVBoxLayout(self)
        title = QLabel("Relative band power, averaged over active channels (last 4 s)")
        title.setStyleSheet("color:#9aa0aa;")
        root.addWidget(title)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("left", "Power")
        names, _ = compute_band_powers(np.zeros((1, 1)), manager.sampling_rate)
        self._names = names
        self._plot.getAxis("bottom").setTicks([list(enumerate(self._names))])
        self._bar = pg.BarGraphItem(
            x=list(range(len(self._names))), height=[0] * len(self._names),
            width=0.6, brushes=_BAND_COLORS,
        )
        self._plot.addItem(self._bar)
        root.addWidget(self._plot, 1)

    def update_plot(self, settings: FilterSettings, active: List[bool]) -> None:
        data = self._manager.recent_eeg(4.0)
        if data.shape[1] == 0:
            return
        rows = [i for i in range(data.shape[0]) if (i >= len(active) or active[i])]
        if not rows:
            return
        names, values = compute_band_powers(data[rows, :], self._manager.sampling_rate)
        self._bar.setOpts(height=list(values))
