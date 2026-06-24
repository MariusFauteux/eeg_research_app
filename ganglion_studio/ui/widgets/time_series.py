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

from ganglion_studio import palette
from ganglion_studio.core import board_config as cfg
from ganglion_studio.ui import theme
from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.dsp import (
    FILTER_WARMUP_SEC,
    FilterSettings,
    apply_filters_windowed,
)


class TimeSeriesWidget(QWidget):
    def __init__(self, manager: BoardManager) -> None:
        super().__init__()
        self._manager = manager
        self._n = len(manager.eeg_channels)

        root = QVBoxLayout(self)
        root.addLayout(self._build_controls())

        self._glw = pg.GraphicsLayoutWidget()
        root.addWidget(self._glw, 1)

        # Scrolling/zooming is only allowed while the stream is paused; the live
        # view auto-follows "now" and ignores mouse input until set_paused(True).
        self._paused = False

        self._plots: List[pg.PlotItem] = []
        self._curves: List[pg.PlotDataItem] = []
        for i in range(self._n):
            p = self._glw.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.2)
            p.setLabel("left", self._manager.channel_names[i], units="uV")
            p.setMouseEnabled(x=False, y=False)  # enabled only while paused
            if i < self._n - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            else:
                p.setLabel("bottom", "Time", units="s")
            color = cfg.CHANNEL_COLORS[i % len(cfg.CHANNEL_COLORS)]
            curve = p.plot(pen=pg.mkPen(color, width=1))
            self._plots.append(p)
            self._curves.append(curve)
        # Don't let setData auto-range the time axis (we drive it via Follow),
        # and X-link channels so touchpad zoom/pan on one moves them together.
        for p in self._plots:
            p.getViewBox().enableAutoRange(x=False)
        for p in self._plots[1:]:
            p.setXLink(self._plots[0])
        # Marker overlay: (line, plot) pairs are reused across frames; we only
        # rebuild the pool when the marker count or visible-plot set changes.
        self._marker_lines: List = []
        self._marker_state = None

    def set_channel_names(self, names: List[str]) -> None:
        for i, p in enumerate(self._plots):
            if i < len(names):
                p.setLabel("left", names[i], units="uV")

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

        self.scroll_hint = QLabel("Pause the stream to scroll/zoom")
        self.scroll_hint.setStyleSheet(theme.HINT_QSS)
        bar.addWidget(self.scroll_hint)
        bar.addStretch(1)
        return bar

    def update_plot(self, settings: FilterSettings, active: List[bool]) -> None:
        # While paused the user is free to scroll/zoom, so leave the frozen view
        # untouched; live updates resume (and re-pin the view) on un-pause.
        if self._paused:
            return
        seconds = self.window_spin.value()
        eeg_rows = self._manager.eeg_channels
        sr = self._manager.sampling_rate
        amp = self.amp_spin.value()
        # Fetch extra past samples as filter warm-up, then only display the most
        # recent `seconds`. This keeps the band-pass from "bending" at the edges.
        full = self._manager.recent(seconds + FILTER_WARMUP_SEC)
        n_vis = min(full.shape[1], int(round(seconds * sr)))
        vis = full[:, -n_vis:] if n_vis else full[:, :0]
        # Newest sample sits at t=0 (right edge); older samples are negative
        # seconds in the past, scrolling left — matches the OpenBCI GUI.
        t = (np.arange(n_vis) - (n_vis - 1)) / sr if n_vis else np.array([])

        for i in range(self._n):
            visible = active[i] if i < len(active) else True
            self._plots[i].setVisible(visible)
            if not visible or n_vis == 0:
                self._curves[i].clear()
                continue
            filtered = apply_filters_windowed(full[eeg_rows[i]], sr, settings, n_vis)
            self._curves[i].setData(t, filtered)
            if self.autoscale.isChecked():
                self._plots[i].enableAutoRange(axis="y")
            else:
                self._plots[i].setYRange(-amp, amp, padding=0)

        # Pin the right edge at 0 ("now"); X-linked plots all follow.
        self._plots[0].setXRange(-seconds, 0, padding=0)

        marker_row = None
        mch = self._manager.marker_channel
        if n_vis and mch < vis.shape[0]:
            marker_row = vis[mch]
        self._draw_markers(t, sr, marker_row)

    def set_paused(self, paused: bool) -> None:
        """Allow trackpad/mouse scroll & zoom only while the stream is paused.
        On resume the next update_plot re-pins the live window ("now" at the
        right edge), so the view automatically returns to normal."""
        self._paused = paused
        for p in self._plots:
            p.setMouseEnabled(x=paused, y=paused)

    def _clear_marker_lines(self) -> None:
        for line, plot in self._marker_lines:
            plot.removeItem(line)
        self._marker_lines = []
        self._marker_state = None

    def _draw_markers(self, t: np.ndarray, sr: int, marker_row) -> None:
        if not self.show_markers.isChecked() or marker_row is None or t.size == 0:
            if self._marker_lines:
                self._clear_marker_lines()
            return
        # Match the time vector: newest sample at 0, older samples negative.
        xs = (np.flatnonzero(marker_row != 0) - (t.size - 1)) / sr
        visible = tuple(i for i in range(self._n) if self._plots[i].isVisible())
        state = (len(xs), visible)
        if state != self._marker_state:
            self._clear_marker_lines()
            for _x in xs:
                for pi in visible:
                    line = pg.InfiniteLine(
                        angle=90,
                        pen=pg.mkPen(palette.MARKER, width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
                    )
                    self._plots[pi].addItem(line)
                    self._marker_lines.append((line, self._plots[pi]))
            self._marker_state = state
        k = 0
        for x in xs:
            for _pi in visible:
                self._marker_lines[k][0].setValue(float(x))
                k += 1
