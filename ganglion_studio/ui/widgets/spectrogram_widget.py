"""Diagnostics: rolling spectrogram + instantaneous FFT for one channel."""

from __future__ import annotations

from typing import List

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio.core import board_config as cfg
from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.dsp import (
    FilterSettings,
    apply_filters,
    compute_fft,
    compute_spectrogram,
)


class SpectrogramWidget(QWidget):
    # Rolling spectrogram + FFT is the heaviest tab; a few fps is plenty.
    refresh_hz = 4.0

    def __init__(self, manager: BoardManager) -> None:
        super().__init__()
        self._manager = manager
        self._n = len(manager.eeg_channels)
        self._levels = None
        self._level_frames = 0

        root = QVBoxLayout(self)
        root.addLayout(self._build_controls())

        glw = pg.GraphicsLayoutWidget()
        root.addWidget(glw, 1)

        self._spec_plot = glw.addPlot(row=0, col=0)
        self._spec_plot.setLabel("left", "Frequency", units="Hz")
        self._spec_plot.setLabel("bottom", "Time", units="s")
        self._img = pg.ImageItem()
        self._spec_plot.addItem(self._img)
        self._cmap = pg.colormap.get("viridis")
        self._img.setLookupTable(self._cmap.getLookupTable())

        self._fft_plot = glw.addPlot(row=1, col=0)
        self._fft_plot.setLabel("left", "Magnitude", units="uV")
        self._fft_plot.setLabel("bottom", "Frequency", units="Hz")
        self._fft_plot.showGrid(x=True, y=True, alpha=0.2)
        self._fft_curve = self._fft_plot.plot(pen=pg.mkPen("#4f8ef7", width=1.5))

    def _build_controls(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Channel"))
        self.ch_combo = QComboBox()
        for i in range(self._n):
            self.ch_combo.addItem(cfg.DEFAULT_CHANNEL_NAMES[i])
        bar.addWidget(self.ch_combo)

        bar.addWidget(QLabel("Window"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(2.0, 30.0)
        self.window_spin.setValue(10.0)
        self.window_spin.setSuffix(" s")
        bar.addWidget(self.window_spin)

        bar.addWidget(QLabel("Max freq"))
        self.fmax_spin = QDoubleSpinBox()
        self.fmax_spin.setRange(5.0, 100.0)
        self.fmax_spin.setValue(60.0)
        self.fmax_spin.setSuffix(" Hz")
        bar.addWidget(self.fmax_spin)
        bar.addStretch(1)
        return bar

    def update_plot(self, settings: FilterSettings, active: List[bool]) -> None:
        ch = self.ch_combo.currentIndex()
        if ch >= self._n:
            return
        seconds = self.window_spin.value()
        data = self._manager.recent_eeg(seconds)
        sr = self._manager.sampling_rate
        if data.shape[1] < 64:
            return
        filtered = apply_filters(data[ch], sr, settings)
        fmax = self.fmax_spin.value()

        freqs, times, sxx = compute_spectrogram(filtered, sr)
        if sxx.size:
            fmask = freqs <= fmax
            img = sxx[fmask, :].T  # time x freq
            # Recompute colour levels only occasionally - a full min/max scan
            # plus LUT rebuild (autoLevels) every frame is the main cost here.
            if self._levels is None or self._level_frames % 16 == 0:
                self._levels = (float(img.min()), float(img.max()))
            self._level_frames += 1
            self._img.setImage(img, autoLevels=False, levels=self._levels)
            f_sel = freqs[fmask]
            if times.size and f_sel.size:
                self._img.setRect(pg.QtCore.QRectF(0, 0, float(times[-1]), float(f_sel[-1])))

        ffreqs, fmag = compute_fft(filtered, sr)
        if ffreqs.size:
            m = ffreqs <= fmax
            self._fft_curve.setData(ffreqs[m], fmag[m])
