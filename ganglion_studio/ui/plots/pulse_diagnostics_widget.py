"""Live 'Diagnostics' tab: characterize the once-per-second pulse artifact.

A pure consumer of :class:`BoardManager`'s ring buffer. Each refresh it runs
``core.pulse_diagnostics`` on the strongest active channel and shows -- in real
time, while the user touches the laptop / swaps electrodes -- whether the 1 Hz
pulse is a BLE packet-loss seam (digital, fixable in code) or a coupled analog
event (shielding / power). A "Mark event" button drops a marker so the pulse's
before/after is visible on the timeline.

Follows the plot-tab contract used across ``ui/plots/`` (an ``update_plot``
method + a ``refresh_hz`` throttle read by ``SessionView._due``).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio import palette
from ganglion_studio.core import pulse_diagnostics as pdiag
from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.dsp import FilterSettings, apply_filters, interpolate_gaps
from ganglion_studio.ui import theme

# Verdict key -> (banner colour, short headline). Colours come from the palette.
_VERDICT_STYLE = {
    pdiag.VERDICT_DIGITAL: (palette.ACCENT, "DIGITAL — pulses on packet-loss seams"),
    pdiag.VERDICT_FRAME_LOCKED: (palette.OK, "FRAME-LOCKED — coupled / board event"),
    pdiag.VERDICT_BIOLOGICAL: (palette.GOOD, "LOOKS REAL — wandering (e.g. heartbeat)"),
    pdiag.VERDICT_INCONCLUSIVE: (palette.NEUTRAL, "INCONCLUSIVE — mark an event"),
    pdiag.VERDICT_NO_PULSE: (palette.DISABLED, "NO 1 Hz PULSE in this window"),
}

# Marker code dropped by the "Mark event" button (shown as a dashed line here and
# saved to the marker channel if a recording is running).
_MARK_CODE = 9


class PulseDiagnosticsWidget(QWidget):
    # No refresh_hz cap: the scrolling trace + overlays redraw every GUI tick so
    # they're as smooth as the Time Series tab. The *expensive* analysis (filtering
    # every channel, the verdict, the per-channel readout) is slow-moving, so it is
    # throttled internally to _ANALYSIS_HZ instead.
    _ANALYSIS_HZ = 4.0

    def __init__(self, manager: BoardManager) -> None:
        super().__init__()
        self._manager = manager
        self._n = len(manager.eeg_channels)
        self._names = list(manager.channel_names)

        root = QVBoxLayout(self)
        root.addLayout(self._build_controls())

        self.verdict = QLabel("Waiting for data…")
        self.verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict.setStyleSheet(self._banner_qss(palette.DISABLED))
        root.addWidget(self.verdict)

        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet(theme.MUTED_QSS)
        root.addWidget(self.detail)

        self.metrics = QLabel("")
        self.metrics.setStyleSheet(f"color:{palette.FG}; font-family:monospace;")
        root.addWidget(self.metrics)

        self.per_channel = QLabel("")
        self.per_channel.setStyleSheet(theme.HINT_QSS)
        root.addWidget(self.per_channel)

        # --- timeline: filtered strongest channel + overlays ---
        self._timeline = pg.PlotWidget()
        self._timeline.showGrid(x=True, y=True, alpha=0.2)
        self._timeline.setLabel("bottom", "Time", units="s")
        self._timeline.setLabel("left", "Filtered", units="uV")
        self._timeline.setMouseEnabled(x=False, y=False)
        self._legend_hint = QLabel(
            "violet = detected pulse · yellow dotted = BLE packet loss · "
            "red dashed = your marker · green dashed = same channel with gaps repaired"
        )
        self._legend_hint.setStyleSheet(theme.HINT_QSS)
        root.addWidget(self._legend_hint)
        self._raw_curve = self._timeline.plot(pen=pg.mkPen(palette.ACCENT, width=1))
        self._rep_curve = self._timeline.plot(
            pen=pg.mkPen(palette.GOOD, width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
        root.addWidget(self._timeline, 2)

        # --- pulse-triggered average (the pulse's shape) ---
        self._shape = pg.PlotWidget()
        self._shape.showGrid(x=True, y=True, alpha=0.2)
        self._shape.setLabel("bottom", "Time around pulse", units="s")
        self._shape.setLabel("left", "uV")
        self._shape.setMaximumHeight(170)
        self._shape_curve = self._shape.plot(pen=pg.mkPen(palette.VIOLET, width=2))
        root.addWidget(self._shape, 1)

        # InfiniteLine overlay pools (grown/shrunk to match the event count).
        self._pulse_lines: List[pg.InfiniteLine] = []
        self._loss_lines: List[pg.InfiniteLine] = []
        self._marker_lines: List[pg.InfiniteLine] = []

        # Throttle state for the heavy analysis path.
        self._last_analysis = 0.0
        self._current_ch: Optional[int] = None  # channel the scope is showing

    # --------------------------------------------------------------- controls
    def _build_controls(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Channel"))
        self.chan_combo = QComboBox()
        self.chan_combo.addItem("Auto (strongest)", -1)
        for i in range(self._n):
            self.chan_combo.addItem(self._name(i), i)
        bar.addWidget(self.chan_combo)

        bar.addWidget(QLabel("Window"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(4.0, 30.0)
        self.window_spin.setValue(10.0)
        self.window_spin.setSuffix(" s")
        bar.addWidget(self.window_spin)
        bar.addStretch(1)

        self.mark_btn = QPushButton("Mark event ⬇")
        self.mark_btn.setToolTip(
            "Drop a marker now (e.g. 'touched laptop') to see the pulse's "
            "before/after on the timeline below.")
        self.mark_btn.clicked.connect(self._on_mark)
        bar.addWidget(self.mark_btn)
        return bar

    def _name(self, i: int) -> str:
        return self._names[i] if i < len(self._names) else f"Ch{i + 1}"

    def _on_mark(self) -> None:
        self._manager.insert_marker(_MARK_CODE)

    def set_channel_names(self, names: List[str]) -> None:
        self._names = list(names)
        for i in range(self._n):
            row = i + 1  # +1 for the "Auto" entry at index 0
            if row < self.chan_combo.count() and i < len(names):
                self.chan_combo.setItemText(row, names[i])

    # --------------------------------------------------------------- rendering
    def update_plot(self, settings: FilterSettings, active: List[bool]) -> None:
        sr = self._manager.sampling_rate
        window = self.window_spin.value()
        full = self._manager.recent(window)
        n = full.shape[1]
        if n < sr:  # need ~1 s before any verdict is meaningful
            return

        # BLE-loss flags for the same window. They come from a separate lock than
        # `full`, so the lengths can differ by a few samples mid-poll; align on the
        # most-recent end (both are newest-at-right).
        loss = self._manager.recent_loss(window)
        if loss.size != n:
            if loss.size == 0:
                loss = np.zeros(n)
            else:
                m = min(n, loss.size)
                full, loss, n = full[:, -m:], loss[-m:], m

        eeg_rows = self._manager.eeg_channels
        active_idx = [i for i in range(self._n)
                      if (active[i] if i < len(active) else True)]
        ch = self._choose_channel(active_idx, full, eeg_rows)

        # --- LIGHT path (every tick): filter ONLY the displayed channel and draw
        # the scrolling scope + overlays. This is the same per-frame cost profile
        # as one Time Series trace, so the view stays smooth. ---
        chan_f = apply_filters(full[eeg_rows[ch]].astype(float, copy=True), sr, settings)
        peaks = pdiag.detect_pulses(chan_f, sr)
        self._update_timeline(full, eeg_rows[ch], chan_f, loss, peaks, sr,
                              window, settings)
        self._update_shape(chan_f, peaks, sr)

        # --- HEAVY path (throttled): filter every channel for the per-channel
        # readout + auto-pick, and recompute the verdict/metrics. These change
        # slowly, so a few times a second is plenty. ---
        now = time.monotonic()
        if now - self._last_analysis >= 1.0 / self._ANALYSIS_HZ:
            self._last_analysis = now
            self._run_analysis(full, eeg_rows, active_idx, ch, chan_f, loss, sr,
                               settings)

    def _choose_channel(self, active_idx: List[int], full: np.ndarray,
                        eeg_rows: List[int]) -> int:
        """Channel to display: explicit combo pick, else the cached auto-pick
        (refined by the heavy path), else a cheap raw-amplitude fallback."""
        chosen = self.chan_combo.currentData()
        if chosen is not None and chosen >= 0:
            return int(chosen)
        if self._current_ch is not None and self._current_ch in active_idx:
            return self._current_ch
        if active_idx:
            ch = max(active_idx, key=lambda i: float(np.ptp(full[eeg_rows[i]])))
            self._current_ch = ch
            return ch
        return 0

    def _run_analysis(self, full, eeg_rows, active_idx, ch, chan_f, loss, sr,
                      settings) -> None:
        """Throttled: filter all active channels, refine the auto-pick, and update
        the verdict / metrics / per-channel readout."""
        filt: Dict[int, np.ndarray] = {
            i: apply_filters(full[eeg_rows[i]].astype(float, copy=True), sr, settings)
            for i in active_idx
        }
        # Refine the auto-pick (strongest filtered swing) for the next frames.
        if (self.chan_combo.currentData() or -1) < 0 and active_idx:
            self._current_ch = max(active_idx, key=lambda i: float(np.ptp(filt[i])))
        result = pdiag.diagnose(filt.get(ch, chan_f), sr, loss)
        self._update_readouts(result, active_idx, filt, sr)

    def _update_readouts(self, result, active_idx, filt, sr) -> None:
        color, headline = _VERDICT_STYLE.get(
            result.verdict, (palette.NEUTRAL, result.verdict))
        self.verdict.setText(headline)
        self.verdict.setStyleSheet(self._banner_qss(color))
        self.detail.setText(result.message)
        self.metrics.setText(
            f"rate {result.rate_hz:5.2f} Hz   jitter {result.jitter_ms:5.1f} ms   "
            f"phase-lock r {result.phase_r:4.2f}   loss-coincidence "
            f"{result.loss_fraction * 100:3.0f}% (chance {result.loss_chance * 100:.0f}%)   "
            f"BLE loss {self._manager.loss_rate(10.0):.1f}/s "
            f"({self._manager.dropped_packets()} dropped)"
        )
        parts = []
        for i in active_idx:
            peaks = pdiag.detect_pulses(filt[i], sr)
            if peaks.size:
                amp = float(np.median(np.abs(filt[i][peaks])))
                parts.append(f"{self._name(i)}: {peaks.size}× ~{amp:.0f}µV")
            else:
                parts.append(f"{self._name(i)}: —")
        self.per_channel.setText("Per channel:   " + "    ".join(parts))

    def _update_timeline(self, full, row, chan_f, loss, peaks, sr, window,
                         settings) -> None:
        n = chan_f.size
        t = (np.arange(n) - (n - 1)) / sr  # newest sample at t=0, past is negative

        self._raw_curve.setData(t, chan_f)
        # Show the same channel with packet-loss seams interpolated away. If the
        # pulses vanish here, they WERE the held-flat seam (the green trace is the
        # "fixed" view); if they remain, the pulse is not packet loss.
        if loss.any():
            mask = loss != 0
            mask[1:] = mask[1:] | (loss[:-1] != 0)  # the drop's 2nd held sample too
            repaired = interpolate_gaps(full[row].astype(float, copy=True), mask)
            self._rep_curve.setData(t, apply_filters(repaired, sr, settings))
            self._rep_curve.setVisible(True)
        else:
            self._rep_curve.setVisible(False)

        pulse_x = (peaks - (n - 1)) / sr if peaks.size else np.empty(0)
        loss_x = (np.flatnonzero(loss != 0) - (n - 1)) / sr if loss.any() else np.empty(0)
        mch = self._manager.marker_channel
        mrow = full[mch] if mch < full.shape[0] else None
        marker_x = ((np.flatnonzero(mrow != 0) - (n - 1)) / sr
                    if mrow is not None else np.empty(0))

        self._sync_lines(self._pulse_lines, pulse_x, palette.VIOLET,
                         pg.QtCore.Qt.PenStyle.SolidLine)
        self._sync_lines(self._loss_lines, loss_x, palette.OK,
                         pg.QtCore.Qt.PenStyle.DotLine)
        self._sync_lines(self._marker_lines, marker_x, palette.MARKER,
                         pg.QtCore.Qt.PenStyle.DashLine)
        self._timeline.setXRange(-window, 0, padding=0)

    def _update_shape(self, chan_f, peaks, sr) -> None:
        t, avg = pdiag.pulse_triggered_average(chan_f, peaks, sr)
        if t.size:
            self._shape_curve.setData(t, avg)
        else:
            self._shape_curve.clear()

    # ----------------------------------------------------------- overlay pool
    def _sync_lines(self, pool: List[pg.InfiniteLine], xs: np.ndarray,
                    color: str, style) -> None:
        """Grow/shrink an InfiniteLine pool to match ``xs`` and position them."""
        while len(pool) < len(xs):
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen(color, width=1, style=style))
            self._timeline.addItem(line)
            pool.append(line)
        while len(pool) > len(xs):
            self._timeline.removeItem(pool.pop())
        for line, x in zip(pool, xs):
            line.setValue(float(x))

    @staticmethod
    def _banner_qss(bg: str) -> str:
        return (f"background:{bg}; color:{palette.WHITE}; font-size:18px; "
                "font-weight:700; border-radius:6px; padding:8px;")
