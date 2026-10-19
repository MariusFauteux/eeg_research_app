"""Post-recording review: browse the recording, edit markers, export."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio import palette
from ganglion_studio.core import board_config as cfg
from ganglion_studio.core import exporter
from ganglion_studio.core.dsp import FilterSettings, apply_filters
from ganglion_studio.core.exporter import ReviewMarker
from ganglion_studio.ui import theme

# Packet-drop overlay is drawn as one curve item whose vertical ticks span this
# fixed (huge) y-extent, so each tick covers the full plot height no matter the
# amplitude/zoom. Added with ignoreBounds=True so it never drags auto-range.
_OVERLAY_Y = 1e9


class ReviewWindow(QMainWindow):
    """Browse a finished recording and curate markers before exporting."""

    def __init__(self, raw_data: np.ndarray, meta: dict,
                 code_labels: Optional[Dict[int, str]] = None,
                 marker_types: Optional[list] = None,
                 title: str = "Recording",
                 loss_samples: Optional[List[int]] = None) -> None:
        super().__init__()
        self.setWindowTitle(f"Review - {title}")
        self.resize(1200, 800)

        self._raw = raw_data
        self._meta = dict(meta)
        self._sr = float(meta.get("sampling_rate", 200))
        self._eeg_rows = list(meta.get("eeg_channels", []))
        self._names = list(meta.get("channel_names",
                                    [f"Ch{i+1}" for i in range(len(self._eeg_rows))]))
        self._code_labels = code_labels or {}
        self._marker_types = marker_types or []
        self._n = len(self._eeg_rows)
        self._total = raw_data.shape[1] if raw_data.ndim == 2 else 0
        self._duration = self._total / self._sr if self._sr else 0.0
        self._click_time: Optional[float] = None
        self._filter = FilterSettings(notch_freq=int(meta.get("notch_freq", 50)))
        self._marker_lines: List[pg.InfiniteLine] = []
        self._impedance_windows = []
        # Sample indices where native BLE dropped a packet (from _packet_loss.csv).
        self._loss_samples: List[int] = list(loss_samples if loss_samples is not None
                                            else meta.get("loss_samples", []))
        self._meta["loss_samples"] = self._loss_samples

        self._markers: List[ReviewMarker] = self._extract_markers()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.addLayout(self._build_toolbar())

        body = QHBoxLayout()
        root.addLayout(body, 1)
        body.addLayout(self._build_plot(), 3)
        body.addWidget(self._build_marker_panel(), 1)

        self._redraw()
        self._update_marker_table()

    # ------------------------------------------------------------- markers
    def _extract_markers(self) -> List[ReviewMarker]:
        if "marker_events" in self._meta:
            return [ReviewMarker(m["sample"], m["code"], m["label"])
                    for m in self._meta["marker_events"]
                    if 0 <= m["sample"] < self._total]
        mch = int(self._meta.get("marker_channel", 0))
        markers: List[ReviewMarker] = []
        if self._total and 0 <= mch < self._raw.shape[0]:
            row = self._raw[mch, :]
            for idx in np.flatnonzero(row != 0):
                code = int(row[idx])
                label = self._code_labels.get(code, f"code {code}")
                markers.append(ReviewMarker(int(idx), code, label))
        return markers

    # ------------------------------------------------------------- build UI
    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        info = QLabel(f"{self._n} ch | {self._sr} Hz | {self._duration:0.1f} s "
                      f"| {len(self._markers)} markers")
        info.setStyleSheet(theme.MUTED_QSS)
        self._info_label = info
        bar.addWidget(info)
        bar.addStretch(1)

        if self._meta.get("impedance_file"):
            impedance_btn = QPushButton("View impedance")
            impedance_btn.clicked.connect(self._open_impedance)
            bar.addWidget(impedance_btn)

        bar.addWidget(QLabel("Window"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(1.0, max(2.0, self._duration))
        self.window_spin.setValue(min(10.0, self._duration or 10.0))
        self.window_spin.setSuffix(" s")
        self.window_spin.valueChanged.connect(self._on_view_changed)
        bar.addWidget(self.window_spin)

        bar.addWidget(QLabel("Amplitude"))
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(1.0, 100000.0)
        self.amp_spin.setValue(200.0)
        self.amp_spin.setSuffix(" uV")
        self.amp_spin.valueChanged.connect(self._redraw)
        bar.addWidget(self.amp_spin)

        self.filter_chk = QCheckBox("Display filter")
        self.filter_chk.setToolTip("Band-pass + notch for viewing only (export stays raw)")
        self.filter_chk.toggled.connect(self._redraw)
        bar.addWidget(self.filter_chk)

        self.markers_chk = QCheckBox("Markers")
        self.markers_chk.setChecked(True)
        self.markers_chk.setToolTip("Show the manual event markers (dashed lines)")
        self.markers_chk.toggled.connect(self._draw_marker_lines)
        bar.addWidget(self.markers_chk)

        self.drops_chk = QCheckBox("Packet drops")
        has_loss = bool(self._loss_samples)
        self.drops_chk.setChecked(has_loss)
        self.drops_chk.setEnabled(has_loss)
        self.drops_chk.setToolTip(
            "Show where the native-Bluetooth link dropped a packet (dotted lines)"
            if has_loss else "No packet drops were recorded"
        )
        self.drops_chk.toggled.connect(self._draw_loss_marks)
        bar.addWidget(self.drops_chk)

        save_btn = QPushButton("Save / Export...")
        save_btn.setStyleSheet("QPushButton { background:#2f7d4f; font-weight:600; }")
        save_btn.clicked.connect(self._on_save)
        bar.addWidget(save_btn)
        return bar

    def _open_impedance(self) -> None:
        from ganglion_studio.ui.impedance_session_view import ImpedanceRecordingWindow

        try:
            window = ImpedanceRecordingWindow(self._meta["impedance_file"])
        except Exception as exc:
            QMessageBox.warning(self, "Could not open impedance recording", str(exc))
            return
        self._impedance_windows.append(window)
        window.show()

    def _build_plot(self) -> QVBoxLayout:
        col = QVBoxLayout()
        self._plot_widget = pg.PlotWidget()
        self._plot = self._plot_widget.getPlotItem()
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setMouseEnabled(x=True, y=False)  # touchpad: scroll/pinch = zoom, drag = pan
        self._plot.setMenuEnabled(False)
        if self._duration > 0:
            self._plot.setLimits(xMin=0.0, xMax=self._duration)
        self._curves: List[pg.PlotDataItem] = []
        for i in range(self._n):
            color = cfg.CHANNEL_COLORS[i % len(cfg.CHANNEL_COLORS)]
            self._curves.append(self._plot.plot(pen=pg.mkPen(color, width=1)))
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)
        self._cursor = pg.InfiniteLine(angle=90, movable=False,
                                       pen=pg.mkPen(palette.WHITE, width=1, style=Qt.PenStyle.DotLine))
        self._plot.addItem(self._cursor)
        # Packet-drop overlay: a single curve drawing one vertical tick per drop.
        self._loss_curve = pg.PlotCurveItem(
            pen=pg.mkPen(palette.OK, width=1, style=Qt.PenStyle.DotLine))
        self._plot.addItem(self._loss_curve, ignoreBounds=True)
        col.addWidget(self._plot_widget, 1)

        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar.valueChanged.connect(self._on_view_changed)
        col.addWidget(self.scrollbar)
        self._configure_scrollbar()
        # Only manual (mouse/trackpad) zoom/pan updates the scrollbar, so our own
        # programmatic setXRange and pyqtgraph autorange don't fight it.
        self._plot.getViewBox().sigRangeChangedManually.connect(self._on_plot_xrange)
        return col

    def _build_marker_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Markers"))

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Time", "Sample", "Code", "Label"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        layout.addWidget(self.table, 1)

        add_row = QHBoxLayout()
        self.type_combo = QComboBox()
        if self._marker_types:
            for mt in self._marker_types:
                self.type_combo.addItem(f"{mt.label} ({mt.code})", mt)
        else:
            self.type_combo.addItem("Marker (1)", None)
        add_row.addWidget(self.type_combo, 1)
        add_btn = QPushButton("Add")
        add_btn.setToolTip("Add a marker at the clicked position (or view center)")
        add_btn.clicked.connect(self._on_add_marker)
        add_row.addWidget(add_btn)
        layout.addLayout(add_row)

        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._on_remove_marker)
        layout.addWidget(remove_btn)

        hint = QLabel("Click the plot to place the cursor, then Add.")
        hint.setStyleSheet(theme.HINT_QSS)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return panel

    # --------------------------------------------------------------- view
    def _configure_scrollbar(self) -> None:
        window = self.window_spin.value()
        max_start = max(0.0, self._duration - window)
        # scrollbar works in integer milliseconds for smooth stepping
        self.scrollbar.blockSignals(True)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(int(max_start * 1000))
        self.scrollbar.setPageStep(int(window * 1000))
        self.scrollbar.setSingleStep(int(window * 100))
        self.scrollbar.blockSignals(False)

    def _on_view_changed(self, *_args) -> None:
        self._configure_scrollbar()
        start = self.scrollbar.value() / 1000.0
        window = self.window_spin.value()
        self._plot.setXRange(start, start + window, padding=0)

    def _on_plot_xrange(self, *_args) -> None:
        """A manual zoom/pan changed the view -> reflect it in the controls."""
        x0, x1 = self._plot.getViewBox().viewRange()[0]
        window = max(0.05, x1 - x0)
        self.window_spin.blockSignals(True)
        self.window_spin.setValue(min(window, self.window_spin.maximum()))
        self.window_spin.blockSignals(False)
        self.scrollbar.blockSignals(True)
        self._configure_scrollbar()
        self.scrollbar.setValue(int(max(0.0, x0) * 1000))
        self.scrollbar.blockSignals(False)

    def _redraw(self, *_args) -> None:
        if self._total == 0:
            return
        t = np.arange(self._total) / self._sr
        amp = self.amp_spin.value()
        spacing = amp * 2.2
        ticks = []
        for i in range(self._n):
            y = np.ascontiguousarray(self._raw[self._eeg_rows[i], :], dtype=np.float64)
            if self.filter_chk.isChecked():
                y = apply_filters(y, self._sr, self._filter)
            offset = (self._n - 1 - i) * spacing
            self._curves[i].setData(t, y + offset)
            self._curves[i].setDownsampling(auto=True, method="peak")
            self._curves[i].setClipToView(True)
            ticks.append((offset, self._names[i] if i < len(self._names) else f"Ch{i+1}"))
        self._plot.getAxis("left").setTicks([ticks])
        self._plot.setYRange(-spacing, self._n * spacing, padding=0)
        self._draw_marker_lines()
        self._draw_loss_marks()
        self._on_view_changed()

    def _draw_loss_marks(self, *_args) -> None:
        """Dotted vertical ticks where native BLE dropped a packet."""
        if not self._loss_samples or not self.drops_chk.isChecked():
            self._loss_curve.clear()
            return
        xs = np.array([s / self._sr for s in self._loss_samples
                       if 0 <= s < self._total], dtype=float)
        if xs.size == 0:
            self._loss_curve.clear()
            return
        # One curve, one vertical tick per drop: repeat each x and pair it with a
        # full-height y-span; connect="pairs" joins (x,-Y)-(x,+Y) for each drop.
        x = np.repeat(xs, 2)
        y = np.tile([-_OVERLAY_Y, _OVERLAY_Y], xs.size)
        self._loss_curve.setData(x, y, connect="pairs")

    def _draw_marker_lines(self, *_args) -> None:
        for line in self._marker_lines:
            self._plot.removeItem(line)
        self._marker_lines.clear()
        if not self.markers_chk.isChecked():
            return
        for m in self._markers:
            x = m.sample / self._sr
            color = self._color_for_code(m.code)
            line = pg.InfiniteLine(
                pos=x, angle=90, movable=False,
                pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                label=m.label, labelOpts={"position": 0.95, "color": color, "fill": "#00000080"},
            )
            self._plot.addItem(line)
            self._marker_lines.append(line)

    def _color_for_code(self, code: int) -> str:
        for mt in self._marker_types:
            if getattr(mt, "code", None) == code:
                return getattr(mt, "color", palette.MARKER)
        return palette.MARKER

    # ------------------------------------------------------------- markers
    def _on_plot_clicked(self, event) -> None:
        pos = event.scenePos()
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        x = self._plot.vb.mapSceneToView(pos).x()
        self._click_time = max(0.0, min(self._duration, float(x)))
        self._cursor.setValue(self._click_time)

    def _on_add_marker(self) -> None:
        if self._total == 0:
            return
        if self._click_time is None:
            start = self.scrollbar.value() / 1000.0
            self._click_time = start + self.window_spin.value() / 2.0
        sample = int(round(self._click_time * self._sr))
        sample = max(0, min(self._total - 1, sample))
        mt = self.type_combo.currentData()
        code = getattr(mt, "code", 1) if mt is not None else 1
        label = getattr(mt, "label", "Marker") if mt is not None else "Marker"
        self._markers.append(ReviewMarker(sample, int(code), label))
        self._markers.sort(key=lambda m: m.sample)
        self._refresh_after_edit()

    def _on_remove_marker(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self._markers):
                del self._markers[r]
        self._refresh_after_edit()

    def _on_table_selection(self) -> None:
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        if len(rows) == 1:
            r = next(iter(rows))
            if 0 <= r < len(self._markers):
                self._cursor.setValue(self._markers[r].sample / self._sr)

    def _refresh_after_edit(self) -> None:
        self._draw_marker_lines()
        self._update_marker_table()
        self._info_label.setText(
            f"{self._n} ch | {self._sr} Hz | {self._duration:0.1f} s | {len(self._markers)} markers"
        )

    def _update_marker_table(self) -> None:
        self.table.setRowCount(len(self._markers))
        for r, m in enumerate(self._markers):
            tsec = m.sample / self._sr
            self.table.setItem(r, 0, QTableWidgetItem(f"{tsec:0.3f}"))
            self.table.setItem(r, 1, QTableWidgetItem(str(m.sample)))
            self.table.setItem(r, 2, QTableWidgetItem(str(m.code)))
            self.table.setItem(r, 3, QTableWidgetItem(m.label))

    # --------------------------------------------------------------- save
    def _on_save(self) -> None:
        avail = exporter.available_formats()
        if not any(avail.values()):
            QMessageBox.warning(
                self, "Export unavailable",
                "No export backend found.\nRun: pip install mne eeglabio edfio",
            )
            return

        filters = []
        for key, (desc, ext) in exporter.FORMATS.items():
            tag = "" if avail.get(key) else "  [unavailable]"
            filters.append(f"{desc}{tag} (*{ext})")
        filter_str = ";;".join(filters)

        path, selected = QFileDialog.getSaveFileName(
            self, "Export recording", self.windowTitle().replace("Review - ", ""), filter_str
        )
        if not path:
            return
        fmt = self._fmt_from_filter(selected, path)
        try:
            written = exporter.export(path, fmt, self._raw, self._meta, self._markers)
        except exporter.ExportError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.critical(self, "Export failed", f"Unexpected error: {exc}")
            return
        QMessageBox.information(self, "Exported", f"Saved:\n{written}")

    @staticmethod
    def _fmt_from_filter(selected: str, path: str) -> str:
        for key, (_desc, ext) in exporter.FORMATS.items():
            if path.lower().endswith(ext) or (selected and f"*{ext}" in selected):
                return key
        return "fif"
