"""Recording Editor: reopen a saved .fif to crop it, curate markers and edit
per-channel info (placement / type / electrode), then Save As a new .fif.

This borrows the pyqtgraph browsing + marker editing from ``review_window.py``
and the per-channel Placement/Type/Electrode grid from ``channel_setup_dialog.py``.
Unlike the live Review window it starts empty and loads a file from disk via
``recording_loader.load_recording``.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio import palette
from ganglion_studio.core import board_config as cfg
from ganglion_studio.core import exporter, recording_loader
from ganglion_studio.core.dsp import FilterSettings, apply_filters
from ganglion_studio.core.exporter import ReviewMarker
from ganglion_studio.ui import theme


class RecordingEditorWindow(QMainWindow):
    """Open a saved recording, crop / edit markers / edit channels, Save As."""

    def __init__(self, initial_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Recording Editor")
        self.resize(1300, 820)

        # --- state (populated on load) ------------------------------------
        self._rec: Optional[recording_loader.LoadedRecording] = None
        self._sr = 200
        self._names: List[str] = []
        self._n = 0
        self._total = 0
        self._duration = 0.0
        self._markers: List[ReviewMarker] = []
        self._filter = FilterSettings()
        self._click_time: Optional[float] = None
        self._curves: List[pg.PlotDataItem] = []
        self._marker_lines: List[pg.InfiniteLine] = []
        self._place_combos: List[QComboBox] = []
        self._type_combos: List[QComboBox] = []
        self._elec_combos: List[QComboBox] = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.addLayout(self._build_toolbar())

        body = QHBoxLayout()
        root.addLayout(body, 1)
        body.addLayout(self._build_plot(), 3)
        body.addWidget(self._build_side_panel(), 1)

        # Prompt for a file straight away (or load the one we were given).
        self._open_file(initial_path)

    # ---------------------------------------------------------------- build
    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()

        open_btn = QPushButton("Open…")
        open_btn.setToolTip("Open a saved recording (.fif and other MNE formats)")
        open_btn.clicked.connect(lambda: self._open_file(None))
        bar.addWidget(open_btn)

        self._info_label = QLabel("No recording loaded")
        self._info_label.setStyleSheet(theme.MUTED_QSS)
        bar.addWidget(self._info_label)
        bar.addStretch(1)

        bar.addWidget(QLabel("Window"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(1.0, 10.0)
        self.window_spin.setValue(10.0)
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
        self.markers_chk.setToolTip("Show the event markers (dashed lines)")
        self.markers_chk.toggled.connect(self._draw_marker_lines)
        bar.addWidget(self.markers_chk)

        self.save_btn = QPushButton("Save As .fif…")
        self.save_btn.setStyleSheet("QPushButton { background:#2f7d4f; font-weight:600; }")
        self.save_btn.clicked.connect(self._on_save)
        bar.addWidget(self.save_btn)
        return bar

    def _build_plot(self) -> QVBoxLayout:
        col = QVBoxLayout()
        self._plot_widget = pg.PlotWidget()
        self._plot = self._plot_widget.getPlotItem()
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setMouseEnabled(x=True, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        self._cursor = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(palette.WHITE, width=1, style=Qt.PenStyle.DotLine))
        self._plot.addItem(self._cursor)

        # Green dashed lines marking the kept range; everything outside is cut.
        keep_pen = pg.mkPen(palette.GOOD, width=2, style=Qt.PenStyle.DashLine)
        self._keep_lo = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=keep_pen)
        self._keep_hi = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=keep_pen)
        self._plot.addItem(self._keep_lo, ignoreBounds=True)
        self._plot.addItem(self._keep_hi, ignoreBounds=True)

        col.addWidget(self._plot_widget, 1)

        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar.valueChanged.connect(self._on_view_changed)
        col.addWidget(self.scrollbar)
        self._plot.getViewBox().sigRangeChangedManually.connect(self._on_plot_xrange)
        return col

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # --- Keep range -----------------------------------------------------
        keep_box = QGroupBox("Keep range (crop)")
        keep_layout = QVBoxLayout(keep_box)
        from_row = QHBoxLayout()
        from_row.addWidget(QLabel("Keep from"))
        self.keep_from = QDoubleSpinBox()
        self.keep_from.setRange(0.0, 0.0)
        self.keep_from.setDecimals(3)
        self.keep_from.setSuffix(" s")
        self.keep_from.valueChanged.connect(self._update_keep_lines)
        from_row.addWidget(self.keep_from)
        keep_layout.addLayout(from_row)

        to_row = QHBoxLayout()
        to_row.addWidget(QLabel("Keep to"))
        self.keep_to = QDoubleSpinBox()
        self.keep_to.setRange(0.0, 0.0)
        self.keep_to.setDecimals(3)
        self.keep_to.setSuffix(" s")
        self.keep_to.valueChanged.connect(self._update_keep_lines)
        to_row.addWidget(self.keep_to)
        keep_layout.addLayout(to_row)

        use_view_btn = QPushButton("Use current view")
        use_view_btn.setToolTip("Set the keep range to the time span currently visible")
        use_view_btn.clicked.connect(self._on_use_current_view)
        keep_layout.addWidget(use_view_btn)

        keep_hint = QLabel("Everything outside this range is discarded on save.")
        keep_hint.setStyleSheet(theme.HINT_QSS)
        keep_hint.setWordWrap(True)
        keep_layout.addWidget(keep_hint)
        layout.addWidget(keep_box)

        # --- Markers --------------------------------------------------------
        marker_box = QGroupBox("Markers")
        marker_layout = QVBoxLayout(marker_box)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Time", "Sample", "Label"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        marker_layout.addWidget(self.table, 1)

        add_row = QHBoxLayout()
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Marker label")
        add_row.addWidget(self.label_edit, 1)
        add_btn = QPushButton("Add at cursor")
        add_btn.setToolTip("Add a marker at the clicked position (or view center)")
        add_btn.clicked.connect(self._on_add_marker)
        add_row.addWidget(add_btn)
        marker_layout.addLayout(add_row)

        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._on_remove_marker)
        marker_layout.addWidget(remove_btn)

        marker_hint = QLabel("Click the plot to place the cursor, then Add.")
        marker_hint.setStyleSheet(theme.HINT_QSS)
        marker_hint.setWordWrap(True)
        marker_layout.addWidget(marker_hint)
        layout.addWidget(marker_box, 1)

        # --- Channels -------------------------------------------------------
        chan_box = QGroupBox("Channels")
        chan_outer = QVBoxLayout(chan_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._chan_container = QWidget()
        self._chan_grid = QGridLayout(self._chan_container)
        scroll.setWidget(self._chan_container)
        chan_outer.addWidget(scroll)
        layout.addWidget(chan_box, 1)

        return panel

    # ---------------------------------------------------------------- load
    def _open_file(self, path: Optional[str]) -> None:
        if not path:
            exts = " ".join(f"*{e}" for e in sorted(recording_loader.MNE_EXTENSIONS))
            start_dir = self._default_dir()
            path, _ = QFileDialog.getOpenFileName(
                self, "Open recording", start_dir,
                f"MNE recordings ({exts});;FIF (*.fif);;All files (*)",
            )
        if not path:
            return
        try:
            rec = recording_loader.load_recording(path)
        except recording_loader.LoadError as exc:
            QMessageBox.warning(self, "Could not open", str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.critical(self, "Could not open", f"Unexpected error: {exc}")
            return

        self._rec = rec
        self._sr = int(rec.sampling_rate) or 200
        self._names = list(rec.channel_names)
        self._n = rec.n_channels
        self._total = rec.n_samples
        self._duration = self._total / self._sr if self._sr else 0.0
        self._markers = list(rec.markers)
        self._filter = FilterSettings()
        self._click_time = None
        self.setWindowTitle(f"Recording Editor - {os.path.basename(path)}")

        self._rebuild_curves()
        self._rebuild_channels()
        self._configure_ranges()
        self._redraw()
        self._update_marker_table()
        self._update_info()

    def _default_dir(self) -> str:
        """Prefer the repo's mne_data_raw folder, else the current directory."""
        if self._rec and self._rec.source_path:
            return os.path.dirname(self._rec.source_path)
        cand = os.path.join(os.getcwd(), "mne_data_raw")
        return cand if os.path.isdir(cand) else os.getcwd()

    def _configure_ranges(self) -> None:
        dur = self._duration or 0.0
        self.window_spin.blockSignals(True)
        self.window_spin.setRange(1.0, max(2.0, dur))
        self.window_spin.setValue(min(10.0, dur or 10.0))
        self.window_spin.blockSignals(False)
        for spin, default in ((self.keep_from, 0.0), (self.keep_to, dur)):
            spin.blockSignals(True)
            spin.setRange(0.0, dur)
            spin.setValue(default)
            spin.blockSignals(False)
        if self._duration > 0:
            self._plot.setLimits(xMin=0.0, xMax=self._duration)
        self._update_keep_lines()

    def _rebuild_curves(self) -> None:
        for c in self._curves:
            self._plot.removeItem(c)
        self._curves = []
        for i in range(self._n):
            color = cfg.CHANNEL_COLORS[i % len(cfg.CHANNEL_COLORS)]
            self._curves.append(self._plot.plot(pen=pg.mkPen(color, width=1)))

    def _rebuild_channels(self) -> None:
        # Clear the existing grid widgets.
        while self._chan_grid.count():
            item = self._chan_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._place_combos = []
        self._type_combos = []
        self._elec_combos = []

        for c, header in enumerate(["Channel", "Placement", "Type", "Electrode"]):
            lab = QLabel(header)
            lab.setStyleSheet("font-weight:600;")
            self._chan_grid.addWidget(lab, 0, c)

        rec = self._rec
        for i in range(self._n):
            self._chan_grid.addWidget(QLabel(self._names[i]), i + 1, 0)

            place = QComboBox()
            place.addItems(cfg.TEN_TWENTY)
            place.setCurrentText(self._initial_placement(i))
            self._chan_grid.addWidget(place, i + 1, 1)
            self._place_combos.append(place)

            tcombo = QComboBox()
            tcombo.addItems(cfg.CHANNEL_TYPES)
            saved_t = rec.channel_types[i] if rec and i < len(rec.channel_types) else "EEG"
            tcombo.setCurrentText(saved_t or "EEG")
            self._chan_grid.addWidget(tcombo, i + 1, 2)
            self._type_combos.append(tcombo)

            ecombo = QComboBox()
            ecombo.addItems(cfg.ELECTRODES)
            saved_e = rec.electrodes[i] if rec and i < len(rec.electrodes) else cfg.ELECTRODES[0]
            ecombo.setCurrentText(saved_e or cfg.ELECTRODES[0])
            self._chan_grid.addWidget(ecombo, i + 1, 3)
            self._elec_combos.append(ecombo)

    def _initial_placement(self, i: int) -> str:
        rec = self._rec
        if rec and i < len(rec.placements) and rec.placements[i]:
            return rec.placements[i]
        # Fall back to the channel name if it is a known 10-20 label.
        name = self._names[i] if i < len(self._names) else ""
        return name if name in cfg.TEN_TWENTY else "None"

    # ---------------------------------------------------------------- view
    def _configure_scrollbar(self) -> None:
        window = self.window_spin.value()
        max_start = max(0.0, self._duration - window)
        self.scrollbar.blockSignals(True)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(int(max_start * 1000))
        self.scrollbar.setPageStep(int(window * 1000))
        self.scrollbar.setSingleStep(int(window * 100))
        self.scrollbar.blockSignals(False)

    def _on_view_changed(self, *_args) -> None:
        if self._total == 0:
            return
        self._configure_scrollbar()
        start = self.scrollbar.value() / 1000.0
        window = self.window_spin.value()
        self._plot.setXRange(start, start + window, padding=0)

    def _on_plot_xrange(self, *_args) -> None:
        if self._total == 0:
            return
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
        if self._rec is None or self._total == 0:
            return
        t = np.arange(self._total) / self._sr
        amp = self.amp_spin.value()
        spacing = amp * 2.2
        ticks = []
        for i in range(self._n):
            y = np.ascontiguousarray(self._rec.eeg[i, :], dtype=np.float64)
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
        self._on_view_changed()

    def _draw_marker_lines(self, *_args) -> None:
        for line in self._marker_lines:
            self._plot.removeItem(line)
        self._marker_lines.clear()
        if not self.markers_chk.isChecked():
            return
        for m in self._markers:
            x = m.sample / self._sr
            line = pg.InfiniteLine(
                pos=x, angle=90, movable=False,
                pen=pg.mkPen(palette.MARKER, width=1, style=Qt.PenStyle.DashLine),
                label=m.label,
                labelOpts={"position": 0.95, "color": palette.MARKER, "fill": "#00000080"},
            )
            self._plot.addItem(line)
            self._marker_lines.append(line)

    def _update_keep_lines(self, *_args) -> None:
        self._keep_lo.setValue(self.keep_from.value())
        self._keep_hi.setValue(self.keep_to.value())

    def _on_use_current_view(self) -> None:
        if self._total == 0:
            return
        x0, x1 = self._plot.getViewBox().viewRange()[0]
        self.keep_from.setValue(max(0.0, x0))
        self.keep_to.setValue(min(self._duration, x1))

    # --------------------------------------------------------------- markers
    def _on_plot_clicked(self, event) -> None:
        if self._total == 0:
            return
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
        label = self.label_edit.text().strip() or "Marker"
        self._markers.append(ReviewMarker(sample, 0, label))
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
        self._update_info()

    def _update_marker_table(self) -> None:
        self.table.setRowCount(len(self._markers))
        for r, m in enumerate(self._markers):
            tsec = m.sample / self._sr
            self.table.setItem(r, 0, QTableWidgetItem(f"{tsec:0.3f}"))
            self.table.setItem(r, 1, QTableWidgetItem(str(m.sample)))
            self.table.setItem(r, 2, QTableWidgetItem(m.label))

    def _update_info(self) -> None:
        if self._rec is None:
            self._info_label.setText("No recording loaded")
            return
        self._info_label.setText(
            f"{self._n} ch | {self._sr} Hz | {self._duration:0.1f} s | "
            f"{len(self._markers)} markers"
        )

    # ---------------------------------------------------------------- save
    def _edited_names(self) -> List[str]:
        """Display names: placement when set, else the loaded channel name."""
        out = []
        for i in range(self._n):
            place = self._place_combos[i].currentText()
            if place in ("None", "Custom"):
                out.append(self._names[i] if i < len(self._names) else f"Ch{i+1}")
            else:
                out.append(place)
        return out

    def _on_save(self) -> None:
        if self._rec is None or self._total == 0:
            QMessageBox.information(self, "Nothing to save", "Open a recording first.")
            return
        if not exporter.mne_available():
            QMessageBox.warning(
                self, "Export unavailable",
                "MNE is not installed.\nRun: pip install mne",
            )
            return

        # Resolve the keep range to samples and crop.
        s0 = int(round(self.keep_from.value() * self._sr))
        s1 = int(round(self.keep_to.value() * self._sr))
        s0 = max(0, min(self._total, s0))
        s1 = max(0, min(self._total, s1))
        if s1 - s0 < 1:
            QMessageBox.warning(self, "Empty range",
                                "Keep-to must be after Keep-from.")
            return
        cropped = np.ascontiguousarray(self._rec.eeg[:, s0:s1], dtype=np.float64)

        # Shift markers into the cropped window; drop those outside it.
        remapped = [ReviewMarker(m.sample - s0, m.code, m.label)
                    for m in self._markers if s0 <= m.sample < s1]

        names = self._edited_names()
        placements = [c.currentText() for c in self._place_combos]
        types = [c.currentText() for c in self._type_combos]
        electrodes = [c.currentText() for c in self._elec_combos]

        default_name = os.path.basename(self._rec.source_path) or "recording.fif"
        start_dir = os.path.join(self._default_dir(), default_name)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save As .fif", start_dir, "FIF (*.fif)")
        if not path:
            return

        meta = {
            "sampling_rate": self._sr,
            "eeg_channels": list(range(self._n)),
            "channel_names": names,
        }
        try:
            written = exporter.export(path, "fif", cropped, meta, remapped)
        except exporter.ExportError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.critical(self, "Save failed", f"Unexpected error: {exc}")
            return

        self._write_sidecar(written, names, types, electrodes, placements)
        QMessageBox.information(self, "Saved", f"Saved:\n{written}")

    def _write_sidecar(self, fif_path: str, names: List[str], types: List[str],
                       electrodes: List[str], placements: List[str]) -> None:
        """Write the channel edits .fif cannot store to a sibling _meta.json.

        Uses the same base-name convention the loader looks for, so re-opening
        the saved file restores these edits.
        """
        active = list(self._rec.channel_active) if self._rec else []
        if len(active) != self._n:
            active = [True] * self._n
        meta = {
            "sampling_rate": self._sr,
            "eeg_channels": list(range(self._n)),
            "channel_names": names,
            "channel_types": types,
            "electrodes": electrodes,
            "placements": placements,
            "channel_active": active,
        }
        sidecar = os.path.splitext(fif_path)[0] + "_meta.json"
        try:
            with open(sidecar, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
        except Exception:  # pragma: no cover - non-fatal
            pass
