"""Processing Lab: load a recording and compare original vs processed signal."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio.core import board_config as cfg
from ganglion_studio.core import processing as proc
from ganglion_studio.core.dsp import FILTER_TYPES, compute_psd
from ganglion_studio.core.recording_loader import LoadError, LoadedRecording, load_recording


class ProcessingWorker(QThread):
    done = pyqtSignal(object, list)
    failed = pyqtSignal(str)

    def __init__(self, eeg, sr, names, config) -> None:
        super().__init__()
        self._eeg = eeg
        self._sr = sr
        self._names = names
        self._config = config

    def run(self) -> None:
        try:
            out, messages = proc.apply_pipeline(self._eeg, self._sr, self._names, self._config)
            self.done.emit(out, messages)
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(str(exc))


class ProcessingWindow(QWidget):
    """Standalone window: configuration on the left, original/processed on the right."""

    def __init__(self, initial_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Processing Lab")
        self.resize(1300, 860)

        self._rec: Optional[LoadedRecording] = None
        self._processed: Optional[np.ndarray] = None
        self._worker: Optional[ProcessingWorker] = None
        self._orig_curves: List[pg.PlotDataItem] = []
        self._proc_curves: List[pg.PlotDataItem] = []

        self._avail = proc.available_methods()

        root = QHBoxLayout(self)
        root.addWidget(self._build_config_panel())
        root.addWidget(self._build_views(), 1)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._recompute)

        if initial_path:
            self._load(initial_path)

    # ----------------------------------------------------------- config UI
    def _build_config_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        file_box = QGroupBox("Recording")
        fb = QVBoxLayout(file_box)
        self.open_btn = QPushButton("Open file...")
        self.open_btn.clicked.connect(self._on_open)
        fb.addWidget(self.open_btn)
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color:#9aa0aa; font-size:11px;")
        fb.addWidget(self.file_label)
        layout.addWidget(file_box)

        # Re-reference + detrend
        pre_box = QGroupBox("Pre-processing")
        pf = QFormLayout(pre_box)
        self.reref_chk = QCheckBox("Common average (CAR)")
        self.reref_chk.toggled.connect(self._schedule)
        pf.addRow(self.reref_chk)
        self.detrend_combo = QComboBox()
        self.detrend_combo.addItems(["none", "constant", "linear"])
        self.detrend_combo.currentTextChanged.connect(self._schedule)
        pf.addRow("Detrend", self.detrend_combo)
        layout.addWidget(pre_box)

        layout.addWidget(self._build_filter_box())
        layout.addWidget(self._build_wavelet_box())
        layout.addWidget(self._build_asr_box())
        layout.addWidget(self._build_aas_box())

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._recompute)
        layout.addWidget(self.apply_btn)
        self.status = QLabel("Load a recording to begin.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#9aa0aa; font-size:11px;")
        layout.addWidget(self.status)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setFixedWidth(320)
        return scroll

    def _build_filter_box(self) -> QGroupBox:
        box = QGroupBox("Filters")
        box.setCheckable(True)
        box.setChecked(True)
        box.toggled.connect(self._schedule)
        self.filter_box = box
        form = QFormLayout(box)
        self.bp_chk = QCheckBox("Band-pass")
        self.bp_chk.setChecked(True)
        self.bp_chk.toggled.connect(self._schedule)
        form.addRow(self.bp_chk)
        self.low_spin = self._dspin(0.1, 95.0, 1.0, " Hz")
        form.addRow("Low cut", self.low_spin)
        self.high_spin = self._dspin(1.0, 99.0, 45.0, " Hz")
        form.addRow("High cut", self.high_spin)
        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 8)
        self.order_spin.setValue(4)
        self.order_spin.valueChanged.connect(self._schedule)
        form.addRow("Order", self.order_spin)
        self.type_combo = QComboBox()
        self.type_combo.addItems(list(FILTER_TYPES.keys()))
        self.type_combo.currentTextChanged.connect(self._schedule)
        form.addRow("Type", self.type_combo)
        self.notch_chk = QCheckBox("Mains notch")
        self.notch_chk.setChecked(True)
        self.notch_chk.toggled.connect(self._schedule)
        form.addRow(self.notch_chk)
        self.notch_combo = QComboBox()
        self.notch_combo.addItems(["50 Hz", "60 Hz"])
        self.notch_combo.currentIndexChanged.connect(self._schedule)
        form.addRow("Notch freq", self.notch_combo)
        return box

    def _build_wavelet_box(self) -> QGroupBox:
        box = QGroupBox("Wavelet denoising")
        box.setCheckable(True)
        box.setChecked(False)
        box.toggled.connect(self._schedule)
        self.wavelet_box = box
        form = QFormLayout(box)
        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(proc.WAVELETS)
        self.wavelet_combo.currentTextChanged.connect(self._schedule)
        form.addRow("Wavelet", self.wavelet_combo)
        self.wlevel_spin = QSpinBox()
        self.wlevel_spin.setRange(1, 8)
        self.wlevel_spin.setValue(3)
        self.wlevel_spin.valueChanged.connect(self._schedule)
        form.addRow("Level", self.wlevel_spin)
        self.wden_combo = QComboBox()
        self.wden_combo.addItems(["SURESHRINK", "VISUSHRINK"])
        self.wden_combo.currentTextChanged.connect(self._schedule)
        form.addRow("Denoising", self.wden_combo)
        self.wthr_combo = QComboBox()
        self.wthr_combo.addItems(["SOFT", "HARD"])
        self.wthr_combo.currentTextChanged.connect(self._schedule)
        form.addRow("Threshold", self.wthr_combo)
        return box

    def _build_asr_box(self) -> QGroupBox:
        box = QGroupBox("ASR")
        box.setCheckable(True)
        box.setChecked(False)
        box.toggled.connect(self._schedule)
        self.asr_box = box
        form = QFormLayout(box)
        self.asr_cutoff = self._dspin(1.0, 100.0, 20.0, " SD")
        form.addRow("Cutoff", self.asr_cutoff)
        if not self._avail.get("asr"):
            box.setChecked(False)
            box.setEnabled(False)
            box.setToolTip("Install 'meegkit' to enable ASR (pip install meegkit)")
        return box

    def _build_aas_box(self) -> QGroupBox:
        box = QGroupBox("ECG removal (R-peak AAS)")
        box.setCheckable(True)
        box.setChecked(False)
        box.toggled.connect(self._schedule)
        self.aas_box = box
        form = QFormLayout(box)
        self.aas_ref = QComboBox()
        self.aas_ref.currentIndexChanged.connect(self._schedule)
        form.addRow("ECG ref channel", self.aas_ref)
        self.aas_pre = self._dspin(50.0, 1000.0, 250.0, " ms")
        form.addRow("Window before R", self.aas_pre)
        self.aas_post = self._dspin(50.0, 1000.0, 450.0, " ms")
        form.addRow("Window after R", self.aas_post)
        self.aas_agg = QComboBox()
        self.aas_agg.addItems(["median", "mean"])
        self.aas_agg.currentTextChanged.connect(self._schedule)
        form.addRow("Template", self.aas_agg)
        if not self._avail.get("aas"):
            box.setChecked(False)
            box.setEnabled(False)
            box.setToolTip("Install 'neurokit2' to enable R-peak detection")
        return box

    def _dspin(self, lo, hi, val, suffix) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSuffix(suffix)
        s.valueChanged.connect(self._schedule)
        return s

    # ------------------------------------------------------------- views UI
    def _build_views(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("View"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Time", "PSD"])
        self.view_combo.currentTextChanged.connect(self._on_view_mode)
        bar.addWidget(self.view_combo)
        bar.addWidget(QLabel("Window"))
        self.window_spin = self._plain_dspin(1.0, 60.0, 10.0, " s")
        self.window_spin.valueChanged.connect(self._on_view_changed)
        bar.addWidget(self.window_spin)
        bar.addWidget(QLabel("Amplitude"))
        self.amp_spin = self._plain_dspin(1.0, 100000.0, 200.0, " uV")
        self.amp_spin.valueChanged.connect(self._redraw_all)
        bar.addWidget(self.amp_spin)
        bar.addStretch(1)
        layout.addLayout(bar)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self._orig_pw, self._orig_plot = self._make_plot("Original")
        self._proc_pw, self._proc_plot = self._make_plot("Processed")
        self._proc_plot.setXLink(self._orig_plot)
        splitter.addWidget(self._orig_pw)
        splitter.addWidget(self._proc_pw)
        splitter.setSizes([400, 400])
        layout.addWidget(splitter, 1)

        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar.valueChanged.connect(self._on_scroll)
        layout.addWidget(self.scrollbar)
        return container

    def _plain_dspin(self, lo, hi, val, suffix) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    def _make_plot(self, title: str):
        pw = pg.PlotWidget()
        plot = pw.getPlotItem()
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setTitle(title)
        plot.setMouseEnabled(x=False, y=False)
        plot.setMenuEnabled(False)
        return pw, plot

    # --------------------------------------------------------------- load
    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open recording", "recordings",
            "Recordings (*.csv *.fif *.edf *.bdf *.set *.gdf *.vhdr);;All files (*)",
        )
        if path:
            self._load(path)

    def _load(self, path: str) -> None:
        try:
            rec = load_recording(path)
        except LoadError as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        if rec.n_channels == 0 or rec.n_samples == 0:
            QMessageBox.warning(self, "Load failed", "Recording has no usable channels.")
            return
        self._rec = rec
        self._processed = rec.eeg.copy()
        dur = rec.n_samples / rec.sampling_rate
        self.file_label.setText(
            f"{os.path.basename(path)}\n{rec.n_channels} ch | {rec.sampling_rate} Hz | {dur:0.1f} s"
        )
        # populate AAS reference channel options
        self.aas_ref.blockSignals(True)
        self.aas_ref.clear()
        for i, name in enumerate(rec.channel_names):
            self.aas_ref.addItem(f"{name}", i)
        self.aas_ref.blockSignals(False)

        self._rebuild_curves()
        self.window_spin.setMaximum(max(2.0, dur))
        self._redraw_all()
        self._recompute()

    def _rebuild_curves(self) -> None:
        for c in self._orig_curves:
            self._orig_plot.removeItem(c)
        for c in self._proc_curves:
            self._proc_plot.removeItem(c)
        self._orig_curves = []
        self._proc_curves = []
        n = self._rec.n_channels if self._rec else 0
        for i in range(n):
            color = cfg.CHANNEL_COLORS[i % len(cfg.CHANNEL_COLORS)]
            self._orig_curves.append(self._orig_plot.plot(pen=pg.mkPen(color, width=1)))
            self._proc_curves.append(self._proc_plot.plot(pen=pg.mkPen(color, width=1)))

    # ----------------------------------------------------------- recompute
    def _schedule(self, *_args) -> None:
        if self._rec is not None:
            self._debounce.start()

    def _build_config(self) -> proc.ProcessingConfig:
        c = proc.ProcessingConfig()
        c.reref_car = self.reref_chk.isChecked()
        c.detrend = self.detrend_combo.currentText()
        c.filters.enabled = self.filter_box.isChecked()
        c.filters.bandpass_enabled = self.bp_chk.isChecked()
        c.filters.bp_low = self.low_spin.value()
        c.filters.bp_high = self.high_spin.value()
        c.filters.order = self.order_spin.value()
        c.filters.filter_type = self.type_combo.currentText()
        c.filters.notch_enabled = self.notch_chk.isChecked()
        c.filters.notch_freq = 50 if self.notch_combo.currentIndex() == 0 else 60
        c.wavelet.enabled = self.wavelet_box.isChecked()
        c.wavelet.wavelet = self.wavelet_combo.currentText()
        c.wavelet.level = self.wlevel_spin.value()
        c.wavelet.denoising = self.wden_combo.currentText()
        c.wavelet.threshold = self.wthr_combo.currentText()
        c.asr.enabled = self.asr_box.isChecked() and self.asr_box.isEnabled()
        c.asr.cutoff = self.asr_cutoff.value()
        c.aas.enabled = self.aas_box.isChecked() and self.aas_box.isEnabled()
        c.aas.ref_channel = max(0, self.aas_ref.currentIndex())
        c.aas.pre_ms = self.aas_pre.value()
        c.aas.post_ms = self.aas_post.value()
        c.aas.aggregation = self.aas_agg.currentText()
        return c

    def _recompute(self) -> None:
        if self._rec is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._debounce.start()  # retry shortly
            return
        self.status.setText("Processing...")
        self.apply_btn.setEnabled(False)
        config = self._build_config()
        self._worker = ProcessingWorker(self._rec.eeg, self._rec.sampling_rate,
                                        self._rec.channel_names, config)
        self._worker.done.connect(self._on_processed)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_processed(self, processed, messages: list) -> None:
        self._processed = processed
        self.apply_btn.setEnabled(True)
        self.status.setText("\n".join(messages) if messages else "No steps enabled.")
        self._redraw(self._proc_plot, self._proc_curves, processed)

    def _on_failed(self, message: str) -> None:
        self.apply_btn.setEnabled(True)
        self.status.setText(f"Processing failed: {message}")

    # ------------------------------------------------------------- drawing
    def _on_view_mode(self, *_args) -> None:
        psd = self.view_combo.currentText() == "PSD"
        for plot in (self._orig_plot, self._proc_plot):
            plot.setLogMode(x=False, y=psd)
            plot.setLabel("bottom", "Frequency" if psd else "Time", units="Hz" if psd else "s")
        self.scrollbar.setEnabled(not psd)
        self.amp_spin.setEnabled(not psd)
        self._redraw_all()

    def _redraw_all(self, *_args) -> None:
        if self._rec is None:
            return
        self._redraw(self._orig_plot, self._orig_curves, self._rec.eeg)
        if self._processed is not None:
            self._redraw(self._proc_plot, self._proc_curves, self._processed)
        self._configure_scrollbar()
        self._on_view_changed()

    def _redraw(self, plot, curves, data) -> None:
        if data is None or self._rec is None or data.shape[0] != len(curves):
            return
        sr = self._rec.sampling_rate
        if self.view_combo.currentText() == "PSD":
            for i in range(len(curves)):
                freqs, amps = compute_psd(np.ascontiguousarray(data[i]), sr)
                if freqs.size:
                    m = freqs <= 70.0
                    curves[i].setData(freqs[m], amps[m])
                else:
                    curves[i].clear()
            plot.getAxis("left").setTicks(None)
            plot.setLabel("left", "PSD", units="uV^2/Hz")
            plot.enableAutoRange()
            return
        # time view: stacked offsets
        t = np.arange(data.shape[1]) / sr
        amp = self.amp_spin.value()
        spacing = amp * 2.2
        ticks = []
        n = len(curves)
        for i in range(n):
            offset = (n - 1 - i) * spacing
            curves[i].setData(t, data[i] + offset)
            curves[i].setDownsampling(auto=True, method="peak")
            curves[i].setClipToView(True)
            name = self._rec.channel_names[i] if i < len(self._rec.channel_names) else f"Ch{i+1}"
            ticks.append((offset, name))
        plot.getAxis("left").setTicks([ticks])
        plot.setLabel("left", "")
        plot.setYRange(-spacing, n * spacing, padding=0)

    def _configure_scrollbar(self) -> None:
        if self._rec is None:
            return
        dur = self._rec.n_samples / self._rec.sampling_rate
        window = self.window_spin.value()
        self.scrollbar.blockSignals(True)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(int(max(0.0, dur - window) * 1000))
        self.scrollbar.setPageStep(int(window * 1000))
        self.scrollbar.blockSignals(False)

    def _on_scroll(self, *_args) -> None:
        self._on_view_changed()

    def _on_view_changed(self, *_args) -> None:
        if self._rec is None or self.view_combo.currentText() == "PSD":
            return
        self._configure_scrollbar()
        start = self.scrollbar.value() / 1000.0
        window = self.window_spin.value()
        self._orig_plot.setXRange(start, start + window, padding=0)  # linked -> proc
