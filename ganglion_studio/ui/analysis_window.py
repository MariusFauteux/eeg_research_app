"""Tabbed analysis report: EEG analysis, electrode characterization, comparison."""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio.core import analysis as A
from ganglion_studio.core.analysis import ChannelMeta


class FigurePanel(QWidget):
    """A single matplotlib figure with toolbar and an individual Save button."""

    def __init__(self, key: str, fig: Figure) -> None:
        super().__init__()
        self.key = key
        self._fig = fig
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setMinimumHeight(360)
        toolbar = NavigationToolbar2QT(self._canvas, self)
        bar = QHBoxLayout()
        bar.addWidget(toolbar)
        bar.addStretch(1)
        save_btn = QPushButton("Save figure...")
        save_btn.clicked.connect(self._save)
        bar.addWidget(save_btn)
        layout.addLayout(bar)
        layout.addWidget(self._canvas)

    def figure(self) -> Figure:
        return self._fig

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", f"{self.key}.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if not path:
            return
        try:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
        except Exception as exc:  # pragma: no cover
            QMessageBox.warning(self, "Save failed", str(exc))


class AnalysisWindow(QMainWindow):
    def __init__(self, processed: np.ndarray, original: np.ndarray,
                 sampling_rate: int, metas: List[ChannelMeta],
                 title: str = "Recording") -> None:
        super().__init__()
        self.setWindowTitle(f"Analysis - {title}")
        self.resize(1100, 820)

        self._processed = processed
        self._original = original
        self._sr = sampling_rate
        self._metas = metas
        self._panels: List[Tuple[str, FigurePanel]] = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.addLayout(self._build_toolbar())

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self._rebuild()

    # ------------------------------------------------------------- toolbar
    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Source"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Processed", "Original"])
        self.source_combo.currentTextChanged.connect(self._rebuild)
        bar.addWidget(self.source_combo)

        # General A vs B channel selectors for the comparison tab.
        self.chan_a_combo = QComboBox()
        self.chan_b_combo = QComboBox()
        self.chan_a_combo.currentIndexChanged.connect(self._rebuild_comparison_only)
        self.chan_b_combo.currentIndexChanged.connect(self._rebuild_comparison_only)
        self._pair_label = QLabel("Compare:")
        bar.addWidget(self._pair_label)
        bar.addWidget(QLabel("A"))
        bar.addWidget(self.chan_a_combo)
        bar.addWidget(QLabel("vs B"))
        bar.addWidget(self.chan_b_combo)
        bar.addStretch(1)

        save_all = QPushButton("Save all figures...")
        save_all.clicked.connect(self._save_all)
        bar.addWidget(save_all)
        self._populate_pair_combos()
        return bar

    def _populate_pair_combos(self) -> None:
        eeg = A.eeg_metas(self._metas)
        for combo in (self.chan_a_combo, self.chan_b_combo):
            combo.blockSignals(True)
            combo.clear()
            for m in eeg:
                combo.addItem(f"{m.name} ({m.electrode})", m.index)
            combo.blockSignals(False)
        # Defaults: A = first PEDOT, B = first Ag/AgCl, else first two channels.
        pedots = [i for i, m in enumerate(eeg) if m.is_pedot]
        agagcls = [i for i, m in enumerate(eeg) if m.is_agagcl]
        self.chan_a_combo.blockSignals(True)
        self.chan_b_combo.blockSignals(True)
        if pedots:
            self.chan_a_combo.setCurrentIndex(pedots[0])
        if agagcls:
            self.chan_b_combo.setCurrentIndex(agagcls[0])
        elif self.chan_b_combo.count() > 1 and self.chan_b_combo.currentIndex() == self.chan_a_combo.currentIndex():
            self.chan_b_combo.setCurrentIndex(1)
        self.chan_a_combo.blockSignals(False)
        self.chan_b_combo.blockSignals(False)
        show = len(eeg) >= 2
        for w in (self._pair_label, self.chan_a_combo, self.chan_b_combo):
            w.setVisible(show)

    # --------------------------------------------------------------- data
    def _eeg(self) -> np.ndarray:
        if self.source_combo.currentText() == "Original":
            return self._original
        return self._processed

    def _selected_pair(self) -> Optional[Tuple[int, int, str, str]]:
        if self.chan_a_combo.count() == 0 or self.chan_b_combo.count() == 0:
            return None
        ai = self.chan_a_combo.currentData()
        bi = self.chan_b_combo.currentData()
        if ai is None or bi is None:
            return None
        return ai, bi, self.chan_a_combo.currentText(), self.chan_b_combo.currentText()

    # ------------------------------------------------------------- build
    def _rebuild(self, *_args) -> None:
        if not hasattr(self, "tabs"):
            return
        current = self.tabs.currentIndex()
        self.tabs.clear()
        self._panels = []
        eeg = self._eeg()
        sr = self._sr
        metas = self._metas

        self._add_tab("EEG analysis", [
            ("eeg_psd", A.fig_psd(eeg, sr, metas)),
            ("eeg_band_power", A.fig_band_powers(eeg, sr, metas)),
            ("eeg_quality_table", A.fig_quality_table(eeg, sr, metas)),
        ])
        self._add_tab("Electrode characterization", [
            ("char_noise", A.fig_char_noise(eeg, sr, metas)),
            ("char_psd_by_material", A.fig_char_psd_by_material(eeg, sr, metas)),
        ])
        if len(A.eeg_metas(metas)) >= 2:
            self._add_tab("Compare channels", self._compare_figures(eeg, sr))
        if A.comparison_available(metas):
            self._add_tab("Material groups", [
                ("group_psd", A.fig_cmp_psd(eeg, sr, metas)),
                ("group_band_power", A.fig_cmp_bandpower(eeg, sr, metas)),
            ])

        if 0 <= current < self.tabs.count():
            self.tabs.setCurrentIndex(current)

    def _rebuild_comparison_only(self, *_args) -> None:
        # Rebuild everything (cheap) so paired figures reflect the new selection.
        self._rebuild()

    def _compare_figures(self, eeg, sr) -> List[Tuple[str, Figure]]:
        pair = self._selected_pair()
        if not pair:
            return []
        ai, bi, alabel, blabel = pair
        ag = A.pair_agreement(eeg[ai], eeg[bi], sr)
        return [
            ("cmp_stats_table", A.fig_pair_stats_table(eeg[ai], eeg[bi], sr, ag, alabel, blabel)),
            ("cmp_timeseries", A.fig_pair_timeseries(eeg[ai], eeg[bi], sr, alabel, blabel)),
            ("cmp_psd_overlay", A.fig_pair_psd(eeg[ai], eeg[bi], sr, alabel, blabel)),
            ("cmp_coherence", A.fig_cmp_coherence(eeg[ai], eeg[bi], sr, ag, alabel, blabel)),
            ("cmp_correlation", A.fig_cmp_correlation(eeg[ai], eeg[bi], ag, alabel, blabel)),
            ("cmp_bland_altman", A.fig_cmp_bland_altman(eeg[ai], eeg[bi], ag, alabel, blabel)),
            ("cmp_cross_correlation", A.fig_pair_cross_correlation(eeg[ai], eeg[bi], sr, alabel, blabel)),
            ("cmp_histogram", A.fig_pair_histogram(eeg[ai], eeg[bi], alabel, blabel)),
            ("cmp_band_power_ratio", A.fig_pair_bandpower_ratio(eeg[ai], eeg[bi], sr, alabel, blabel)),
        ]

    def _add_tab(self, title: str, figures: List[Tuple[str, Figure]]) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        for key, fig in figures:
            panel = FigurePanel(key, fig)
            self._panels.append((key, panel))
            layout.addWidget(panel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.tabs.addTab(scroll, title)

    # --------------------------------------------------------------- save
    def _save_all(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Save all figures to folder")
        if not directory:
            return
        saved = 0
        for key, panel in self._panels:
            try:
                panel.figure().savefig(os.path.join(directory, f"{key}.png"),
                                       dpi=150, bbox_inches="tight")
                saved += 1
            except Exception:
                pass
        QMessageBox.information(self, "Saved", f"Saved {saved} figures to:\n{directory}")
