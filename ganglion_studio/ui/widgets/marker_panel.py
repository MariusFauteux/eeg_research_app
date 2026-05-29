"""Marker / annotation panel for triggering experiment protocols.

Define marker types (label + numeric code + colour + hotkey), fire them with a
click or keyboard shortcut, and keep a timestamped event log that can be
exported to CSV. Each fire calls ``BoardManager.insert_marker`` so the code is
embedded in the recorded BrainFlow marker channel.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from typing import List

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

DEFAULT_MARKERS = [
    ("Eyes Open", 1, "#5fd38d"),
    ("Eyes Closed", 2, "#4f8ef7"),
    ("Stimulus", 3, "#e2c044"),
    ("Movement", 4, "#f7766f"),
    ("Rest", 5, "#9aa0aa"),
]


@dataclass
class MarkerType:
    label: str
    code: int
    color: str


class MarkerPanel(QGroupBox):
    marker_fired = pyqtSignal(int, str, float)  # code, label, timestamp

    def __init__(self, parent_widget: QWidget) -> None:
        super().__init__("Markers / Annotations")
        self._types: List[MarkerType] = [MarkerType(*m) for m in DEFAULT_MARKERS]
        self._parent = parent_widget
        self._shortcuts: List[QShortcut] = []

        root = QVBoxLayout(self)
        hint = QLabel("Click or press the number key to drop a marker.")
        hint.setStyleSheet("color:#9aa0aa;")
        root.addWidget(hint)

        self._buttons_box = QVBoxLayout()
        root.addLayout(self._buttons_box)

        add_btn = QPushButton("+ Add marker type")
        add_btn.clicked.connect(self._add_type)
        root.addWidget(add_btn)

        root.addWidget(QLabel("Event log"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Time", "Code", "Label"])
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, 1)

        export_btn = QPushButton("Export markers (CSV)")
        export_btn.clicked.connect(self._export)
        root.addWidget(export_btn)

        self._rebuild_buttons()

    # ------------------------------------------------------------ build UI
    def _rebuild_buttons(self) -> None:
        while self._buttons_box.count():
            item = self._buttons_box.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        for sc in self._shortcuts:
            sc.setParent(None)
        self._shortcuts.clear()

        for i, mt in enumerate(self._types):
            row = QHBoxLayout()
            btn = QPushButton(f"[{i+1}] {mt.label}  (code {mt.code})")
            btn.setStyleSheet(
                f"QPushButton {{ background:{mt.color}; color:#15171c; font-weight:600; }}"
            )
            btn.clicked.connect(lambda _checked, m=mt: self.fire(m))
            row.addWidget(btn)
            holder = QWidget()
            holder.setLayout(row)
            self._buttons_box.addWidget(holder)

            if i < 9:
                sc = QShortcut(QKeySequence(str(i + 1)), self._parent)
                sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
                sc.activated.connect(lambda m=mt: self.fire(m))
                self._shortcuts.append(sc)

    def _add_type(self) -> None:
        label, ok = QInputDialog.getText(self, "Add marker type", "Label:")
        if not ok or not label.strip():
            return
        code, ok = QInputDialog.getInt(self, "Add marker type", "Numeric code:", len(self._types) + 1, 1, 999)
        if not ok:
            return
        palette = ["#5fd38d", "#4f8ef7", "#e2c044", "#f7766f", "#b48ef7", "#f79edb"]
        color = palette[len(self._types) % len(palette)]
        self._types.append(MarkerType(label.strip(), code, color))
        self._rebuild_buttons()

    # ------------------------------------------------------------- firing
    def fire(self, marker: MarkerType) -> None:
        ts = time.time()
        self.marker_fired.emit(marker.code, marker.label, ts)
        self._append_log(ts, marker)

    def _append_log(self, ts: float, marker: MarkerType) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        tstr = time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int((ts % 1) * 1000):03d}"
        self.table.setItem(row, 0, QTableWidgetItem(tstr))
        self.table.setItem(row, 1, QTableWidgetItem(str(marker.code)))
        item = QTableWidgetItem(marker.label)
        item.setForeground(QColor(marker.color))
        self.table.setItem(row, 2, item)
        self.table.scrollToBottom()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export markers", "markers.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time", "code", "label"])
            for r in range(self.table.rowCount()):
                writer.writerow([self.table.item(r, c).text() for c in range(3)])
