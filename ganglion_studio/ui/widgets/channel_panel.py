"""Channel enable/disable panel. Sends Ganglion commands and toggles plotting."""

from __future__ import annotations

from typing import List

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio.core import board_config as cfg
from ganglion_studio.core.board_manager import BoardManager


class ChannelPanel(QGroupBox):
    channels_changed = pyqtSignal(list)  # list[bool]

    def __init__(self, manager: BoardManager) -> None:
        super().__init__("Channels")
        self._manager = manager
        self._checks: List[QCheckBox] = []
        layout = QVBoxLayout(self)

        names = cfg.DEFAULT_CHANNEL_NAMES
        for i, _ch in enumerate(manager.eeg_channels):
            row = QHBoxLayout()
            swatch = QLabel("  ")
            color = cfg.CHANNEL_COLORS[i % len(cfg.CHANNEL_COLORS)]
            swatch.setStyleSheet(f"background:{color}; border-radius:3px;")
            swatch.setFixedSize(14, 14)
            chk = QCheckBox(names[i] if i < len(names) else f"Ch{i+1}")
            chk.setChecked(True)
            chk.toggled.connect(lambda state, idx=i: self._on_toggle(idx, state))
            self._checks.append(chk)
            row.addWidget(swatch)
            row.addWidget(chk)
            row.addStretch(1)
            layout.addLayout(row)
        layout.addStretch(1)

    def _on_toggle(self, index: int, state: bool) -> None:
        self._manager.set_channel_active(index, state)
        self.channels_changed.emit(self.active_channels())

    def active_channels(self) -> List[bool]:
        return [c.isChecked() for c in self._checks]
