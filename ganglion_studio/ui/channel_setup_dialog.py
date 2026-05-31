"""Dialog to set per-channel placement (10-20), type and electrode material.

The chosen placement becomes the channel's display name across the session;
the full configuration is also stored on the board manager and written into the
recording metadata. A montage can be saved/loaded as a reusable JSON preset.
"""

from __future__ import annotations

import json
from typing import List

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ganglion_studio.core import board_config as cfg
from ganglion_studio.core.board_manager import BoardManager


class ChannelSetupDialog(QDialog):
    def __init__(self, manager: BoardManager, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Channel setup")
        self._manager = manager
        self._n = len(manager.eeg_channels)
        self._place_combos: List[QComboBox] = []
        self._type_combos: List[QComboBox] = []
        self._elec_combos: List[QComboBox] = []

        root = QVBoxLayout(self)
        hint = QLabel("Set the 10-20 placement, signal type and electrode for each channel. "
                      "The placement is used as the channel name.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9aa0aa;")
        root.addWidget(hint)

        grid = QGridLayout()
        for c, header in enumerate(["Channel", "Placement (10-20)", "Type", "Electrode"]):
            lab = QLabel(header)
            lab.setStyleSheet("font-weight:600;")
            grid.addWidget(lab, 0, c)

        for i in range(self._n):
            grid.addWidget(QLabel(f"Ch{i + 1}"), i + 1, 0)

            place = QComboBox()
            place.addItems(cfg.TEN_TWENTY)
            place.setCurrentText(manager.placements[i] if i < len(manager.placements) else "None")
            grid.addWidget(place, i + 1, 1)
            self._place_combos.append(place)

            tcombo = QComboBox()
            tcombo.addItems(cfg.CHANNEL_TYPES)
            tcombo.setCurrentText(manager.channel_types[i] if i < len(manager.channel_types) else "EEG")
            grid.addWidget(tcombo, i + 1, 2)
            self._type_combos.append(tcombo)

            ecombo = QComboBox()
            ecombo.addItems(cfg.ELECTRODES)
            ecombo.setCurrentText(manager.electrodes[i] if i < len(manager.electrodes) else cfg.ELECTRODES[0])
            grid.addWidget(ecombo, i + 1, 3)
            self._elec_combos.append(ecombo)
        root.addLayout(grid)

        preset_row = QHBoxLayout()
        save_btn = QPushButton("Save montage...")
        save_btn.clicked.connect(self._save_montage)
        load_btn = QPushButton("Load montage...")
        load_btn.clicked.connect(self._load_montage)
        preset_row.addWidget(save_btn)
        preset_row.addWidget(load_btn)
        preset_row.addStretch(1)
        root.addLayout(preset_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ------------------------------------------------------------- values
    def names(self) -> List[str]:
        """Display names: placement when set, else Ch#."""
        out = []
        for i in range(self._n):
            place = self._place_combos[i].currentText()
            out.append(place if place not in ("None", "Custom") else f"Ch{i + 1}")
        return out

    def placements(self) -> List[str]:
        return [c.currentText() for c in self._place_combos]

    def types(self) -> List[str]:
        return [c.currentText() for c in self._type_combos]

    def electrodes(self) -> List[str]:
        return [c.currentText() for c in self._elec_combos]

    # ------------------------------------------------------------- presets
    def _save_montage(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save montage", "montage.json", "JSON (*.json)")
        if not path:
            return
        data = {
            "placements": self.placements(),
            "types": self.types(),
            "electrodes": self.electrodes(),
        }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:  # pragma: no cover
            QMessageBox.warning(self, "Save failed", str(exc))

    def _load_montage(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load montage", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # pragma: no cover
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        for i in range(self._n):
            for key, combos in (("placements", self._place_combos),
                                ("types", self._type_combos),
                                ("electrodes", self._elec_combos)):
                vals = data.get(key, [])
                if i < len(vals):
                    combos[i].setCurrentText(str(vals[i]))
