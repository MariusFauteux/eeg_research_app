"""Display-filter panel: band-pass, notch and detrend (applied to plots only)."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
)

from ganglion_studio.core.dsp import FILTER_TYPES, FilterSettings


class FilterPanel(QGroupBox):
    filters_changed = pyqtSignal(object)  # FilterSettings

    def __init__(self, notch_default: int = 50) -> None:
        super().__init__("Filters (display)")
        self._settings = FilterSettings(notch_freq=notch_default)
        root = QVBoxLayout(self)

        self.detrend_chk = QCheckBox("Detrend (remove drift)")
        self.detrend_chk.setChecked(self._settings.detrend)
        self.detrend_chk.toggled.connect(self._emit)
        root.addWidget(self.detrend_chk)

        # --- band-pass ---
        self.bp_chk = QCheckBox("Band-pass")
        self.bp_chk.setChecked(self._settings.bandpass_enabled)
        self.bp_chk.toggled.connect(self._emit)
        root.addWidget(self.bp_chk)

        bp_form = QFormLayout()
        self.low_spin = QDoubleSpinBox()
        self.low_spin.setRange(0.1, 95.0)
        self.low_spin.setValue(self._settings.bp_low)
        self.low_spin.setSuffix(" Hz")
        self.low_spin.valueChanged.connect(self._emit)
        bp_form.addRow("Low cut", self.low_spin)

        self.high_spin = QDoubleSpinBox()
        self.high_spin.setRange(1.0, 99.0)
        self.high_spin.setValue(self._settings.bp_high)
        self.high_spin.setSuffix(" Hz")
        self.high_spin.valueChanged.connect(self._emit)
        bp_form.addRow("High cut", self.high_spin)

        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 8)
        self.order_spin.setValue(self._settings.order)
        self.order_spin.valueChanged.connect(self._emit)
        bp_form.addRow("Order", self.order_spin)

        self.type_combo = QComboBox()
        self.type_combo.addItems(list(FILTER_TYPES.keys()))
        self.type_combo.currentTextChanged.connect(self._emit)
        bp_form.addRow("Type", self.type_combo)
        root.addLayout(bp_form)

        # --- notch ---
        self.notch_chk = QCheckBox("Mains notch")
        self.notch_chk.setChecked(self._settings.notch_enabled)
        self.notch_chk.toggled.connect(self._emit)
        root.addWidget(self.notch_chk)

        notch_form = QFormLayout()
        self.notch_combo = QComboBox()
        self.notch_combo.addItems(["50 Hz", "60 Hz"])
        self.notch_combo.setCurrentIndex(0 if notch_default == 50 else 1)
        self.notch_combo.currentIndexChanged.connect(self._emit)
        notch_form.addRow("Frequency", self.notch_combo)
        root.addLayout(notch_form)
        root.addStretch(1)

    def _emit(self, *_args) -> None:
        self._settings = FilterSettings(
            detrend=self.detrend_chk.isChecked(),
            bandpass_enabled=self.bp_chk.isChecked(),
            bp_low=self.low_spin.value(),
            bp_high=self.high_spin.value(),
            order=self.order_spin.value(),
            filter_type=self.type_combo.currentText(),
            notch_enabled=self.notch_chk.isChecked(),
            notch_freq=50 if self.notch_combo.currentIndex() == 0 else 60,
        )
        self.filters_changed.emit(self._settings)

    def settings(self) -> FilterSettings:
        return self._settings
