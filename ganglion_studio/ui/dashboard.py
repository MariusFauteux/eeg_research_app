"""Dashboard: scan for the Ganglion over native BLE, name the session, start."""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio import palette
from ganglion_studio.core import ble_scanner
from ganglion_studio.core.ble_scanner import BleDevice, BleUnavailable
from ganglion_studio.core.session import SessionConfig
from ganglion_studio.ui import theme


class ScanWorker(QThread):
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, timeout: float, ganglion_only: bool) -> None:
        super().__init__()
        self._timeout = timeout
        self._ganglion_only = ganglion_only

    def run(self) -> None:  # noqa: D401 - QThread entry
        try:
            devices = ble_scanner.scan(self._timeout, self._ganglion_only)
            self.finished_ok.emit(devices)
        except BleUnavailable as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(f"Unexpected scan error: {exc}")


class Dashboard(QWidget):
    """Connection / session-setup screen."""

    start_session = pyqtSignal(object)  # emits SessionConfig
    open_processing = pyqtSignal()  # request to open the Processing Lab

    def __init__(self) -> None:
        super().__init__()
        self._worker: Optional[ScanWorker] = None
        self._devices: List[BleDevice] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(16)

        title = QLabel("Ganglion EEG Studio")
        title.setStyleSheet(f"font-size: 26px; font-weight: 700; color: {palette.ACCENT};")
        subtitle = QLabel("Connect to your OpenBCI Ganglion over native Bluetooth")
        subtitle.setStyleSheet(theme.MUTED_QSS)
        header = QHBoxLayout()
        header_text = QVBoxLayout()
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header.addLayout(header_text)
        header.addStretch(1)
        self.processing_btn = QPushButton("Processing Lab")
        self.processing_btn.setToolTip("Open the offline processing playground (no board needed)")
        self.processing_btn.clicked.connect(self.open_processing.emit)
        header.addWidget(self.processing_btn, alignment=Qt.AlignmentFlag.AlignTop)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(20)
        root.addLayout(body, 1)

        # --- Device discovery -------------------------------------------
        scan_box = QGroupBox("1. Find your board")
        scan_layout = QVBoxLayout(scan_box)
        scan_controls = QHBoxLayout()
        self.scan_btn = QPushButton("Scan Bluetooth")
        self.scan_btn.clicked.connect(self._start_scan)
        self.ganglion_only = QCheckBox("Ganglion only")
        self.ganglion_only.setChecked(True)
        scan_controls.addWidget(self.scan_btn)
        scan_controls.addWidget(self.ganglion_only)
        scan_controls.addStretch(1)
        scan_layout.addLayout(scan_controls)

        self.device_list = QListWidget()
        self.device_list.itemSelectionChanged.connect(self._on_device_selected)
        scan_layout.addWidget(self.device_list, 1)

        self.scan_status = QLabel("Idle. Click Scan, or use Demo mode.")
        self.scan_status.setStyleSheet(theme.MUTED_QSS)
        scan_layout.addWidget(self.scan_status)
        body.addWidget(scan_box, 2)

        # --- Session setup ----------------------------------------------
        setup_box = QGroupBox("2. Session settings")
        form = QFormLayout(setup_box)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. alpha-baseline-subject01")
        self.name_edit.textChanged.connect(self._update_start_enabled)
        form.addRow("Session name", self.name_edit)

        self.mac_edit = QLineEdit()
        self.mac_edit.setPlaceholderText("auto-discover (optional)")
        form.addRow("MAC / address", self.mac_edit)

        self.fw_combo = QComboBox()
        self.fw_combo.addItems(["3 (default)", "2 (legacy)", "auto"])
        form.addRow("Firmware", self.fw_combo)

        self.notch_combo = QComboBox()
        self.notch_combo.addItems(["50 Hz (EU)", "60 Hz (US)"])
        form.addRow("Mains notch", self.notch_combo)

        self.demo_check = QCheckBox("Demo mode (synthetic board, no hardware)")
        self.demo_check.toggled.connect(self._on_demo_toggled)
        form.addRow(self.demo_check)

        self.start_btn = QPushButton("Start Session  \u25B6")
        self.start_btn.setStyleSheet(
            "QPushButton { background:#2f7d4f; font-weight:600; padding:9px; }"
            "QPushButton:hover { background:#379159; }"
        )
        self.start_btn.clicked.connect(self._on_start)
        self.start_btn.setEnabled(False)
        form.addRow(self.start_btn)
        body.addWidget(setup_box, 1)

    # ------------------------------------------------------------- scanning
    def _start_scan(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self.device_list.clear()
        self._devices = []
        self.scan_btn.setEnabled(False)
        self.scan_status.setText("Scanning for Bluetooth devices (up to 8s)...")
        self._worker = ScanWorker(8.0, self.ganglion_only.isChecked())
        self._worker.finished_ok.connect(self._on_scan_done)
        self._worker.failed.connect(self._on_scan_failed)
        self._worker.start()

    def _on_scan_done(self, devices: List[BleDevice]) -> None:
        self._devices = devices
        self.scan_btn.setEnabled(True)
        for dev in devices:
            tag = "  [Ganglion]" if dev.is_ganglion else ""
            item = QListWidgetItem(f"{dev.name}  ({dev.address})  RSSI {dev.rssi}{tag}")
            item.setData(Qt.ItemDataRole.UserRole, dev)
            self.device_list.addItem(item)
        if not devices:
            self.scan_status.setText("No devices found. Power the Ganglion or try Demo mode.")
        else:
            self.scan_status.setText(f"Found {len(devices)} device(s). Select one.")

    def _on_scan_failed(self, message: str) -> None:
        self.scan_btn.setEnabled(True)
        self.scan_status.setText("Bluetooth unavailable - use Demo mode.")
        QMessageBox.warning(
            self, "Bluetooth scan failed",
            f"{message}\n\nYou can still explore the app using Demo mode.",
        )

    def _on_device_selected(self) -> None:
        items = self.device_list.selectedItems()
        if items:
            dev: BleDevice = items[0].data(Qt.ItemDataRole.UserRole)
            self.mac_edit.setText(dev.address)
            if not self.name_edit.text():
                self.name_edit.setText(dev.name.replace(" ", "_"))
        self._update_start_enabled()

    # ------------------------------------------------------------- session
    def _on_demo_toggled(self, checked: bool) -> None:
        self.scan_btn.setEnabled(not checked)
        self.device_list.setEnabled(not checked)
        self.mac_edit.setEnabled(not checked)
        self.fw_combo.setEnabled(not checked)
        if checked and not self.name_edit.text():
            self.name_edit.setText("demo-session")
        self._update_start_enabled()

    def _update_start_enabled(self) -> None:
        has_name = bool(self.name_edit.text().strip())
        ready = self.demo_check.isChecked() or bool(self.device_list.selectedItems()) or bool(self.mac_edit.text().strip())
        self.start_btn.setEnabled(has_name and ready)

    def _on_start(self) -> None:
        fw = self.fw_combo.currentText().split()[0]
        notch = 50 if self.notch_combo.currentIndex() == 0 else 60
        config = SessionConfig(
            name=self.name_edit.text().strip(),
            demo=self.demo_check.isChecked(),
            mac_address=self.mac_edit.text().strip(),
            firmware=fw,
            notch_freq=notch,
        )
        self.start_session.emit(config)
