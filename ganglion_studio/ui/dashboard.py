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
    QInputDialog,
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
from ganglion_studio.core import ble_scanner, saved_devices
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

        # Saved devices: reconnect without scanning every time.
        saved_row = QHBoxLayout()
        saved_row.addWidget(QLabel("Saved"))
        self.saved_combo = QComboBox()
        self.saved_combo.setToolTip("Pick a saved board to skip scanning")
        self.saved_combo.currentIndexChanged.connect(self._on_saved_selected)
        saved_row.addWidget(self.saved_combo, 1)
        self.save_btn = QPushButton("Save")
        self.save_btn.setToolTip("Save the selected (or entered) device for one-click reconnect")
        self.save_btn.clicked.connect(self._save_current_device)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setToolTip("Remove the selected saved device")
        self.remove_btn.clicked.connect(self._remove_saved_device)
        saved_row.addWidget(self.save_btn)
        saved_row.addWidget(self.remove_btn)
        scan_layout.addLayout(saved_row)

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
        self.scan_box = scan_box
        body.addWidget(scan_box, 2)

        # --- Session setup ----------------------------------------------
        setup_box = QGroupBox("2. Session settings")
        form = QFormLayout(setup_box)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. alpha-baseline-subject01")
        self.name_edit.textChanged.connect(self._update_start_enabled)
        form.addRow("Session name", self.name_edit)

        self.conn_combo = QComboBox()
        # "Custom" = our bleak driver (no once-per-second pulse); "BrainFlow" =
        # the stock native backend, kept for comparison/fallback.
        self.conn_combo.addItems(
            ["Native Bluetooth (Custom)", "Native Bluetooth (BrainFlow)", "Dongle (BLED112)"]
        )
        self.conn_combo.setToolTip(
            "Custom: bypasses BrainFlow's native BLE to avoid the 1 Hz pulse.\n"
            "BrainFlow: stock native backend (for A/B comparison).\n"
            "Dongle: OpenBCI BLED112 USB dongle."
        )
        self.conn_combo.currentTextChanged.connect(self._update_connection_ui)
        form.addRow("Connection", self.conn_combo)

        self.mac_edit = QLineEdit()
        self.mac_edit.setPlaceholderText("auto-discover (optional)")
        form.addRow("MAC / address", self.mac_edit)

        # Dongle serial-port picker (hidden unless Connection = Dongle).
        port_row = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.currentIndexChanged.connect(self._update_start_enabled)
        self.refresh_ports_btn = QPushButton("Refresh")
        self.refresh_ports_btn.clicked.connect(self._refresh_ports)
        port_row.addWidget(self.port_combo, 1)
        port_row.addWidget(self.refresh_ports_btn)
        self._port_row = QWidget()
        self._port_row.setLayout(port_row)
        self._form = form
        form.addRow("Dongle port", self._port_row)
        form.setRowVisible(self._port_row, False)

        self.fw_combo = QComboBox()
        self.fw_combo.addItems(["3 (default)", "2 (legacy)", "auto"])
        form.addRow("Firmware", self.fw_combo)

        self.decode_combo = QComboBox()
        self.decode_combo.addItems(["Delta (firmware ≤ 2.x)", "MSB (firmware 3.0.2+)"])
        self.decode_combo.setToolTip(
            "Native-Bluetooth sample encoding. Firmware 3.0.2+ sends absolute MSB "
            "samples; older firmware sends deltas. Pick the one matching your board "
            "(wrong choice looks like noise). Ignored for the dongle."
        )
        form.addRow("Decoding", self.decode_combo)

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

        self._refresh_saved()

    # --------------------------------------------------------- saved devices
    def _refresh_saved(self, select_address: Optional[str] = None) -> None:
        """Reload the saved-device dropdown from disk."""
        self.saved_combo.blockSignals(True)
        self.saved_combo.clear()
        self.saved_combo.addItem("— saved devices —", None)
        for dev in saved_devices.load():
            self.saved_combo.addItem(f"{dev.name}  ({dev.address})", dev.address)
        self.saved_combo.blockSignals(False)
        if select_address:
            idx = self.saved_combo.findData(select_address)
            if idx > 0:
                self.saved_combo.setCurrentIndex(idx)  # fires _on_saved_selected
        self.remove_btn.setEnabled(bool(self.saved_combo.currentData()))

    def _on_saved_selected(self, _idx: int = 0) -> None:
        address = self.saved_combo.currentData()
        self.remove_btn.setEnabled(bool(address))
        if not address:
            return
        # Saved devices are native BLE; make sure we're not in dongle mode.
        if self.conn_combo.currentText().startswith("Dongle"):
            self.conn_combo.setCurrentIndex(0)
        self.mac_edit.setText(address)
        label = self.saved_combo.currentText().split("  (")[0]
        if not self.name_edit.text():
            self.name_edit.setText(label.replace(" ", "_"))
        self.scan_status.setText(f"Using saved device '{label}'. Click Start Session.")
        self._update_start_enabled()

    def _save_current_device(self) -> None:
        address = self.mac_edit.text().strip()
        if not address:
            QMessageBox.information(
                self, "Save device",
                "Select a scanned device (or type an address) first.",
            )
            return
        default = ""
        items = self.device_list.selectedItems()
        if items:
            default = items[0].data(Qt.ItemDataRole.UserRole).name
        default = default or self.name_edit.text().strip() or "Ganglion"
        label, ok = QInputDialog.getText(self, "Save device", "Label:", text=default)
        if not ok:
            return
        saved_devices.add(label, address)
        self._refresh_saved(select_address=address)
        self.scan_status.setText(f"Saved '{label.strip() or address}'.")

    def _remove_saved_device(self) -> None:
        address = self.saved_combo.currentData()
        if not address:
            return
        saved_devices.remove(address)
        self._refresh_saved()

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
        if checked and not self.name_edit.text():
            self.name_edit.setText("demo-session")
        self._update_connection_ui()

    def _update_connection_ui(self, *_args) -> None:
        """Enable only the controls relevant to the chosen connection mode."""
        demo = self.demo_check.isChecked()
        dongle = self.conn_combo.currentText().startswith("Dongle")
        native_live = (not demo) and (not dongle)
        # The host BLE scan + MAC are native-only. The dongle does its OWN
        # discovery, and on macOS the scan returns a CoreBluetooth UUID (not a
        # real MAC) that the dongle cannot connect to -- so scanning is disabled
        # in dongle mode to avoid a doomed connection attempt.
        self.scan_box.setEnabled(native_live)
        self.mac_edit.setEnabled(native_live)
        self.fw_combo.setEnabled(not demo)  # firmware applies to the dongle too
        self.decode_combo.setEnabled(native_live)  # custom native driver only
        self._form.setRowVisible(self._port_row, dongle and not demo)
        if dongle and not demo:
            if self.port_combo.count() == 0:
                self._refresh_ports()
            self.scan_status.setText(
                "Dongle mode: pick the OpenBCI BLED112 port at right (a 'usbserial' "
                "/ Silicon Labs CP210x device). No scan needed - it finds the board."
            )
        elif native_live:
            self.scan_status.setText("Idle. Click Scan, or use Demo mode.")
        self._update_start_enabled()

    def _refresh_ports(self) -> None:
        from ganglion_studio.core.serial_ports import list_serial_ports
        self.port_combo.clear()
        for device, desc in list_serial_ports():
            self.port_combo.addItem(f"{device}  ({desc})", device)
        if self.port_combo.count() == 0:
            self.port_combo.addItem("No serial ports found", "")
        self._update_start_enabled()

    def _update_start_enabled(self) -> None:
        has_name = bool(self.name_edit.text().strip())
        if self.demo_check.isChecked():
            ready = True
        elif self.conn_combo.currentText().startswith("Dongle"):
            ready = bool(self.port_combo.currentData())
        else:
            ready = bool(self.device_list.selectedItems()) or bool(self.mac_edit.text().strip())
        self.start_btn.setEnabled(has_name and ready)

    def _on_start(self) -> None:
        fw = self.fw_combo.currentText().split()[0]
        notch = 50 if self.notch_combo.currentIndex() == 0 else 60
        dongle = self.conn_combo.currentText().startswith("Dongle")
        config = SessionConfig(
            name=self.name_edit.text().strip(),
            demo=self.demo_check.isChecked(),
            # The dongle auto-discovers its board; never pass the host scan's
            # macOS UUID as a MAC (the dongle can't connect to it).
            mac_address="" if dongle else self.mac_edit.text().strip(),
            serial_port=(self.port_combo.currentData() or "") if dongle else "",
            firmware=fw,
            notch_freq=notch,
            use_custom_native=self.conn_combo.currentText().endswith("(Custom)"),
            decode_mode="msb" if self.decode_combo.currentIndex() == 1 else "delta",
        )
        self.start_session.emit(config)
