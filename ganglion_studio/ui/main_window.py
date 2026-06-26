"""Main window: hosts the dashboard and the session view in a stack."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QStackedWidget,
)

from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.session import SessionConfig
from ganglion_studio.ui.dashboard import Dashboard
from ganglion_studio.ui.processing_window import ProcessingWindow
from ganglion_studio.ui.session_view import SessionView


class PrepareWorker(QThread):
    """Prepare + start the board off the UI thread (BLE connect can block)."""

    ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, manager: BoardManager) -> None:
        super().__init__()
        self._manager = manager

    def run(self) -> None:
        try:
            self._manager.prepare()
            self._manager.start()
            self.ok.emit()
        except Exception as exc:  # pragma: no cover - hardware path
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Ganglion EEG Studio")
        self.resize(1500, 950)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.dashboard = Dashboard()
        self.dashboard.start_session.connect(self._on_start_session)
        self.dashboard.open_processing.connect(self._open_processing_lab)
        self.stack.addWidget(self.dashboard)

        self.session_view: Optional[SessionView] = None
        self.manager: Optional[BoardManager] = None
        self._worker: Optional[PrepareWorker] = None
        self._progress: Optional[QProgressDialog] = None
        self._pending_config: Optional[SessionConfig] = None
        self._processing_windows: list[ProcessingWindow] = []

    def _open_processing_lab(self) -> None:
        window = ProcessingWindow()
        window.show()
        window.raise_()
        self._processing_windows.append(window)

    def _on_start_session(self, config: SessionConfig) -> None:
        self._pending_config = config
        self.manager = BoardManager(
            demo=config.demo,
            mac_address=config.mac_address,
            serial_port=config.serial_port,
            firmware=config.firmware,
            use_custom_native=config.use_custom_native,
            decode_mode=config.decode_mode,
        )
        self._progress = QProgressDialog(
            "Connecting to the board..." if not config.demo else "Starting demo board...",
            None, 0, 0, self,
        )
        self._progress.setWindowTitle("Please wait")
        self._progress.setCancelButton(None)
        self._progress.setMinimumDuration(0)
        self._progress.show()

        self._worker = PrepareWorker(self.manager)
        self._worker.ok.connect(self._on_prepared)
        self._worker.failed.connect(self._on_prepare_failed)
        self._worker.start()

    def _on_prepared(self) -> None:
        if self._progress:
            self._progress.close()
        assert self.manager is not None and self._pending_config is not None
        self.session_view = SessionView(self.manager, self._pending_config)
        self.session_view.exit_session.connect(self._on_exit_session)
        self.session_view.open_processing.connect(self._open_processing_lab)
        self.stack.addWidget(self.session_view)
        self.stack.setCurrentWidget(self.session_view)

    def _on_prepare_failed(self, message: str) -> None:
        if self._progress:
            self._progress.close()
        if self.manager:
            self.manager.release()
            self.manager = None
        QMessageBox.critical(
            self, "Connection failed",
            f"Could not start the session:\n\n{message}\n\n"
            "Check that the board is powered and in range, or try Demo mode.",
        )

    def _on_exit_session(self) -> None:
        if self.session_view is not None:
            self.session_view.shutdown()
            self.stack.setCurrentWidget(self.dashboard)
            self.stack.removeWidget(self.session_view)
            self.session_view.deleteLater()
            self.session_view = None
        if self.manager is not None:
            self.manager.release()
            self.manager = None

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.session_view is not None:
            self.session_view.shutdown()
        if self.manager is not None:
            self.manager.release()
        for window in self._processing_windows:
            window.close()
        super().closeEvent(event)
