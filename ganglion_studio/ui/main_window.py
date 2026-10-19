"""Main window: hosts the dashboard and the session view in a stack."""

from __future__ import annotations

import os
import logging
from typing import Optional

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QStackedWidget,
)

from ganglion_studio.core import rebuild_fif
from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.exporter import ExportError
from ganglion_studio.core.recording_loader import LoadError
from ganglion_studio.core.session import SessionConfig
from ganglion_studio.ui.dashboard import Dashboard
from ganglion_studio.ui.fif_editor_window import RecordingEditorWindow
from ganglion_studio.ui.processing_window import ProcessingWindow
from ganglion_studio.ui.session_view import SessionView
from ganglion_studio.ui.impedance_session_view import ImpedanceSessionView, ImpedanceRecordingWindow

logger = logging.getLogger(__name__)


class PrepareWorker(QThread):
    """Prepare + start the board off the UI thread (BLE connect can block)."""

    ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, manager: BoardManager) -> None:
        super().__init__()
        self._manager = manager
        self.prepared = False
        self.error: Optional[str] = None

    def run(self) -> None:
        try:
            self._manager.prepare()
            if self.isInterruptionRequested():
                return
            self._manager.start()
            self.prepared = True
            self.ok.emit()
        except Exception as exc:  # pragma: no cover - hardware path
            self.error = str(exc)
            self.failed.emit(str(exc))


class ReleaseWorker(QThread):
    """Wait for acquisition and release the backend without blocking Qt."""

    def __init__(self, manager: BoardManager) -> None:
        super().__init__()
        self.manager = manager
        self.error: Optional[str] = None

    def run(self) -> None:
        try:
            self.manager.release()
        except Exception as exc:  # pragma: no cover - hardware path
            self.error = str(exc)
            logger.exception("Board release failed")


class AcquisitionStopWorker(QThread):
    """Join the producer and take its final recording snapshot off the GUI."""

    def __init__(self, manager: BoardManager) -> None:
        super().__init__()
        self.manager = manager
        self.raw = None
        self.error: Optional[str] = None

    def run(self) -> None:
        try:
            if not self.manager.stop_acquisition(timeout=None):
                raise RuntimeError("Acquisition could not be stopped")
            if self.manager.recording:
                self.raw = self.manager.stop_recording()
        except Exception as exc:  # pragma: no cover - hardware path
            self.error = str(exc)
            logger.exception("Acquisition shutdown failed")


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
        self.dashboard.open_editor.connect(self._open_editor)
        self.dashboard.rebuild_fif.connect(self._rebuild_fif)
        self.dashboard.open_impedance_recording.connect(self._open_impedance_recording)
        self.stack.addWidget(self.dashboard)

        self.session_view: Optional[SessionView | ImpedanceSessionView] = None
        self.manager: Optional[BoardManager] = None
        self._worker: Optional[PrepareWorker] = None
        self._release_worker: Optional[ReleaseWorker] = None
        self._session_stop_worker: Optional[AcquisitionStopWorker] = None
        self._session_shutdown_ready = False
        self._shutdown_raw = None
        self._shutdown_action: Optional[str] = None
        self._waiting_transition = False
        self._closing = False
        self._close_retry = QTimer(self)
        self._close_retry.setSingleShot(True)
        self._close_retry.timeout.connect(self.close)
        self._progress: Optional[QProgressDialog] = None
        self._pending_config: Optional[SessionConfig] = None
        self._processing_windows: list[ProcessingWindow] = []
        self._editor_windows: list[RecordingEditorWindow] = []
        self._impedance_windows: list[ImpedanceRecordingWindow] = []

    def _open_impedance_recording(self) -> None:
        if self._closing:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open impedance recording", self._recordings_dir(),
            "Impedance recordings (*_impedance*.csv);;CSV files (*.csv)",
        )
        if not path:
            return
        try:
            window = ImpedanceRecordingWindow(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not open impedance recording", str(exc))
            return
        window.show()
        self._impedance_windows.append(window)

    def _open_processing_lab(self) -> None:
        if self._closing:
            return
        window = ProcessingWindow()
        window.shutdown_finished.connect(self._retry_close)
        window.show()
        window.raise_()
        self._processing_windows.append(window)

    def _open_editor(self) -> None:
        if self._closing:
            return
        window = RecordingEditorWindow()
        window.show()
        window.raise_()
        self._editor_windows.append(window)

    # ------------------------------------------------------------ rebuild
    def _recordings_dir(self) -> str:
        cand = os.path.join(os.getcwd(), "recordings")
        return cand if os.path.isdir(cand) else os.getcwd()

    def _rebuild_fif(self) -> None:
        """Regenerate a .fif from a session's original _raw.csv (always raw)."""
        csv_path, _ = QFileDialog.getOpenFileName(
            self, "Pick the session's _raw.csv", self._recordings_dir(),
            "Session CSV (*_raw.csv);;CSV files (*.csv)",
        )
        if not csv_path:
            return

        meta_path = rebuild_fif.meta_path_for_csv(csv_path)
        if not os.path.exists(meta_path):
            answer = QMessageBox.question(
                self, "No _meta.json found",
                f"No metadata file next to this CSV:\n{os.path.basename(meta_path)}\n\n"
                "The rebuild will assume a standard Ganglion layout "
                "(4 EEG channels, 200 Hz). Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        # A save dialog rather than writing straight to the default path, so
        # overwriting an existing .fif is always the user's explicit choice.
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save rebuilt .fif",
            rebuild_fif.default_output_path(csv_path), "FIF (*.fif)",
        )
        if not out_path:
            return

        try:
            result = rebuild_fif.rebuild(csv_path, out_path)
        except (LoadError, ExportError) as exc:
            QMessageBox.warning(self, "Rebuild failed", str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.critical(self, "Rebuild failed", f"Unexpected error: {exc}")
            return

        sidecar = (f"\nChannel info: {os.path.basename(result.sidecar_path)}"
                   if result.sidecar_path else "")
        QMessageBox.information(
            self, "Rebuilt",
            f"Saved:\n{result.fif_path}\n\n"
            f"{result.n_channels} ch | {result.sampling_rate} Hz | "
            f"{result.duration_s:0.1f} s | {result.n_markers} markers\n"
            f"Unfiltered raw data, straight from the CSV.{sidecar}",
        )

    def _on_start_session(self, config: SessionConfig) -> None:
        if self._closing or self._worker is not None or self._release_worker is not None \
                or self.manager is not None:
            return
        self._pending_config = config
        self.manager = BoardManager(
            demo=config.demo,
            mac_address=config.mac_address,
            serial_port=config.serial_port,
            firmware=config.firmware,
            use_custom_native=config.use_custom_native,
            decode_mode=config.decode_mode,
            acquisition_mode=config.acquisition_mode,
            impedance_units=config.impedance_units,
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
        self._worker.finished.connect(self._on_prepare_finished)
        self.dashboard.setEnabled(False)
        self._worker.start()

    def _on_prepare_finished(self) -> None:
        worker = self.sender()
        if worker is not self._worker:
            return
        self._worker = None
        if self._progress:
            self._progress.close()
            self._progress.deleteLater()
            self._progress = None
        worker.deleteLater()
        if self._closing or worker.isInterruptionRequested():
            self._begin_release()
            self._retry_close()
        elif worker.error is not None:
            self._on_prepare_failed(worker.error)
        elif worker.prepared:
            self._on_prepared()

    def _on_prepared(self) -> None:
        if self._closing:
            self._begin_release()
            return
        assert self.manager is not None and self._pending_config is not None
        view_class = (ImpedanceSessionView if self._pending_config.acquisition_mode == "impedance"
                      else SessionView)
        self.session_view = view_class(self.manager, self._pending_config)
        self.session_view.exit_session.connect(self._on_exit_session)
        self.session_view.open_processing.connect(self._open_processing_lab)
        self.stack.addWidget(self.session_view)
        self.stack.setCurrentWidget(self.session_view)

    def _on_prepare_failed(self, message: str) -> None:
        self._begin_release()
        if self._closing:
            return
        QMessageBox.critical(
            self, "Connection failed",
            f"Could not start the session:\n\n{message}\n\n"
            "Check that the board is powered and in range, or try Demo mode.",
        )

    def _begin_release(self) -> None:
        if self._worker is not None or self._release_worker is not None \
                or self._session_stop_worker is not None or self._waiting_transition:
            return
        if self.manager is None:
            return
        self._release_worker = ReleaseWorker(self.manager)
        self._release_worker.finished.connect(self._on_released)
        self._release_worker.start()

    def _on_released(self) -> None:
        worker = self.sender()
        if worker is not self._release_worker:
            return
        self._release_worker = None
        worker.deleteLater()
        if worker.error is not None:
            self._closing = False
            QMessageBox.warning(
                self, "Disconnect failed",
                f"Close this window again to retry disconnecting.\n\n{worker.error}",
            )
            return
        if self.manager is worker.manager:
            self.manager = None
        self.dashboard.setEnabled(not self._closing)
        self._retry_close()

    def _retry_close(self) -> None:
        if self._closing:
            self._close_retry.start(0)

    def _request_session_shutdown(self, action: str) -> bool:
        if self.session_view is None or self.manager is None or self._session_shutdown_ready:
            return True
        if self._session_stop_worker is not None:
            if action == "close":
                self._shutdown_action = action
            return False
        if self._waiting_transition:
            if action == "close":
                self._shutdown_action = action
            return False
        self.session_view.quiesce_for_shutdown()
        self._shutdown_action = action
        if getattr(self.session_view, "has_pending_transition", lambda: False)():
            self._waiting_transition = True
            self.session_view.shutdown_ready.connect(self._on_transition_shutdown_ready)
            return False
        self._session_stop_worker = AcquisitionStopWorker(self.manager)
        self._session_stop_worker.finished.connect(self._on_session_stopped)
        self._session_stop_worker.start()
        return False

    def _on_transition_shutdown_ready(self) -> None:
        if not self._waiting_transition:
            return
        self._waiting_transition = False
        self.session_view.shutdown_ready.disconnect(self._on_transition_shutdown_ready)
        self._request_session_shutdown(self._shutdown_action or "exit")

    def _on_session_stopped(self) -> None:
        worker = self.sender()
        if worker is not self._session_stop_worker:
            return
        self._session_stop_worker = None
        worker.deleteLater()
        action = self._shutdown_action
        self._shutdown_action = None
        if worker.error is not None:
            self.session_view.resume_after_failed_shutdown()
            QMessageBox.warning(self, "Could not finish recording", worker.error)
            return
        self._shutdown_raw = worker.raw
        self._session_shutdown_ready = True
        if action == "close":
            self._close_retry.start(0)
        else:
            QTimer.singleShot(0, self._on_exit_session)

    def _resume_session_after_failed_save(self) -> None:
        self._shutdown_raw = None  # the view retains the snapshot for save retry
        self._session_shutdown_ready = False
        self.session_view.resume_after_failed_shutdown()

    def _on_exit_session(self) -> None:
        if self._closing:
            return
        if self.session_view is not None:
            if not self._request_session_shutdown("exit"):
                return
            if not self.session_view.shutdown(raw_data=self._shutdown_raw):
                self._resume_session_after_failed_save()
                return
            self.stack.setCurrentWidget(self.dashboard)
            self.stack.removeWidget(self.session_view)
            self.session_view.deleteLater()
            self.session_view = None
            self._shutdown_raw = None
            self._session_shutdown_ready = False
        if self.manager is not None:
            self._begin_release()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.session_view is not None:
            if not self._request_session_shutdown("close"):
                event.ignore()
                return
            if not self.session_view.shutdown(raw_data=self._shutdown_raw):
                self._resume_session_after_failed_save()
                event.ignore()
                return
            self._shutdown_raw = None
        self._closing = True
        self.dashboard.setEnabled(False)
        if self._worker is not None:
            self._worker.requestInterruption()
        else:
            self._begin_release()
        children_closed = True
        for window in self._processing_windows:
            children_closed = window.close() and children_closed
        for window in self._editor_windows:
            window.close()
        for window in self._impedance_windows:
            window.close()
        if self._worker is not None or self._release_worker is not None or not children_closed:
            event.ignore()
            return
        super().closeEvent(event)
