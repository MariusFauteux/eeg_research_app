"""Application entry point for Ganglion EEG Studio."""

from __future__ import annotations

import logging
import sys
import traceback

from PyQt6.QtWidgets import QApplication, QMessageBox

from ganglion_studio.ui.main_window import MainWindow
from ganglion_studio.ui.theme import apply_dark_theme

logger = logging.getLogger(__name__)


def _install_excepthook() -> None:
    """Show uncaught exceptions in a dialog instead of aborting the app.

    Without this, PyQt aborts the process when a Python exception escapes a slot
    or callback. We keep the app alive and report the error to the user.
    """

    def hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.error("Uncaught exception:\n%s", text)
        try:
            QMessageBox.critical(
                None, "Unexpected error",
                f"{exc_type.__name__}: {exc_value}\n\nThe app will keep running.",
            )
        except Exception:
            pass

    sys.excepthook = hook


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    app.setApplicationName("Ganglion EEG Studio")
    apply_dark_theme(app)
    _install_excepthook()
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
