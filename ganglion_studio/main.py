"""Application entry point for Ganglion EEG Studio."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QApplication

from ganglion_studio.ui.main_window import MainWindow
from ganglion_studio.ui.theme import apply_dark_theme


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    app.setApplicationName("Ganglion EEG Studio")
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
