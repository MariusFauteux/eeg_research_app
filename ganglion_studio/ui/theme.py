"""Dark theme for the application and pyqtgraph defaults."""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

BG = "#1b1d23"
BG_ALT = "#23262e"
FG = "#e6e6e6"
ACCENT = "#4f8ef7"


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(FG))
    palette.setColor(QPalette.ColorRole.Base, QColor(BG_ALT))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(BG))
    palette.setColor(QPalette.ColorRole.Text, QColor(FG))
    palette.setColor(QPalette.ColorRole.Button, QColor(BG_ALT))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(FG))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_ALT))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(FG))
    app.setPalette(palette)

    app.setStyleSheet(
        """
        QGroupBox {
            border: 1px solid #3a3f4b; border-radius: 6px;
            margin-top: 10px; padding-top: 8px; font-weight: 600;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QPushButton {
            background: #2c303a; border: 1px solid #3a3f4b; border-radius: 5px;
            padding: 5px 12px;
        }
        QPushButton:hover { background: #353a46; }
        QPushButton:pressed { background: #4f8ef7; }
        QPushButton:disabled { color: #777; }
        QTabBar::tab {
            background: #23262e; padding: 7px 14px; border: 1px solid #3a3f4b;
            border-bottom: none; border-top-left-radius: 5px; border-top-right-radius: 5px;
        }
        QTabBar::tab:selected { background: #2c303a; color: #4f8ef7; }
        QListWidget, QTableWidget { border: 1px solid #3a3f4b; border-radius: 5px; }
        """
    )

    pg.setConfigOptions(antialias=True, background=BG, foreground=FG)
