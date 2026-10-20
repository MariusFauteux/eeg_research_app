"""Pop-out cue window for the Protocol Helper.

A separate top-level window that shows the current instruction in large text with
a countdown, colour-coded by protocol state. Being its own window means it can be
dragged onto a second screen / in front of the subject while the researcher keeps
the main recording window (plots + controls) on the primary display.

The window is driven entirely by the panel: :meth:`show_step` on each transition
and :meth:`show_progress` every timer tick. It holds no timing logic of its own.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio import palette
from ganglion_studio.core.protocols import ProtocolStep

# state -> (background, text colour). Bright states use dark text like the marker
# buttons (marker_panel uses "#15171c"); dim states keep the light foreground.
_STATE_STYLE = {
    "lead_in": (palette.BG_ALT, palette.FG),
    "blink": (palette.OK, "#15171c"),
    "eyes_open": (palette.GOOD, "#15171c"),
    "eyes_closed": (palette.ACCENT, palette.WHITE),
    "rest": (palette.BG_ALT, palette.MUTED),
    "done": (palette.BG_ALT, palette.FG),
    # Added for the extended protocols:
    "look": (palette.VIOLET, "#15171c"),      # guided eye movements
    "center": (palette.BG_ALT, palette.MUTED),  # return-to-centre baseline
    "action": (palette.OK, "#15171c"),         # cued movement / artifact (attention)
    "task": (palette.ACCENT, palette.WHITE),   # cognitive active-task block
    "fixation": (palette.BG_ALT, palette.FG),  # oddball stimulus (neutral "+")
}


# On a fixation step the countdown stays blank until this many seconds are left, so
# nothing moves next to the cross except the final warning that the block is ending.
CROSS_COUNTDOWN_S = 5.0


def _fmt(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds >= 60:
        return f"{seconds // 60}:{seconds % 60:02d}"
    return f"{seconds}s"


class ProtocolCueWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Protocol cue")
        self.resize(640, 480)
        self._cross = False   # is the current step a fixation-cross step?

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._frame = QFrame()
        self._frame.setObjectName("cueRoot")
        root.addWidget(self._frame)

        inner = QVBoxLayout(self._frame)
        inner.setContentsMargins(24, 24, 24, 16)

        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction_label = QLabel("Ready")
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction_label.setWordWrap(True)
        # Big steady "+" for steps that ask the subject to fixate (Eyes Open blocks).
        self.cross_label = QLabel("+")
        self.cross_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cross_label.setVisible(False)
        self.countdown_label = QLabel("")
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        inner.addStretch(1)
        inner.addWidget(self.progress_label)
        inner.addWidget(self.instruction_label)
        inner.addWidget(self.cross_label)
        inner.addWidget(self.countdown_label)
        inner.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._full_btn = QPushButton("⤢ Fullscreen")
        self._full_btn.clicked.connect(self._toggle_fullscreen)
        btn_row.addWidget(self._full_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        inner.addLayout(btn_row)

        self._apply_style(palette.BG_ALT, palette.FG)

    # ------------------------------------------------------------------ view
    def _apply_style(self, bg: str, fg: str, cross: bool = False) -> None:
        """Colour the window; ``cross`` shrinks the text so the "+" is what the eye lands on."""
        self._frame.setStyleSheet(f"#cueRoot {{ background: {bg}; }}")
        self.progress_label.setStyleSheet(f"color:{fg}; font-size:20px; font-weight:600;")
        if cross:
            self.instruction_label.setStyleSheet(f"color:{fg}; font-size:32px; font-weight:700;")
            self.cross_label.setStyleSheet(f"color:{fg}; font-size:180px; font-weight:300;")
            self.countdown_label.setStyleSheet(f"color:{fg}; font-size:28px;")
        else:
            self.instruction_label.setStyleSheet(f"color:{fg}; font-size:64px; font-weight:800;")
            self.countdown_label.setStyleSheet(f"color:{fg}; font-size:40px;")

    def show_step(self, step: ProtocolStep) -> None:
        """Switch to a new step: instruction text, colour, and progress caption."""
        bg, fg = _STATE_STYLE.get(step.state, (palette.BG, palette.FG))
        cross = bool(getattr(step, "fixation_cross", False))
        self._apply_style(bg, fg, cross)
        self._cross = cross
        self.cross_label.setVisible(cross)
        # On a fixation step the countdown starts blank (a number ticking next to the
        # cross all block long would pull the eyes off it) and show_progress fills it in
        # for the last CROSS_COUNTDOWN_S seconds. The label stays *visible* though, so it
        # keeps reserving its line and the cross never shifts when the digits appear.
        self.countdown_label.setText("")
        self.instruction_label.setText(step.instruction)
        self.progress_label.setText(step.progress)
        if not self.isVisible():
            self.show()

    def show_progress(self, remaining: float, progress_text: str = "") -> None:
        """Update the live countdown (seconds until the next step)."""
        if self._cross and remaining > CROSS_COUNTDOWN_S:
            self.countdown_label.setText("")   # blank until the block is about to end
        else:
            self.countdown_label.setText(_fmt(remaining))
        if progress_text:
            self.progress_label.setText(progress_text)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self._full_btn.setText("⤢ Fullscreen")
        else:
            self.showFullScreen()
            self._full_btn.setText("⤣ Exit fullscreen")
