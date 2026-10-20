"""Protocol Helper panel (right column of the recording window).

Pick a protocol (Blinks, Eyes Open/Closed, Resting state), tune its parameters,
and press Start. The panel then:

* walks the protocol timeline with a :class:`ProtocolRunner` (a small QTimer state
  machine),
* shows the pop-out :class:`ProtocolCueWindow` with the current instruction + countdown,
* plays a distinct tone on each transition (:class:`CueAudio`), and
* emits :pyattr:`marker` on every step that carries an annotation.

``SessionView`` connects :pyattr:`marker` to the existing marker path (so the
annotation lands on the recording) and :pyattr:`started` to auto-start recording.
The panel itself never touches the board or the recorder -- it just drives cues
and asks the session to drop markers.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ganglion_studio.core import protocol_config, protocols
from ganglion_studio.ui import theme
from ganglion_studio.ui.cue_audio import CueAudio
from ganglion_studio.ui.protocol_cue_window import ProtocolCueWindow


class ProtocolRunner(QObject):
    """Drives a protocol timeline off a QTimer, using a monotonic clock.

    Emits ``step_entered(ProtocolStep)`` as each step's time arrives,
    ``progress(remaining_s, progress_text)`` every tick, and ``finished()`` once
    the last step (the "done" step) has fired.
    """

    step_entered = pyqtSignal(object)   # ProtocolStep
    progress = pyqtSignal(float, str)   # seconds until next step, progress caption
    finished = pyqtSignal()

    def __init__(self, steps, total: float, parent=None) -> None:
        super().__init__(parent)
        self._steps = steps
        self._total = total
        self._idx = 0
        self._t0 = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(15)  # ~66 Hz: smooth countdown + tight stimulus onsets
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._idx = 0
        self._t0 = time.monotonic()
        self._timer.start()
        self._tick()  # fire the t=0 step right away

    def stop(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        elapsed = time.monotonic() - self._t0
        while self._idx < len(self._steps) and elapsed >= self._steps[self._idx].at:
            step = self._steps[self._idx]
            self._idx += 1
            self.step_entered.emit(step)
        if self._idx < len(self._steps):
            remaining = max(0.0, self._steps[self._idx].at - elapsed)
            caption = self._steps[self._idx - 1].progress if self._idx > 0 else ""
            self.progress.emit(remaining, caption)
        else:
            self._timer.stop()
            self.finished.emit()


class ProtocolPanel(QGroupBox):
    started = pyqtSignal()          # a protocol run began (session auto-starts recording)
    stopped = pyqtSignal()          # a protocol run ended (finished or stopped)
    marker = pyqtSignal(int, str)   # code, label -- an annotation to drop

    def __init__(self, audio_enabled: bool = True) -> None:
        super().__init__("Protocol helper")
        self._running = False
        self._runner: ProtocolRunner | None = None
        self._audio = CueAudio(enabled=audio_enabled)
        self._cue = ProtocolCueWindow()

        root = QVBoxLayout(self)
        hint = QLabel("Pick a protocol, set the timing, then Start. Cues show in a "
                      "pop-out window and annotations are added for you.")
        hint.setWordWrap(True)
        hint.setStyleSheet(theme.MUTED_QSS)
        root.addWidget(hint)

        self.proto_combo = QComboBox()
        for proto in protocols.PROTOCOLS:
            self.proto_combo.addItem(proto.label, proto.key)
        self.proto_combo.currentIndexChanged.connect(self._on_protocol_changed)
        root.addWidget(self.proto_combo)

        # One config page per protocol, added in the SAME order as protocols.PROTOCOLS
        # (the combo is populated from it and _on_protocol_changed maps index -> page).
        self.stack = QStackedWidget()
        for page in (self._build_blink_page(), self._build_eyes_page(),
                     self._build_rest_page(), self._build_oddball_page(),
                     self._build_eye_move_page(), self._build_cued_page(),
                     self._build_cognitive_page()):
            self.stack.addWidget(page)
        root.addWidget(self.stack)

        self.run_btn = QPushButton("▶ Start protocol")
        self.run_btn.clicked.connect(self._on_run_clicked)
        root.addWidget(self.run_btn)

        self.readout = QLabel("Idle.")
        self.readout.setWordWrap(True)
        self.readout.setStyleSheet(theme.HINT_QSS)
        root.addWidget(self.readout)
        root.addStretch(1)

        self._load_saved()

    # ------------------------------------------------------------ build pages
    @staticmethod
    def _dspin(lo: float, hi: float, step: float, suffix: str = " s") -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(1)
        w.setSuffix(suffix)
        return w

    @staticmethod
    def _ispin(lo: int, hi: int) -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        return w

    def _build_blink_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.blink_lead = self._dspin(0.0, 120.0, 1.0)
        self.blink_per_set = self._ispin(1, 100)
        self.blink_sets = self._ispin(1, 100)
        self.blink_inter_set = self._dspin(0.0, 600.0, 1.0)
        self.blink_inter_blink = self._dspin(0.2, 60.0, 0.5)
        form.addRow("Time before first blink", self.blink_lead)
        form.addRow("Blinks per set", self.blink_per_set)
        form.addRow("Sets per recording", self.blink_sets)
        form.addRow("Time between sets", self.blink_inter_set)
        form.addRow("Time between blinks", self.blink_inter_blink)
        return page

    def _build_eyes_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.eyes_conditions = self._ispin(1, 100)
        self.eyes_seconds = self._dspin(1.0, 3600.0, 5.0)
        self.eyes_start = QComboBox()
        self.eyes_start.addItem("Eyes Open", "EO")
        self.eyes_start.addItem("Eyes Closed", "EC")
        self.eyes_lead = self._dspin(0.0, 600.0, 1.0)
        self.eyes_cross = QCheckBox("Fixation cross during Eyes Open")
        self.eyes_cross.setToolTip("Shows a big steady “+” to stare at, which keeps the "
                                   "eyes still and cuts eye-movement artifacts.")
        form.addRow("Number of conditions", self.eyes_conditions)
        form.addRow("Time per condition", self.eyes_seconds)
        form.addRow("Start with", self.eyes_start)
        form.addRow("Delay before first condition", self.eyes_lead)
        form.addRow("", self.eyes_cross)
        return page

    def _build_rest_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.rest_duration = self._dspin(1.0, 7200.0, 10.0)
        self.rest_lead = self._dspin(0.0, 120.0, 1.0)
        form.addRow("Rest duration", self.rest_duration)
        form.addRow("Time before rest", self.rest_lead)
        return page

    def _build_oddball_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        note = QLabel("Count the rare high (target) beeps. Onset timing is software-"
                      "scheduled — good for the P300, not for precise early components.")
        note.setWordWrap(True)
        note.setStyleSheet(theme.HINT_QSS)
        form.addRow(note)
        self.odd_trials = self._ispin(2, 5000)
        self.odd_target = self._dspin(0.05, 0.9, 0.05, suffix="")
        self.odd_target.setDecimals(2)
        self.odd_isi = self._dspin(0.2, 5.0, 0.1)
        self.odd_jitter = self._dspin(0.0, 2.0, 0.05)
        self.odd_lead = self._dspin(0.0, 60.0, 1.0)
        self.odd_minstd = self._ispin(0, 20)
        self.odd_seed = self._ispin(0, 999999)
        form.addRow("Number of trials", self.odd_trials)
        form.addRow("Target probability", self.odd_target)
        form.addRow("Inter-stimulus interval", self.odd_isi)
        form.addRow("ISI jitter (±)", self.odd_jitter)
        form.addRow("Time before first tone", self.odd_lead)
        form.addRow("Min standards before target", self.odd_minstd)
        form.addRow("Random seed (0 = random)", self.odd_seed)
        return page

    def _build_eye_move_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        # Compact arrow glyphs (with tooltips) so all four fit the narrow side panel.
        self.eye_left = QCheckBox("◀")
        self.eye_right = QCheckBox("▶")
        self.eye_up = QCheckBox("▲")
        self.eye_down = QCheckBox("▼")
        dirs = QWidget()
        dbox = QHBoxLayout(dirs)
        dbox.setContentsMargins(0, 0, 0, 0)
        for cb, tip in ((self.eye_left, "Look left"), (self.eye_right, "Look right"),
                        (self.eye_up, "Look up"), (self.eye_down, "Look down")):
            cb.setToolTip(tip)
            dbox.addWidget(cb)
        self.eye_hold = self._dspin(0.5, 10.0, 0.5)
        self.eye_center = self._dspin(0.5, 10.0, 0.5)
        self.eye_repeats = self._ispin(1, 100)
        self.eye_lead = self._dspin(0.0, 60.0, 1.0)
        self.eye_random = QCheckBox("Shuffle direction order")
        form.addRow("Directions", dirs)
        form.addRow("Hold each look", self.eye_hold)
        form.addRow("Return to centre", self.eye_center)
        form.addRow("Repeats", self.eye_repeats)
        form.addRow("Time before first look", self.eye_lead)
        form.addRow("", self.eye_random)
        return page

    def _build_cued_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.cued_action = QComboBox()
        for action in protocols.CUED_ACTIONS:
            self.cued_action.addItem(action)
        self.cued_lead = self._dspin(0.0, 120.0, 1.0)
        self.cued_reps = self._ispin(1, 100)
        self.cued_sets = self._ispin(1, 100)
        self.cued_inter_set = self._dspin(0.0, 600.0, 1.0)
        self.cued_inter_rep = self._dspin(0.5, 60.0, 0.5)
        form.addRow("Action", self.cued_action)
        form.addRow("Time before first", self.cued_lead)
        form.addRow("Repetitions per set", self.cued_reps)
        form.addRow("Sets per recording", self.cued_sets)
        form.addRow("Time between sets", self.cued_inter_set)
        form.addRow("Time between reps", self.cued_inter_rep)
        return page

    def _build_cognitive_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.cog_blocks = self._ispin(1, 100)
        self.cog_seconds = self._dspin(1.0, 3600.0, 5.0)
        self.cog_start = QComboBox()
        self.cog_start.addItem("Rest", "REST")
        self.cog_start.addItem("Task", "TASK")
        self.cog_task = QLineEdit()
        form.addRow("Number of blocks", self.cog_blocks)
        form.addRow("Seconds per block", self.cog_seconds)
        form.addRow("Start with", self.cog_start)
        form.addRow("Task instruction", self.cog_task)
        return page

    # ------------------------------------------------------- params <-> widgets
    def _read_params(self, key: str):
        if key == "blinks":
            return protocols.BlinkParams(
                lead_in_s=self.blink_lead.value(),
                blinks_per_set=self.blink_per_set.value(),
                sets=self.blink_sets.value(),
                inter_set_s=self.blink_inter_set.value(),
                inter_blink_s=self.blink_inter_blink.value(),
            )
        if key == "eyes":
            return protocols.EyesParams(
                conditions=self.eyes_conditions.value(),
                seconds_per_condition=self.eyes_seconds.value(),
                start_with=self.eyes_start.currentData(),
                lead_in_s=self.eyes_lead.value(),
                fixation_cross=self.eyes_cross.isChecked(),
            )
        if key == "oddball":
            return protocols.OddballParams(
                n_trials=self.odd_trials.value(),
                target_prob=self.odd_target.value(),
                isi_s=self.odd_isi.value(),
                jitter_s=self.odd_jitter.value(),
                lead_in_s=self.odd_lead.value(),
                min_standards_before_target=self.odd_minstd.value(),
                seed=self.odd_seed.value(),
            )
        if key == "eyes_move":
            return protocols.EyeMovementParams(
                left=self.eye_left.isChecked(),
                right=self.eye_right.isChecked(),
                up=self.eye_up.isChecked(),
                down=self.eye_down.isChecked(),
                hold_s=self.eye_hold.value(),
                center_s=self.eye_center.value(),
                repeats=self.eye_repeats.value(),
                lead_in_s=self.eye_lead.value(),
                randomize=self.eye_random.isChecked(),
            )
        if key == "cued":
            return protocols.CuedMovementParams(
                action=self.cued_action.currentText(),
                lead_in_s=self.cued_lead.value(),
                reps_per_set=self.cued_reps.value(),
                sets=self.cued_sets.value(),
                inter_set_s=self.cued_inter_set.value(),
                inter_rep_s=self.cued_inter_rep.value(),
            )
        if key == "cognitive":
            return protocols.CognitiveParams(
                blocks=self.cog_blocks.value(),
                seconds_per_block=self.cog_seconds.value(),
                start_with=self.cog_start.currentData(),
                task_label=self.cog_task.text(),
            )
        return protocols.RestParams(
            duration_s=self.rest_duration.value(),
            lead_in_s=self.rest_lead.value(),
        )

    def _apply_params(self, key: str, d: dict) -> None:
        """Fill the widgets for ``key`` from a saved params dict (missing -> default)."""
        base = protocols.default_params(key)
        # A valid JSON file can still contain corrupt field values. Keep the
        # per-field defaults instead of allowing an optional preference to make
        # the recording view fail during construction.
        for field, value in d.items():
            if field not in base:
                continue
            default = base[field]
            if isinstance(default, bool):
                valid = isinstance(value, bool)
            elif isinstance(default, int):
                valid = (type(value) is int and -(2 ** 31) <= value < 2 ** 31)
            elif isinstance(default, float):
                valid = (type(value) in (int, float)
                         and abs(value) <= 1e308 and math.isfinite(value))
            else:
                valid = isinstance(value, type(default))
            if valid:
                base[field] = value
        if key == "blinks":
            self.blink_lead.setValue(float(base["lead_in_s"]))
            self.blink_per_set.setValue(int(base["blinks_per_set"]))
            self.blink_sets.setValue(int(base["sets"]))
            self.blink_inter_set.setValue(float(base["inter_set_s"]))
            self.blink_inter_blink.setValue(float(base["inter_blink_s"]))
        elif key == "eyes":
            self.eyes_conditions.setValue(int(base["conditions"]))
            self.eyes_seconds.setValue(float(base["seconds_per_condition"]))
            idx = self.eyes_start.findData(base["start_with"])
            self.eyes_start.setCurrentIndex(idx if idx >= 0 else 0)
            self.eyes_lead.setValue(float(base["lead_in_s"]))
            self.eyes_cross.setChecked(bool(base["fixation_cross"]))
        elif key == "oddball":
            self.odd_trials.setValue(int(base["n_trials"]))
            self.odd_target.setValue(float(base["target_prob"]))
            self.odd_isi.setValue(float(base["isi_s"]))
            self.odd_jitter.setValue(float(base["jitter_s"]))
            self.odd_lead.setValue(float(base["lead_in_s"]))
            self.odd_minstd.setValue(int(base["min_standards_before_target"]))
            self.odd_seed.setValue(int(base["seed"]))
        elif key == "eyes_move":
            self.eye_left.setChecked(bool(base["left"]))
            self.eye_right.setChecked(bool(base["right"]))
            self.eye_up.setChecked(bool(base["up"]))
            self.eye_down.setChecked(bool(base["down"]))
            self.eye_hold.setValue(float(base["hold_s"]))
            self.eye_center.setValue(float(base["center_s"]))
            self.eye_repeats.setValue(int(base["repeats"]))
            self.eye_lead.setValue(float(base["lead_in_s"]))
            self.eye_random.setChecked(bool(base["randomize"]))
        elif key == "cued":
            idx = self.cued_action.findText(str(base["action"]))
            self.cued_action.setCurrentIndex(idx if idx >= 0 else 0)
            self.cued_lead.setValue(float(base["lead_in_s"]))
            self.cued_reps.setValue(int(base["reps_per_set"]))
            self.cued_sets.setValue(int(base["sets"]))
            self.cued_inter_set.setValue(float(base["inter_set_s"]))
            self.cued_inter_rep.setValue(float(base["inter_rep_s"]))
        elif key == "cognitive":
            self.cog_blocks.setValue(int(base["blocks"]))
            self.cog_seconds.setValue(float(base["seconds_per_block"]))
            idx = self.cog_start.findData(base["start_with"])
            self.cog_start.setCurrentIndex(idx if idx >= 0 else 0)
            self.cog_task.setText(str(base["task_label"]))
        else:
            self.rest_duration.setValue(float(base["duration_s"]))
            self.rest_lead.setValue(float(base["lead_in_s"]))

    def _load_saved(self) -> None:
        saved = protocol_config.load()
        for proto in protocols.PROTOCOLS:
            self._apply_params(proto.key, saved.get(proto.key, {}))

    def _current_key(self) -> str:
        return self.proto_combo.currentData()

    # ------------------------------------------------------------------- logic
    def _on_protocol_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)

    def _on_run_clicked(self) -> None:
        if self._running:
            self._stop_run()
        else:
            self._start_run()

    def _start_run(self) -> None:
        key = self._current_key()
        params = self._read_params(key)
        protocol_config.save(key, asdict(params))
        steps, total = protocols.build_steps(key, params)

        self._running = True
        self.run_btn.setText("■ Stop protocol")
        self.proto_combo.setEnabled(False)
        self.stack.setEnabled(False)
        self.started.emit()  # session turns on recording so markers are captured
        # A session callback can stop this run when recording cannot start.
        if not self._running:
            return

        self._runner = ProtocolRunner(steps, total, parent=self)
        self._runner.step_entered.connect(self._on_step)
        self._runner.progress.connect(self._on_progress)
        self._runner.finished.connect(self._on_finished)
        self._cue.show()
        self._runner.start()

    def _stop_run(self) -> None:
        if self._runner is not None:
            self._runner.stop()
        self._end_run("Stopped.")

    def _on_finished(self) -> None:
        self._end_run("Protocol complete.")

    def _end_run(self, message: str, *, emit_stopped: bool = True) -> None:
        self._running = False
        self._runner = None
        self.run_btn.setText("▶ Start protocol")
        self.proto_combo.setEnabled(True)
        self.stack.setEnabled(True)
        self.readout.setText(message)
        if emit_stopped:
            self.stopped.emit()

    def _on_step(self, step) -> None:
        self._audio.play(step.tone)
        self._cue.show_step(step)
        if step.marker_code is not None:
            self.marker.emit(step.marker_code, step.marker_label)

    def _on_progress(self, remaining: float, caption: str) -> None:
        self._cue.show_progress(remaining, caption)
        tail = f" · next in {remaining:.0f}s" if remaining > 0 else ""
        self.readout.setText((caption or "Running") + tail)

    def shutdown(self) -> None:
        """Stop the timer and close the cue window (called on session teardown)."""
        if self._runner is not None:
            self._runner.stop()
        # Teardown must reset the controls too, in case saving fails and leaves
        # this session open. Do not emit stopped here: it may trigger a save.
        self._end_run("Stopped.", emit_stopped=False)
        self._cue.close()
