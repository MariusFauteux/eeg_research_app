"""Corrupt optional protocol preferences must leave usable UI defaults."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ganglion_studio.core import protocol_config, protocols
from ganglion_studio.ui.panels.protocol_panel import ProtocolPanel


def test_corrupt_saved_fields_fall_back_to_defaults(tmp_path, monkeypatch):
    path = tmp_path / "protocols.json"
    monkeypatch.setattr(protocol_config, "PATH", str(path))
    path.write_text(json.dumps({
        "blinks": {"lead_in_s": None, "sets": 4, "blinks_per_set": 2 ** 100},
        "eyes": {"conditions": "broken", "fixation_cross": []},
        "rest": {"duration_s": float("nan")},
        "cognitive": {"task_label": {"unexpected": "object"}},
        "oddball": 4,
    }))
    app = QApplication.instance() or QApplication([])
    panel = ProtocolPanel(audio_enabled=False)
    try:
        blink = panel._read_params("blinks")
        assert blink.sets == 4
        assert blink.lead_in_s == protocols.BlinkParams().lead_in_s
        assert blink.blinks_per_set == protocols.BlinkParams().blinks_per_set
        assert panel._read_params("eyes") == protocols.EyesParams()
        assert panel._read_params("rest") == protocols.RestParams()
        assert panel._read_params("cognitive") == protocols.CognitiveParams()
        assert panel._read_params("oddball") == protocols.OddballParams()
    finally:
        panel.shutdown()
        panel.close()
        app.processEvents()


def test_protocol_starts_when_preferences_cannot_be_written(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol_config, "PATH", str(tmp_path / "protocols.json"))

    def denied(*args, **kwargs):
        raise PermissionError("read-only config directory")

    monkeypatch.setattr(protocol_config, "open", denied, raising=False)
    app = QApplication.instance() or QApplication([])
    panel = ProtocolPanel(audio_enabled=False)
    started = []
    panel.started.connect(lambda: started.append(True))
    try:
        panel._start_run()
        assert panel._running
        assert panel._runner is not None
        assert started == [True]
    finally:
        panel.shutdown()
        panel.close()
        app.processEvents()


def test_shutdown_resets_running_controls_without_emitting_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol_config, "PATH", str(tmp_path / "protocols.json"))
    app = QApplication.instance() or QApplication([])
    panel = ProtocolPanel(audio_enabled=False)
    stopped = []
    panel.stopped.connect(lambda: stopped.append(True))
    try:
        panel._start_run()
        runner = panel._runner
        assert panel._running
        assert runner._timer.isActive()

        panel.shutdown()

        assert not panel._running
        assert panel._runner is None
        assert not runner._timer.isActive()
        assert panel.proto_combo.isEnabled()
        assert panel.stack.isEnabled()
        assert panel.run_btn.text() == "▶ Start protocol"
        assert stopped == []
    finally:
        panel.shutdown()
        panel.close()
        app.processEvents()


def test_session_can_abort_protocol_start_without_creating_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol_config, "PATH", str(tmp_path / "protocols.json"))
    app = QApplication.instance() or QApplication([])
    panel = ProtocolPanel(audio_enabled=False)
    panel.started.connect(panel.shutdown)
    try:
        panel._start_run()

        assert not panel._running
        assert panel._runner is None
        assert not panel._cue.isVisible()
        assert panel.run_btn.text() == "▶ Start protocol"
    finally:
        panel.shutdown()
        panel.close()
        app.processEvents()
