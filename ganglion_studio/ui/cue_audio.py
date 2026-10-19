"""Short audio cues for the Protocol Helper.

Plays a distinct little tone on each protocol transition (a higher "go" beep to
start a block / blink, a lower "stop" beep to end one, a short "tick" between
sets, a low "end" tone when the protocol finishes). There was no audio anywhere
in the app before, so this is deliberately self-contained and optional:

* Tones are *synthesized* at first use (numpy sine + short fades) and cached as
  tiny WAV files under ``~/.ganglion_studio/cues/`` -- no bundled assets.
* Playback uses ``QtMultimedia`` (ships with PyQt6). If that module or an audio
  backend is unavailable, the whole thing turns into a silent no-op: the visual
  cue and the recording are never affected by a missing speaker.
"""

from __future__ import annotations

import os
import wave

import numpy as np
from PyQt6.QtCore import QUrl

# QtMultimedia is optional at runtime -- import defensively so a headless box or a
# machine without an audio backend still runs the rest of the app.
try:
    from PyQt6.QtMultimedia import QSoundEffect
    _HAVE_AUDIO = True
except Exception:  # pragma: no cover - depends on the platform's Qt build
    _HAVE_AUDIO = False

_CACHE_DIR = os.path.expanduser(os.path.join("~", ".ganglion_studio", "cues"))
_SAMPLE_RATE = 44100

# kind -> (frequency Hz, duration s). Higher = "go"/attention, lower = "stop"/end.
# "standard"/"target" are the oddball STIMULUS tones (the subject counts the higher
# "target"); they are deliberately close in length so only pitch distinguishes them.
_TONES = {
    "go": (880.0, 0.15),
    "stop": (440.0, 0.20),
    "tick": (1200.0, 0.06),
    "end": (330.0, 0.45),
    "standard": (1000.0, 0.10),
    "target": (1500.0, 0.10),
}


def _write_tone(path: str, freq: float, dur_s: float, volume: float = 0.6) -> None:
    """Render a mono 16-bit sine tone with short fades to ``path`` (a WAV file)."""
    n = int(_SAMPLE_RATE * dur_s)
    t = np.arange(n) / _SAMPLE_RATE
    wave_f = np.sin(2 * np.pi * freq * t)
    # 8 ms linear fade in/out so the tone does not click on/off.
    fade = max(1, int(_SAMPLE_RATE * 0.008))
    env = np.ones(n)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    pcm = (wave_f * env * volume * 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


class CueAudio:
    """Plays the named protocol tones; a no-op when audio is disabled/unavailable."""

    def __init__(self, enabled: bool = True) -> None:
        # Stay silent on headless/offscreen runs (tests, smoke_test) where there
        # is no audio device -- the visual cue still works everywhere.
        headless = os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        self.enabled = bool(enabled) and _HAVE_AUDIO and not headless
        self._effects: dict = {}
        if not self.enabled:
            return
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            for kind, (freq, dur) in _TONES.items():
                path = os.path.join(_CACHE_DIR, f"{kind}.wav")
                if not os.path.exists(path):
                    _write_tone(path, freq, dur)
                effect = QSoundEffect()
                effect.setSource(QUrl.fromLocalFile(path))
                effect.setVolume(0.6)
                self._effects[kind] = effect
        except Exception:  # pragma: no cover - never let audio setup break the app
            self.enabled = False
            self._effects = {}

    def play(self, kind) -> None:
        """Play the tone for ``kind`` (silently ignores None / unknown / errors)."""
        if not self.enabled or not kind:
            return
        effect = self._effects.get(kind)
        if effect is None:
            return
        try:
            effect.play()
        except Exception:  # pragma: no cover - a playback error must not propagate
            pass
