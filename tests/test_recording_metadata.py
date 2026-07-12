"""channel_active round-trips through save -> load.

A channel switched off records as a flat-zero column; saving its on/off state
lets a reloaded recording tell "off on purpose" apart from "on but dead".
"""

import numpy as np

from ganglion_studio.core.session import MarkerEvent, SessionConfig, SessionRecorder
from ganglion_studio.core.recording_loader import load_recording


def _raw():
    # 15-row Ganglion layout; ch4 (row 4) left at zero, as a disabled channel.
    raw = np.zeros((15, 6), dtype=np.float64)
    raw[1] = [-101.0, -62.0, -71.0, -97.0, -101.0, -76.0]
    raw[2] = [11.0, 15.0, 13.0, 14.0, 12.0, 14.0]
    raw[3] = [15.0, 15.0, 17.0, 17.0, 18.0, 14.0]
    raw[13] = 1782750697.0 + np.arange(6) / 200.0
    return raw


def _meta(active):
    return {
        "sampling_rate": 200,
        "eeg_channels": [1, 2, 3, 4],
        "channel_names": ["Ch1", "Ch2", "Ch3", "Ch4"],
        "channel_active": active,
        "marker_channel": 14,
        "timestamp_channel": 13,
    }


def test_channel_active_round_trips(tmp_path):
    rec = SessionRecorder(SessionConfig(name="t"), base_dir=str(tmp_path))
    rec.begin()
    active = [True, True, True, False]  # ch4 was switched off
    paths = rec.save(_raw(), _meta(active))

    csv = [p for p in paths if p.endswith("_raw.csv")][0]
    loaded = load_recording(csv)
    assert loaded.channel_active == active


def test_marker_labels_come_from_sidecar(tmp_path):
    rec = SessionRecorder(SessionConfig(name="t"), base_dir=str(tmp_path))
    rec.begin()
    rec.add_marker(MarkerEvent(timestamp=1782750697.0, code=3, label="Stimulus"))
    raw = _raw()
    raw[14, 2] = 3  # marker channel (row 14): code 3 at sample 2
    paths = rec.save(raw, _meta([True] * 4))

    csv = [p for p in paths if p.endswith("_raw.csv")][0]
    loaded = load_recording(csv)
    assert len(loaded.markers) == 1
    m = loaded.markers[0]
    assert (m.sample, m.code, m.label) == (2, 3, "Stimulus")  # not "code 3"


def test_marker_label_falls_back_to_code_when_not_in_sidecar(tmp_path):
    rec = SessionRecorder(SessionConfig(name="t"), base_dir=str(tmp_path))
    rec.begin()
    raw = _raw()
    raw[14, 4] = 7  # a channel marker with no matching sidecar row (no add_marker)
    paths = rec.save(raw, _meta([True] * 4))

    csv = [p for p in paths if p.endswith("_raw.csv")][0]
    loaded = load_recording(csv)
    assert len(loaded.markers) == 1
    assert loaded.markers[0].label == "code 7"


def test_channel_active_absent_loads_empty(tmp_path):
    rec = SessionRecorder(SessionConfig(name="t"), base_dir=str(tmp_path))
    rec.begin()
    meta = _meta([True] * 4)
    del meta["channel_active"]  # an older recording with no such field
    paths = rec.save(_raw(), meta)

    csv = [p for p in paths if p.endswith("_raw.csv")][0]
    loaded = load_recording(csv)
    assert loaded.channel_active == []
