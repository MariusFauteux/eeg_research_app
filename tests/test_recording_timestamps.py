"""SessionRecorder rebuilds the saved timestamp column as a clean clock.

The native-BLE driver stamps both samples of a packet with the host's arrival
time, so the raw timestamp row arrives duplicated and bursty. On save we replace
it with an evenly spaced ``t0 + i / sampling_rate`` acquisition clock.
"""

import numpy as np
import pytest

from ganglion_studio.core.session import SessionConfig, SessionRecorder


def _recorder():
    return SessionRecorder(SessionConfig(name="t"))


@pytest.mark.parametrize("sr", [200, 200.5])
def test_acquisition_clock_is_evenly_spaced_and_monotonic(sr):
    ts_row = 13
    n = 8
    raw = np.zeros((15, n), dtype=np.float64)
    raw[1] = np.arange(n)  # an EEG row, must be left untouched
    # Duplicated + bursty arrival times, as the native driver produces them.
    raw[ts_row] = np.array(
        [100.000, 100.000, 100.0015, 100.0015, 100.0016, 100.0016, 100.060, 100.060]
    )

    out = SessionRecorder._with_acquisition_clock(
        _recorder(), raw, {"sampling_rate": sr, "timestamp_channel": ts_row}
    )

    clock = out[ts_row]
    # Anchored on the first real timestamp, then exact 1/sr steps.
    assert clock[0] == 100.000
    assert np.allclose(np.diff(clock), 1.0 / sr)
    assert np.all(np.diff(clock) > 0)  # strictly increasing, no duplicates
    # EEG data must pass through unchanged, and the input is not mutated.
    assert np.array_equal(out[1], np.arange(n))
    assert raw[ts_row][1] == 100.000


def test_no_timestamp_channel_is_a_noop():
    raw = np.ones((4, 5), dtype=np.float64)
    out = SessionRecorder._with_acquisition_clock(
        _recorder(), raw, {"sampling_rate": 200}  # no timestamp_channel
    )
    assert np.array_equal(out, raw)


def test_falls_back_to_started_at_when_no_valid_timestamp():
    from datetime import datetime

    rec = _recorder()
    rec.started_at = datetime(2026, 1, 1, 0, 0, 0)
    raw = np.zeros((15, 4), dtype=np.float64)  # timestamp row is all zeros
    out = SessionRecorder._with_acquisition_clock(
        rec, raw, {"sampling_rate": 200, "timestamp_channel": 13}
    )
    assert out[13][0] == rec.started_at.timestamp()
    # Anchored on a ~1.78e9 epoch, float64 resolves the 5 ms step to ~0.2 us, so
    # allow microsecond slack -- still evenly spaced and strictly increasing.
    assert np.allclose(np.diff(out[13]), 1.0 / 200, atol=1e-6)
    assert np.all(np.diff(out[13]) > 0)
