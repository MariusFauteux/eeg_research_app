"""Regression for the impedance readout: take the latest *non-zero* resistance
sample (Ohm -> kOhm), not a zero-diluted mean; -1.0 when never measured."""

import numpy as np

from ganglion_studio.core.board_manager import BoardManager


def _make() -> BoardManager:
    m = BoardManager(demo=False)
    m.num_rows = 13
    m.sampling_rate = 200
    m.eeg_channels = [1, 2, 3, 4]
    m.resistance_channels = [8, 9, 10, 11, 12]  # rows 8..11 map to channels 1..4
    m._buffer_len = 400
    m._ring = np.zeros((13, 400), dtype=np.float64)
    m._filled = 400
    m._write = 0  # full buffer -> recent() returns the whole ring in order
    return m


def test_latest_nonzero_and_na():
    m = _make()
    m._ring[8, 50] = 9000.0
    m._ring[8, 380] = 15000.0   # most recent non-zero on channel 0's row
    m._ring[10, 100] = 22000.0  # channel 2's row
    vals = m.latest_impedance_kohm()
    assert vals[0] == 15.0      # 15000 Ohm -> 15 kOhm (latest, not mean)
    assert vals[1] == -1.0      # row 9 all zeros -> not measured
    assert vals[2] == 22.0
    assert vals[3] == -1.0
