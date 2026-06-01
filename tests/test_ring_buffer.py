"""Regression for the circular ring buffer in BoardManager.

Exercises partial fill, wrap-around, oversized chunks, and that `recent()` hands
back an independent copy (callers mutate it).
"""

import numpy as np

from ganglion_studio.core.board_manager import BoardManager


def _make(rows: int = 2, length: int = 5) -> BoardManager:
    mgr = BoardManager(demo=True)
    mgr.num_rows = rows
    mgr.sampling_rate = 1  # 1 sample per "second" -> recent(k) ~= k samples
    mgr._buffer_len = length
    mgr._ring = np.zeros((rows, length), dtype=np.float64)
    mgr._filled = 0
    mgr._write = 0
    return mgr


def _chunk(rows: int, vals) -> np.ndarray:
    arr = np.zeros((rows, len(vals)), dtype=np.float64)
    for r in range(rows):
        arr[r] = np.asarray(vals, dtype=np.float64) + r * 1000
    return arr


def test_partial_fill_in_order():
    mgr = _make(rows=2, length=5)
    mgr._append_ring(_chunk(2, [1, 2, 3]))
    out = mgr.recent(10)
    assert out.shape == (2, 3)
    assert list(out[0]) == [1, 2, 3]
    assert list(out[1]) == [1001, 1002, 1003]
    assert mgr._filled == 3


def test_wrap_keeps_time_order():
    mgr = _make(rows=1, length=5)
    mgr._append_ring(_chunk(1, [1, 2, 3]))
    mgr._append_ring(_chunk(1, [4, 5, 6, 7]))  # 7 total -> keep last 5
    out = mgr.recent(10)
    assert out.shape == (1, 5)
    assert list(out[0]) == [3, 4, 5, 6, 7]


def test_recent_subset_after_wrap():
    mgr = _make(rows=1, length=5)
    mgr._append_ring(_chunk(1, [1, 2, 3, 4, 5, 6, 7]))  # keep last 5
    assert list(mgr.recent(2)[0]) == [6, 7]
    assert list(mgr.recent(10)[0]) == [3, 4, 5, 6, 7]


def test_oversized_chunk():
    mgr = _make(rows=1, length=4)
    mgr._append_ring(_chunk(1, [10, 11, 12, 13, 14, 15]))  # keep last 4
    assert list(mgr.recent(10)[0]) == [12, 13, 14, 15]
    assert mgr._filled == 4


def test_recent_returns_independent_copy():
    mgr = _make(rows=1, length=5)
    mgr._append_ring(_chunk(1, [1, 2, 3]))
    out = mgr.recent(10)
    out[0, 0] = 999.0
    assert mgr.recent(10)[0, 0] == 1  # ring untouched
