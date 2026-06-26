"""BLE packet-loss tracking: BoardManager threads a per-sample loss flag from the
native driver into a ring aligned with the data, exposes signal-quality readers,
and records drop sample-indices; SessionRecorder writes them to a sidecar CSV."""

from __future__ import annotations

import csv

import numpy as np

from ganglion_studio.core import processing as proc
from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.dsp import interpolate_gaps
from ganglion_studio.core.recording_loader import load_recording
from ganglion_studio.core.session import SessionConfig, SessionRecorder


class _FakeBoard:
    """Minimal board: returns one chunk (matrix + aligned last_loss), then empty."""

    def __init__(self, matrix: np.ndarray, loss: np.ndarray) -> None:
        self._matrix = matrix
        self.last_loss = loss
        self._sent = False

    def get_board_data(self) -> np.ndarray:
        if self._sent:
            self.last_loss = np.zeros(0)
            return np.zeros((self._matrix.shape[0], 0))
        self._sent = True
        return self._matrix


def _manager() -> BoardManager:
    m = BoardManager(demo=False)
    m.num_rows = 6
    m.sampling_rate = 200
    m.eeg_channels = [1, 2, 3, 4]
    m._buffer_len = 400
    m._ring = np.zeros((6, 400), dtype=np.float64)
    m._loss_ring = np.zeros(400, dtype=np.float64)
    m._filled = 0
    m._write = 0
    m._dropped_total = 0
    m.streaming = True
    return m


def test_poll_tracks_loss_and_records_indices():
    m = _manager()
    matrix = np.ones((6, 10), dtype=np.float64)
    loss = np.zeros(10)
    loss[3] = 1.0
    loss[7] = 1.0  # two dropped packets in this chunk
    m.board = _FakeBoard(matrix, loss)

    m.start_recording()
    n = m.poll()

    assert n == 10
    assert m.dropped_packets() == 2
    # loss ring aligns column-for-column with the data ring
    assert np.array_equal(np.flatnonzero(m.recent_loss(1.0)), [3, 7])
    assert m.loss_rate(10.0) > 0.0
    # recorded drop indices are absolute within the recording (starts at 0 here)
    assert m.recorded_loss_indices() == [3, 7]


def test_loss_absent_for_board_without_flag():
    """Dongle/synthetic boards have no last_loss -> zero loss, no crash."""
    m = _manager()

    class _Plain:
        def get_board_data(self):
            if getattr(self, "_done", False):
                return np.zeros((6, 0))
            self._done = True
            return np.ones((6, 5))

    m.board = _Plain()
    assert m.poll() == 5
    assert m.dropped_packets() == 0
    assert m.recent_loss(1.0).sum() == 0.0


def test_recorder_writes_packet_loss_sidecar(tmp_path):
    rec = SessionRecorder(SessionConfig(name="t"), base_dir=str(tmp_path))
    rec.begin()
    raw = np.ones((6, 100), dtype=np.float64)
    meta = {"sampling_rate": 200, "eeg_channels": [1, 2, 3, 4]}

    written = rec.save(raw, meta, loss_samples=[10, 50, 90])

    loss_files = [p for p in written if p.endswith("_packet_loss.csv")]
    assert loss_files, "packet-loss sidecar was not written"
    with open(loss_files[0], newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["sample_index", "time_s"]
    assert rows[1] == ["10", "0.0500"]   # 10 / 200 Hz = 0.05 s
    assert len(rows) == 1 + 3            # header + three drops


# --------------------------------------------------------------------------- #
# gap interpolation (repair)
# --------------------------------------------------------------------------- #
def test_interpolate_gaps_linear_fill_and_noop():
    sig = np.array([0.0, 1.0, 99.0, 99.0, 4.0, 5.0])      # 99s are held-flat
    bad = np.array([False, False, True, True, False, False])
    out = interpolate_gaps(sig, bad)
    assert np.allclose(out, [0, 1, 2, 3, 4, 5])           # linear between 1 and 4
    assert sig[2] == 99.0                                  # input untouched (copy)
    # no bad samples -> unchanged
    assert np.allclose(interpolate_gaps(sig, np.zeros(6, bool)), sig)


def test_interpolate_gaps_edge_clamps_to_nearest_good():
    sig = np.array([9.0, 9.0, 2.0, 3.0])                  # bad at the very start
    out = interpolate_gaps(sig, np.array([True, True, False, False]))
    assert out[0] == 2.0 and out[1] == 2.0                # clamped to nearest good


def test_pipeline_repair_gaps_puts_samples_back_on_the_ramp():
    n = 200
    ramp = np.arange(n, dtype=float)                      # value == index
    eeg = np.stack([ramp.copy(), ramp.copy()])
    eeg[:, 50:52] = eeg[:, 49:50]                         # held-flat seam at 50,51
    cfg = proc.ProcessingConfig(repair_gaps=True, loss_samples=[50])
    cfg.filters.enabled = False                          # isolate the repair step
    out, msgs = proc.apply_pipeline(eeg, 200, ["a", "b"], cfg)
    assert np.allclose(out[0, 50:52], [50.0, 51.0])      # interpolated back
    assert any("Repaired" in m for m in msgs)


def test_loader_reads_packet_loss_sidecar(tmp_path):
    rec = SessionRecorder(SessionConfig(name="t"), base_dir=str(tmp_path))
    rec.begin()
    raw = np.zeros((13, 100), dtype=np.float64)
    meta = {"sampling_rate": 200, "eeg_channels": [1, 2, 3, 4],
            "channel_names": ["a", "b", "c", "d"], "marker_channel": 12}
    written = rec.save(raw, meta, loss_samples=[10, 42, 88])
    csv_path = next(p for p in written if p.endswith("_raw.csv"))

    loaded = load_recording(csv_path)
    assert loaded.loss_samples == [10, 42, 88]
    assert loaded.n_channels == 4
