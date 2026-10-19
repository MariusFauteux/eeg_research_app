"""Marker events retain their times, collisions and recording boundaries."""

import queue
import threading

import numpy as np
import pytest

from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.native_ganglion import NUM_EEG, NativeGanglionClient


@pytest.fixture
def manager(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("ganglion_studio.core.board_manager.time.time", lambda: clock[0])
    client = NativeGanglionClient("dummy", 8, [1, 2, 3, 4], 6, 7)
    client._data_q = queue.Queue()
    m = BoardManager()
    m.board = client
    m.num_rows = 8
    m.eeg_channels = [1, 2, 3, 4]
    m.timestamp_channel = 6
    m.marker_channel = 7
    m._buffer_len = 400
    m._ring = np.zeros((8, 400))
    m._loss_ring = np.zeros(400)
    m.streaming = True
    return m, clock


def _queue_samples(manager, timestamps):
    block = np.zeros((NUM_EEG + 2, len(timestamps)))
    block[0] = np.arange(len(timestamps)) + 10
    block[NUM_EEG] = timestamps
    manager.board._data_q.put(block)


def test_recording_keeps_rapid_markers_and_same_sample_collisions(manager):
    m, clock = manager
    m.start_recording()
    m.insert_marker(1, timestamp=100.011)
    m.insert_marker(2, timestamp=100.012)
    m.insert_marker(3, timestamp=100.019)
    _queue_samples(m, [100.005, 100.010, 100.015, 100.020])
    clock[0] = 100.03

    raw = m.stop_recording()

    assert np.array_equal(raw[7], [0, 0, 1, 3])
    assert m.recorded_marker_events() == [
        (2, 1.0, 100.011), (2, 2.0, 100.012), (3, 3.0, 100.019)]


def test_record_start_excludes_queued_old_samples_and_markers(manager):
    m, clock = manager
    m.insert_marker(9, timestamp=99.995)
    _queue_samples(m, [99.990, 100.010, 100.020])
    m.start_recording()
    m.insert_marker(2, timestamp=100.015)
    clock[0] = 100.03

    raw = m.stop_recording()

    assert np.array_equal(raw[6], [100.010, 100.020])
    assert np.array_equal(raw[7], [0, 2])
    assert m.recorded_marker_events() == [(1, 2.0, 100.015)]


def test_marker_after_final_sample_is_preserved_and_never_leaks(manager):
    m, clock = manager
    m.start_recording()
    _queue_samples(m, [100.005, 100.010])
    m.poll()
    m.insert_marker(5, timestamp=100.015)
    clock[0] = 100.02

    raw = m.stop_recording()

    assert np.array_equal(raw[7], [0, 5])
    assert m.recorded_marker_events() == [(1, 5.0, 100.015)]
    assert not m.board._pending_markers

    clock[0] = 100.05
    m.start_recording()
    _queue_samples(m, [100.055, 100.060])
    clock[0] = 100.07
    second = m.stop_recording()
    assert not second[7].any()
    assert m.recorded_marker_events() == []


def test_empty_recording_keeps_event_without_inventing_a_sample(manager):
    m, clock = manager
    m.start_recording()
    m.insert_marker(2, timestamp=100.005)
    clock[0] = 100.01

    raw = m.stop_recording()

    assert raw.shape == (8, 0)
    assert m.recorded_marker_events() == [(None, 2.0, 100.005)]
    assert not m.board._pending_markers


def test_stop_drains_backlog_and_excludes_samples_after_stop_boundary(manager):
    m, clock = manager
    m.start_recording()
    _queue_samples(m, [100.005, 100.010, 100.025])
    m.insert_marker(3, timestamp=100.009)
    clock[0] = 100.02

    raw = m.stop_recording()

    assert np.array_equal(raw[6], [100.005, 100.010])
    assert np.array_equal(raw[7], [0, 3])
    assert m.recorded_sample_count() == 2


def test_paused_recording_can_keep_an_event_at_its_final_sample(manager):
    m, clock = manager
    m.start_recording()
    _queue_samples(m, [100.005, 100.010])
    m.poll()
    m.streaming = False
    m.insert_marker(7, timestamp=100.015)
    clock[0] = 100.02

    raw = m.stop_recording()

    assert np.array_equal(raw[7], [0, 7])
    assert m.recorded_marker_events() == [(1, 7.0, 100.015)]


def test_stop_waits_for_inflight_poll_before_freezing_recording(manager):
    m, clock = manager
    entered = threading.Event()
    finish = threading.Event()

    class BlockingBoard:
        sent = False

        def get_board_data(self):
            if self.sent:
                return np.zeros((8, 0))
            self.sent = True
            entered.set()
            assert finish.wait(2)
            raw = np.zeros((8, 2))
            raw[6] = [100.005, 100.010]
            return raw

    m.board = BlockingBoard()
    m.start_recording()
    producer = threading.Thread(target=m.poll, daemon=True)
    producer.start()
    assert entered.wait(2)
    clock[0] = 100.02
    result = []
    stop = threading.Thread(target=lambda: result.append(m.stop_recording()), daemon=True)
    stop.start()
    finish.set()
    producer.join(2)
    stop.join(2)

    assert not producer.is_alive()
    assert not stop.is_alive()
    assert np.array_equal(result[0][6], [100.005, 100.010])
