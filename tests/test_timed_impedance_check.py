"""Temporary checks keep the EEG wall-time gap and genuine impedance events."""

import queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.impedance import ImpedanceReading
from ganglion_studio.core.impedance_check import CHECK_MARKERS
from ganglion_studio.core.native_ganglion import NativeGanglionClient, _ble_worker_main


class Board:
    def __init__(self, clock):
        self.clock = clock
        self.data = []
        self.readings = []
        self.commands = []
        self.fail_resume = False
        self.fail_begin = False

    def get_board_data(self):
        return self.data.pop(0) if self.data else np.zeros((8, 0))

    def get_impedance_data(self):
        readings, self.readings = self.readings, []
        return readings

    def switch_acquisition_mode(self, mode):
        self.commands.append(mode)
        if mode == "eeg" and self.fail_resume or mode == "impedance" and self.fail_begin:
            raise RuntimeError("simulated write failure")
        return self.clock[0]

    def config_board(self, command):
        self.commands.append(command)

    def insert_marker(self, value):
        self.commands.append(("marker", value))


@pytest.fixture
def setup(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("ganglion_studio.core.board_manager.time.time", lambda: clock[0])
    manager = BoardManager()
    manager.num_rows = 8
    manager.sampling_rate = 10
    manager.eeg_channels = [1, 2, 3, 4]
    manager.timestamp_channel = 6
    manager.marker_channel = 7
    manager._buffer_len = 400
    manager._ring = np.zeros((8, 400))
    manager._loss_ring = np.zeros(400)
    manager.board = Board(clock)
    manager.streaming = True
    return manager, clock


def samples(timestamps):
    data = np.ones((8, len(timestamps)))
    data[6] = timestamps
    data[7] = 0
    return data


def test_check_keeps_missing_eeg_time_and_separate_impedance_readings(setup):
    m, clock = setup
    m.start_recording()
    m.board.data = [samples([100.1, 100.2, 100.3, 100.4])]
    clock[0] = 100.4
    m.begin_impedance_check(2)
    assert m.recording and m.acquisition_mode == "eeg"
    assert m.impedance_check_status == "measuring"
    readings = [ImpedanceReading(100.9, 0, 12000), ImpedanceReading(101.2, 4, 33000)]
    m.board.readings = readings
    clock[0] = 101.4
    m.poll()
    m.insert_marker(7, timestamp=101.4)  # A protocol continues on wall time.
    assert ("marker", 7) not in m.board.commands
    clock[0] = 102.4
    m.end_impedance_check()
    m.board.data = [samples([102.5, 102.6])]
    clock[0] = 102.6
    m.poll()
    clock[0] = 102.7
    raw = m.stop_recording()

    assert raw.shape == (8, 26)
    assert np.isnan(raw[1:5, 4:24]).all()
    assert np.isfinite(raw[1:5, :4]).all() and np.isfinite(raw[1:5, 24:]).all()
    assert m.recorded_impedance_checks() == readings
    assert m.recorded_eeg_gaps() == [dict(start_sample=4, length=20,
        reason="impedance_check", started_at=100.4, ended_at=102.4)]
    assert [e["code"] for e in m.impedance_check_events()] == [90, 91, 92, 93]
    assert set(CHECK_MARKERS).issubset({code for _, code, _ in m.recorded_marker_events()})
    assert next(sample for sample, code, _ in m.recorded_marker_events() if code == 7) == 14
    assert m.board.commands == ["impedance", "eeg", "N"]


def test_stopping_recording_midcheck_finalizes_without_a_false_resume(setup):
    m, clock = setup
    m.start_recording()
    m.begin_impedance_check(10)
    clock[0] = 101
    m.board.readings = [ImpedanceReading(100.5, 1, 5000)]

    raw = m.stop_recording()

    assert raw.shape == (8, 10)
    assert np.isnan(raw[1:5]).all()
    assert not m.recording and not m.streaming
    assert m.impedance_check_status == "idle"
    assert [event["code"] for event in m.impedance_check_events()] == [90, 91, 92]
    assert m.recorded_impedance_checks() == [ImpedanceReading(100.5, 1, 5000)]
    assert m.board.commands == ["impedance", None]


def test_failed_resume_stays_missing_until_retry_and_never_claims_resumed(setup):
    m, clock = setup
    m.start_recording()
    m.begin_impedance_check(1)
    clock[0] = 101
    m.board.fail_resume = True
    with pytest.raises(RuntimeError, match="write failure"):
        m.end_impedance_check()
    assert m.impedance_check_status == "restore_failed"
    assert not m.streaming
    assert 93 not in [event["code"] for event in m.impedance_check_events()]
    clock[0] = 102
    m.board.fail_resume = False
    m.end_impedance_check()
    raw = m.stop_recording()
    assert np.isnan(raw[1:5, :20]).all()
    assert m.recorded_eeg_gaps()[0]["length"] == 20
    assert [event["code"] for event in m.impedance_check_events()] == [90, 91, 92, 93]


def test_failed_start_restores_eeg_without_fake_impedance_start_or_end(setup):
    m, clock = setup
    m.start_recording()
    m.board.fail_begin = True
    with pytest.raises(RuntimeError, match="write failure"):
        m.begin_impedance_check(1)
    assert m.impedance_check_status == "idle" and m.streaming
    assert [event["code"] for event in m.impedance_check_events()] == [90, 93]


def test_nested_checks_mode_changes_and_new_recording_are_guarded(setup):
    m, clock = setup
    m.begin_impedance_check(1)
    with pytest.raises(RuntimeError, match="already active"):
        m.begin_impedance_check(1)
    with pytest.raises(RuntimeError, match="Finish"):
        m.set_acquisition_mode("impedance")
    with pytest.raises(RuntimeError, match="Finish"):
        m.start_recording()
    m.end_impedance_check()


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan"), 3601])
def test_invalid_duration_sends_no_commands(setup, duration):
    m, _ = setup
    with pytest.raises(ValueError, match="duration"):
        m.begin_impedance_check(duration)
    assert m.board.commands == []


def test_native_mode_switch_waits_for_matching_ack_and_reports_failure():
    client = NativeGanglionClient("dummy", 8, [1, 2, 3, 4], 6, 7, timeout=0.1)
    client._ctrl_q, client._reply_q = queue.Queue(), queue.Queue()
    client._reply_q.put((1, None, 100.25))
    assert client.switch_acquisition_mode("impedance") == 100.25
    assert client._ctrl_q.get_nowait() == ("switch", ("impedance", 1))
    client._reply_q.put((2, "write failed", 101.0))
    with pytest.raises(RuntimeError, match="write failed"):
        client.switch_acquisition_mode("eeg")
    assert not client._streaming


def test_stop_markers_use_acknowledged_stop_boundaries_and_keep_eeg_tail(setup):
    m, clock = setup

    class DelayedBoard(Board):
        def switch_acquisition_mode(self, mode):
            if mode == "impedance":
                self.last_mode_switch_stopped_at = 100.5
                self.data.append(samples([100.1, 100.4]))
                clock[0] = 101.0
            else:
                self.last_mode_switch_stopped_at = 103.25
                clock[0] = 104.0
            return clock[0]

    m.board = DelayedBoard(clock)
    m.start_recording()
    m.begin_impedance_check(2)
    clock[0] = 103
    m.end_impedance_check()
    raw = m.stop_recording()

    assert raw.shape[1] == 2 + 35
    assert np.array_equal(raw[6, :2], [100.1, 100.4])
    assert np.isnan(raw[1:5, 2:]).all()
    assert [event["timestamp"] for event in m.impedance_check_events()] == [100.5, 101, 103.25, 104]
    assert m.recorded_eeg_gaps()[0]["started_at"] == 100.5


def test_native_check_switch_keeps_subscription_and_acknowledges_stop_and_start(monkeypatch):
    writes = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def connect(self):
            pass

        async def start_notify(self, *args):
            writes.append("subscribe")

        async def write_gatt_char(self, uuid, value, **kwargs):
            writes.append(value)

        async def stop_notify(self, *args):
            writes.append("unsubscribe")

        async def disconnect(self):
            pass

    monkeypatch.setitem(sys.modules, "bleak", SimpleNamespace(BleakClient=Client))
    commands, replies = queue.Queue(), queue.Queue()
    for command in [("start", "eeg"), ("switch", ("impedance", 1)),
                    ("switch", ("eeg", 2)), ("stop", None), ("release", None)]:
        commands.put(command)
    _ble_worker_main("dummy", 1, queue.Queue(), commands, queue.Queue(), reply_q=replies)
    assert writes == ["subscribe", b"b", b"s", b"z", b"Z", b"b", b"s", "unsubscribe"]
    for request in [1, 2]:
        request_id, error, completed, stopped = replies.get_nowait()
        assert request_id == request and error is None
        assert stopped <= completed


def test_brainflow_restore_retry_stops_uncertain_hardware_even_when_flag_false(setup):
    m, clock = setup

    class Backend:
        fail_stop = False
        stop_calls = 0

        def stop_stream(self):
            self.stop_calls += 1
            if self.fail_stop:
                raise RuntimeError("stop failed")

        def start_stream(self, size):
            pass

        def config_board(self, value):
            pass

        def get_board_data(self):
            return np.zeros((8, 0))

        def get_impedance_data(self):
            return []

    m.board = Backend()
    m.start_recording()
    m.begin_impedance_check(1)
    m.board.fail_stop = True
    clock[0] = 101
    with pytest.raises(RuntimeError, match="stop failed"):
        m.end_impedance_check()
    assert not m.streaming
    assert [e["code"] for e in m.impedance_check_events()] == [90, 91]
    m.board.fail_stop = False
    clock[0] = 102
    m.end_impedance_check()
    assert m.board.stop_calls == 3
    assert [e["code"] for e in m.impedance_check_events()] == [90, 91, 92, 93]
    assert m.recorded_eeg_gaps()[0]["length"] == 20
