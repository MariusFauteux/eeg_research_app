"""Impedance acquisition records fresh channel events, never held caches."""

import queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.impedance import ImpedanceReading
from ganglion_studio.core.native_ganglion import (
    NativeGanglionClient, _ble_worker_main, decode_impedance,
)


@pytest.mark.parametrize("packet_id", range(201, 206))
def test_native_impedance_packet_has_own_channel_and_arrival_time(packet_id):
    assert decode_impedance(bytes([packet_id]) + b"22Z\n", 123.5) == (
        ImpedanceReading(123.5, packet_id - 201, 22000.0))


def test_legacy_units_are_explicit_without_half_value_heuristic():
    assert decode_impedance(b"\xc912345Z", 10, "ohms").ohms == 12345
    assert decode_impedance(b"\xc912345Z", 10, "kohms").ohms == 12345000
    assert decode_impedance(b"\xcd0Z", 10).ohms == 0


@pytest.mark.parametrize("packet", [b"", b"\xc9", b"\xc922", b"\xc9Z", b"\xc9-1Z",
                                      b"\xc9nanZ", b"\xc922.5Z", b"\xce22Z"])
def test_malformed_or_non_impedance_packets_are_not_measurements(packet):
    assert decode_impedance(packet) is None


def test_native_queue_drains_once_and_keeps_identical_fresh_values():
    client = NativeGanglionClient("dummy", 8, [1, 2, 3, 4], 6, 7)
    client._impedance_q = queue.Queue()
    readings = [ImpedanceReading(1, 0, 22000), ImpedanceReading(2, 0, 22000)]
    for reading in readings:
        client._impedance_q.put(reading)
    assert client.get_impedance_data() == readings
    assert client.get_impedance_data() == []
    assert client.get_board_data().shape == (8, 0)


def test_native_impedance_subscribes_before_z_and_stops_with_capital_z(monkeypatch):
    calls = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def connect(self):
            calls.append("connect")

        async def start_notify(self, uuid, callback):
            calls.append("subscribe")
            callback(None, bytearray(b"\xc922Z"))

        async def write_gatt_char(self, uuid, value, response=False):
            calls.append(value)

        async def stop_notify(self, uuid):
            calls.append("unsubscribe")

        async def disconnect(self):
            calls.append("disconnect")

    monkeypatch.setitem(sys.modules, "bleak", SimpleNamespace(BleakClient=Client))
    commands, measurements = queue.Queue(), queue.Queue()
    for command in [("start", "impedance"), ("stop", None), ("release", None)]:
        commands.put(command)
    _ble_worker_main("dummy", 1, queue.Queue(), commands, queue.Queue(),
                     impedance_q=measurements)
    assert calls == ["connect", "subscribe", b"z", b"Z", "unsubscribe", "disconnect"]
    assert measurements.get_nowait().ohms == 22000


def _manager(monkeypatch, **kwargs):
    clock = [100.0]
    monkeypatch.setattr("ganglion_studio.core.board_manager.time.time", lambda: clock[0])
    manager = BoardManager(acquisition_mode="impedance", **kwargs)
    manager.board_id = 46
    # Explicit row layout allows tests to exercise held BrainFlow caches.
    manager.num_rows = 13
    manager.eeg_channels = [1, 2, 3, 4]
    manager.channel_names = ["A", "B", "C", "D"]
    manager.resistance_channels = [7, 8, 9, 10, 11]
    manager.package_num_channel = 0
    manager.timestamp_channel = 12
    manager.streaming = True
    return manager, clock


class EventBoard:
    def __init__(self, readings=()):
        self.pending = list(readings)

    def get_impedance_data(self):
        pending, self.pending = self.pending, []
        return pending


def test_recording_fresh_events_is_independent_of_bounded_live_history(monkeypatch):
    m, clock = _manager(monkeypatch, impedance_history_limit=2)
    m.board = EventBoard()
    m.start_recording()
    events = [ImpedanceReading(100.1 + i, 0, 22000) for i in range(3)]
    m.board.pending = [ImpedanceReading(99, 4, 1)] + events
    clock[0] = 104
    assert m.poll() == 4
    assert m.poll() == 0
    assert m.recorded_sample_count() == 3
    assert m.impedance_history() == events[-2:]
    assert m.latest_impedance_readings() == [events[-1], ImpedanceReading(99, 4, 1)]
    assert m.impedance_channel_names == ["A", "B", "C", "D", "REF"]
    assert m.impedance_history(seconds=2) == events[-1:]
    assert m.stop_recording() == events
    assert m.stop_recording() == []


def test_brainflow_packet_id_identifies_only_fresh_channel_not_cached_rows(monkeypatch):
    m, clock = _manager(monkeypatch)
    data = np.zeros((13, 3))
    data[0] = [201, 202, 202]
    data[7] = [12, 12, 12]  # Ch1 remains a held cache in later packets.
    data[8] = [99, 15, 15]  # Ch2's repeated15 is a fresh measurement each time.
    data[12] = [100.1, 100.2, 100.3]

    class BrainFlowBoard:
        def get_board_data(self):
            nonlocal data
            output, data = data, np.zeros((13, 0))
            return output

    m.board = BrainFlowBoard()
    m.start_recording()
    clock[0] = 101
    readings = m.stop_recording()
    assert readings == [ImpedanceReading(100.1, 0, 12000),
                        ImpedanceReading(100.2, 1, 15000),
                        ImpedanceReading(100.3, 1, 15000)]
    assert m._filled == 0


def test_demo_measurements_have_clock_cadence_not_gui_poll_cadence(monkeypatch):
    m, clock = _manager(monkeypatch, demo=True)
    m.board = SimpleNamespace(get_board_data=lambda: np.zeros((13, 0)))
    m.start_recording()
    assert m.poll() == 5
    for _ in range(100):
        assert m.poll() == 0
    clock[0] += 1
    assert m.poll() == 5
    assert len(m.stop_recording()) == 10
    assert m._filled == 0


def test_recording_mode_changes_are_rejected_before_hardware_command(monkeypatch):
    m, clock = _manager(monkeypatch)
    m.board = EventBoard()
    m.start_recording()
    with pytest.raises(RuntimeError, match="Stop recording"):
        m.stop_impedance()
    with pytest.raises(RuntimeError, match="Stop recording"):
        m.send_raw_command("Z")
    with pytest.raises(RuntimeError, match="Stop recording"):
        m.send_raw_command("b")
    with pytest.raises(ValueError, match="separately"):
        m.send_raw_command("zb")
    assert m.acquisition_mode == "impedance"
    clock[0] += 1
    m.stop_recording()


def test_backend_mode_is_selected_before_stream_start():
    calls = []
    board = SimpleNamespace(config_board=lambda value: calls.append(value),
                            start_stream=lambda value: calls.append("start"),
                            stop_stream=lambda: calls.append("stop"))
    m = BoardManager(acquisition_mode="impedance")
    m.board = board
    m.start()
    assert calls == ["z", "start"]
    m.stop()
    assert calls == ["z", "start", "stop"]
