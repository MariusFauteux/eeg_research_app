"""Unit tests for the custom native-BLE Ganglion decoder.

No hardware needed at test time: most tests hand-craft 20-byte packets, but the
final test replays 10 *real* packets captured from a Ganglion (Ch1 held against
mains) and asserts the exact reconstructed counts -- a regression lock on the
field layout, sign, and channel order derived from that capture.

Protocol (verified against hardware, see native_raw_dump.py): pure 19-bit deltas,
2 samples x 4 channels per packet, counter 100..199 wrapping 199->100 (no
periodic anchor). The decoder integrates from zero and holds flat on a genuine
packet gap (the fix for BrainFlow's once-per-second pulse).
"""

from __future__ import annotations

import numpy as np

from ganglion_studio.core.native_ganglion import (
    NUM_EEG,
    SCALE_UV_PER_COUNT,
    GanglionDecoder,
    NativeGanglionClient,
)


def make_anchor(values) -> bytes:
    """id 0 + four 24-bit signed counts, padded to a 20-byte packet."""
    payload = bytearray()
    for v in values:
        payload += int(v & 0xFFFFFF).to_bytes(3, "big")
    payload += bytes(19 - len(payload))
    return bytes([0]) + bytes(payload)


def make_delta(packet_id: int, deltas, width: int = 19) -> bytes:
    """id + 8 signed delta fields (2 samples x 4 ch) packed big-endian."""
    count = 2 * NUM_EEG
    mask = (1 << width) - 1
    v = 0
    for d in deltas:
        v = (v << width) | (d & mask)
    nbytes = (width * count) // 8
    payload = bytearray(v.to_bytes(nbytes, "big"))
    payload += bytes(19 - len(payload))
    return bytes([packet_id]) + bytes(payload)


# --------------------------------------------------------------------------- #
# decoder
# --------------------------------------------------------------------------- #
def test_anchor_decode():
    values = [100, -200, 300, -400]
    dec = GanglionDecoder()
    out = dec.decode(make_anchor(values))
    assert len(out) == 1
    assert np.allclose(out[0], [v * SCALE_UV_PER_COUNT for v in values])


def test_delta_integration_from_zero_no_anchor():
    """No anchor exists on real hardware: integrate deltas from zero.
    new = previous - delta; a constant delta of -1 makes a +1/sample ramp."""
    dec = GanglionDecoder()
    counts = []
    for pid in (100, 101, 102):  # first packet has no predecessor -> still applied
        for sample in dec.decode(make_delta(pid, [-1] * (2 * NUM_EEG))):
            counts.append(sample[0] / SCALE_UV_PER_COUNT)
    assert np.allclose(counts, [1, 2, 3, 4, 5, 6])


def test_counter_wrap_is_not_a_gap():
    """199 -> 100 is the normal once-per-second wrap: deltas must still apply."""
    dec = GanglionDecoder()
    dec._last_id = 199
    dec._running = [10, 10, 10, 10]
    out = dec.decode(make_delta(100, [-5] * (2 * NUM_EEG)))
    counts = [out[0][0] / SCALE_UV_PER_COUNT, out[1][0] / SCALE_UV_PER_COUNT]
    assert np.allclose(counts, [15, 20])  # applied, not held flat


def test_dropped_packet_holds_flat_no_spike():
    """A genuine gap (151 missing) must NOT smear deltas into a spike."""
    dec = GanglionDecoder()
    dec._last_id = 150
    dec._running = [10, 10, 10, 10]
    out = dec.decode(make_delta(152, [-100000] * (2 * NUM_EEG)))
    counts = [s[0] / SCALE_UV_PER_COUNT for s in out]
    assert np.allclose(counts, [10, 10])  # held flat


def test_impedance_and_ascii_packets_ignored():
    dec = GanglionDecoder()
    assert dec.decode(bytes([203]) + bytes(19)) == []  # impedance id
    assert dec.decode(bytes([206]) + bytes(19)) == []  # ASCII message id


# 10 consecutive real packets (ids 107..116), Ch1 held against mains. Expected
# Ch1 (ch0) running counts hand-derived from the verified msb-first 19-bit decode.
_REAL_PACKETS = [
    "6b135600edb01d7903b1a151ec19ac032f30668a",
    "6c18e501c4a0384c0714e16a4c1284024ff04a38",
    "6d15934110a821e1043e01622c1a85834d106a58",
    "6e15ad81911031d00642017e4813b88272e04f0c",
    "6f166bc11c2823170470e1a3ec1e8f03c9c07a3a",
    "7018f901c2a83824070f814494100182037040b8",
    "7110e080c2a81812030b813d00182882fd9060bc",
    "72125a015b482b49056ea1039c0c0f8180b030d6",
    "730e534099e01318026fa0dd0c1244024240493c",
    "740cbbc0ffe0200d040c008b1404828090601256",
]
_EXPECTED_CH0 = [
    -39600, -82854, -133838, -180212, -224398, -269732, -314128, -363060,
    -408978, -462728, -513872, -555418, -589982, -630558, -668142, -701372,
    -730710, -759004, -785082, -802884,
]


def test_real_hardware_packets_regression():
    dec = GanglionDecoder()
    ch0 = []
    for h in _REAL_PACKETS:
        for sample in dec.decode(bytes.fromhex(h)):
            ch0.append(round(sample[0] / SCALE_UV_PER_COUNT))
    assert ch0 == _EXPECTED_CH0


# --------------------------------------------------------------------------- #
# row layout / marker (client matrix builder, no subprocess)
# --------------------------------------------------------------------------- #
def _client():
    return NativeGanglionClient(
        address="dummy",
        num_rows=10,
        eeg_channels=[1, 2, 3, 4],
        timestamp_channel=8,
        marker_channel=7,
    )


def test_to_matrix_layout():
    client = _client()
    block = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [100.0, 101.0]]
    )
    out = client._to_matrix(block)
    assert out.shape == (10, 2)
    assert np.allclose(out[1], [1.0, 2.0])
    assert np.allclose(out[4], [7.0, 8.0])
    assert np.allclose(out[8], [100.0, 101.0])
    assert np.allclose(out[7], [0.0, 0.0])


def test_to_matrix_marker_stamped_once():
    client = _client()
    client.insert_marker(5.0)
    out = client._to_matrix(np.zeros((NUM_EEG + 1, 3)))
    assert out[7, -1] == 5.0
    assert out[7, 0] == 0.0 and out[7, 1] == 0.0
    out2 = client._to_matrix(np.zeros((NUM_EEG + 1, 2)))
    assert np.allclose(out2[7], [0.0, 0.0])
