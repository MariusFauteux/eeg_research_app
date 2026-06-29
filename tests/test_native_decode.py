"""Unit tests for the custom native-BLE Ganglion decoder.

No hardware needed at test time: most tests hand-craft 20-byte packets, but the
final test replays 10 *real* packets captured from a Ganglion (Ch1 held against
mains) and asserts the exact reconstructed counts -- a regression lock on the
field layout, sign, and channel order derived from that capture.

Protocol (verified against hardware, see tools/native_raw_dump.py): pure 19-bit deltas,
2 samples x 4 channels per packet, counter 100..199 wrapping 199->100 (no
periodic anchor). The decoder integrates from zero and holds flat on a genuine
packet gap (the fix for BrainFlow's once-per-second pulse).
"""

from __future__ import annotations

import queue

import numpy as np

from ganglion_studio.core.native_ganglion import (
    NUM_EEG,
    SCALE_UV_PER_COUNT,
    GanglionDecoder,
    NativeGanglionClient,
    _msb_to_count,
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


def _fw_encode_19(values):
    """Emulate firmware 3.0.2 compressData19: absolute 24-bit count -> 19-bit field."""
    out = []
    for V in values:
        u = V & 0xFFFFFF
        sign = (u >> 23) & 1
        u32 = u | (0xFF000000 if sign else 0)   # sign-extend to 32 bit
        if sign:
            u32 |= (1 << 5)                       # bitWrite(.,5,sign)
        else:
            u32 &= ~(1 << 5)
        shifted = u32 >> 5                        # arithmetic >>5
        if sign:
            shifted |= (~0 << 27)
        out.append(shifted & 0x7FFFF)
    return out


def make_msb_packet(packet_id: int, sample_a, sample_b) -> bytes:
    """id + 2 samples x 4 channels of firmware-3.0.2 MSB 19-bit fields."""
    fields = _fw_encode_19(sample_a) + _fw_encode_19(sample_b)
    v = 0
    for f in fields:
        v = (v << 19) | (f & 0x7FFFF)
    payload = v.to_bytes((19 * 2 * NUM_EEG) // 8, "big")
    return bytes([packet_id]) + payload


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
    assert dec.dropped == 1               # the lost packet is counted


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
# MSB decode (firmware 3.0.2+: absolute MSB-truncated samples, no deltas)
# --------------------------------------------------------------------------- #
def test_msb_to_count_round_trips_firmware_encode():
    """_msb_to_count is the exact inverse of compressData19 (to top 18 bits)."""
    for V in [0, 64, 12345, -64, -12345, 100000, -100000, 8388544, -8388608]:
        field = _fw_encode_19([V])[0]
        assert _msb_to_count(field, 19) == (V >> 6) << 6   # low 6 bits truncated


def test_msb_mode_decodes_absolute_values():
    dec = GanglionDecoder(mode="msb")
    sa = [64 * 100, 64 * 200, -64 * 300, 64 * 400]   # multiples of 64 round-trip exact
    sb = [64 * 101, 64 * 201, -64 * 301, 64 * 401]
    out = dec.decode(make_msb_packet(107, sa, sb))
    assert len(out) == 2
    for ch in range(NUM_EEG):
        assert round(out[0][ch] / SCALE_UV_PER_COUNT) == sa[ch]   # absolute, as-is
        assert round(out[1][ch] / SCALE_UV_PER_COUNT) == sb[ch]


def test_msb_mode_does_not_integrate():
    """A constant input stays constant in MSB mode (delta mode would ramp)."""
    const = [64 * 500] * NUM_EEG
    dec = GanglionDecoder(mode="msb")
    o1 = dec.decode(make_msb_packet(107, const, const))
    o2 = dec.decode(make_msb_packet(108, const, const))
    assert np.allclose(o1[0], o2[1])   # no drift across packets


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


def test_get_board_data_peels_loss_row():
    """The worker appends a loss-flag row; get_board_data keeps it in last_loss
    and returns the standard (num_rows, n) matrix without it."""
    client = _client()
    client._data_q = queue.Queue()  # std queue: get_nowait is synchronous here
    block = np.zeros((NUM_EEG + 2, 3))     # eeg(4) + timestamp + loss
    block[NUM_EEG] = [10.0, 11.0, 12.0]    # timestamp row
    block[NUM_EEG + 1] = [0.0, 1.0, 0.0]   # loss flag on the middle sample
    client._data_q.put(block)
    out = client.get_board_data()
    assert out.shape == (10, 3)            # standard layout, no extra row
    assert np.allclose(out[8], [10.0, 11.0, 12.0])  # timestamp_channel
    assert np.allclose(client.last_loss, [0.0, 1.0, 0.0])
    # next call with an empty queue resets last_loss
    assert client.get_board_data().shape == (10, 0)
    assert client.last_loss.size == 0
