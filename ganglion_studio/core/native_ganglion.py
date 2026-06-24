"""Custom native-Bluetooth driver for the OpenBCI Ganglion (no BrainFlow).

Why this exists
---------------
BrainFlow's native-BLE backend (``GANGLION_NATIVE_BOARD``) injects a *strong
pulse roughly once per second* into the EEG. The BLED112 dongle path is clean.
The Ganglion's 19-bit packet counter runs 101..200 and then wraps back to 101 --
that is exactly 200 samples = **one second** at 200 Hz. The most likely cause of
the pulse is that BrainFlow mistakes the once-per-second counter *wrap* for a
dropped packet and patches the stream in a way that produces a spike.

This module talks to the Ganglion directly with ``bleak`` (already a dependency),
implements the OpenBCI Ganglion BLE protocol + delta-decompression in plain
Python, and -- crucially -- handles the counter wrap correctly while only
"holding flat" on a *genuine* gap, so a lost packet can never become a spike.

The public class :class:`NativeGanglionClient` mimics the small subset of
BrainFlow's ``BoardShim`` API that :class:`~ganglion_studio.core.board_manager.
BoardManager` actually uses, so the manager barely changes.

Concurrency / macOS
-------------------
On macOS CoreBluetooth must run on a process's *main* thread; running it on a Qt
worker thread crashes the app (see ``ble_scanner.py``). ``BoardManager.prepare``/
``start`` run on a Qt worker thread, so the live BLE connection runs in a
dedicated **subprocess** whose ``asyncio.run`` owns the main thread. That also
isolates native BLE crashes from the GUI. The parent-side client is a thin proxy
that talks to the child over ``multiprocessing`` queues.

Not implemented yet
-------------------
Impedance (the ``z``/``Z`` LeadOff test) is *not* decoded here. In this mode the
impedance panel shows "unknown"; use the BLED112 dongle for impedance checks.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# --- Ganglion BLE GATT (OpenBCI Ganglion / Simblee radio) -----------------
SERVICE_UUID = "0000fe84-0000-1000-8000-00805f9b34fb"
NOTIFY_UUID = "2d30c082-f39f-4ce6-923f-3484ea480596"  # board -> host (data)
WRITE_UUID = "2d30c083-f39f-4ce6-923f-3484ea480596"   # host -> board (commands)

# EEG count -> microvolts. Same constant BrainFlow uses for the Ganglion
# (MCP3912 ADC, 1.2 V reference, gain 51, extra 1.5 divider). Cross-check the
# amplitude against the dongle on real hardware; if it is off, this single
# number is the only thing to change.
SCALE_UV_PER_COUNT = 1.2e6 / (8388607.0 * 1.5 * 51.0)

NUM_EEG = 4  # the Ganglion has 4 EEG channels


def _sign_extend(value: int, bits: int) -> int:
    """Interpret an unsigned ``bits``-wide field as a two's-complement integer."""
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


def _unpack_fields(payload: bytes, width: int, count: int) -> List[int]:
    """Split the first ``count*width`` bits of ``payload`` into signed ints.

    The Ganglion packs its delta fields big-endian and back-to-back (no byte
    alignment), so we read the bytes as one big integer and slice fixed-width
    fields from the most-significant end.
    """
    nbits = width * count
    nbytes = nbits // 8
    v = int.from_bytes(payload[:nbytes], "big")
    mask = (1 << width) - 1
    out: List[int] = []
    for k in range(count):
        field = (v >> (nbits - (k + 1) * width)) & mask
        out.append(_sign_extend(field, width))
    return out


class GanglionDecoder:
    """Decode 20-byte Ganglion BLE packets into microvolt samples.

    Stateful: compressed packets carry *deltas* relative to the previous sample,
    so we keep a running value per channel and integrate from zero. Call
    :meth:`decode` once per packet; it returns a list of samples, each a list of
    ``NUM_EEG`` microvolt floats (0 for a non-EEG packet, 2 for a delta packet,
    1 for the rare uncompressed anchor).

    Protocol facts verified against real hardware (see native_raw_dump.py):
      * byte[0] is a sample counter. With the accelerometer OFF (the app sends
        'N') it runs 100..199 and wraps 199->100 -- exactly 200 samples = 1 s.
      * Each packet carries 2 samples x 4 channels = 8 signed 19-bit deltas,
        packed big-endian back-to-back across the 19 payload bytes.
      * There is NO periodic uncompressed "anchor": the stream is pure deltas.
        The absolute DC level is therefore arbitrary (set by integrating from 0)
        and is removed downstream by the app's detrend / high-pass anyway.
      * Native BLE drops packets, which is what produced BrainFlow's 1 s pulse:
        applying a post-gap packet's deltas onto a stale reference smears a large
        step. We detect the counter discontinuity and hold flat instead.
    """

    def __init__(self) -> None:
        self._running = [0] * NUM_EEG     # running raw counts per channel
        self._last_id: Optional[int] = None

    def _is_continuous(self, packet_id: int) -> bool:
        """True if ``packet_id`` directly follows the previous one (no loss).

        Normal step is +1. The once-per-second band wrap (199->100, or 200->101
        on firmwares that use 101..200) is also continuous and must NOT count as
        a gap -- that mishandled wrap is exactly what makes BrainFlow pulse.
        """
        if self._last_id is None or self._last_id == 0:
            return True  # fresh start, or right after an id-0 anchor reference
        if packet_id == self._last_id + 1:
            return True
        # band wrap: from near the top (>=198) back down to near the bottom (<=101)
        if self._last_id >= 198 and packet_id <= 101:
            return True
        return False

    # -- main entry point ---------------------------------------------------
    def decode(self, packet: bytes) -> List[List[float]]:
        if len(packet) < 2:
            return []
        packet_id = packet[0]
        payload = packet[1:]

        # id 0: an uncompressed sample (4 channels x 24-bit signed). Rare/optional
        # on this firmware, but if it appears it sets the absolute reference.
        if packet_id == 0:
            for ch in range(NUM_EEG):
                raw = int.from_bytes(payload[ch * 3 : ch * 3 + 3], "big")
                self._running[ch] = _sign_extend(raw, 24)
            self._last_id = 0
            return [self._emit()]
        # (delta packets handled below)

        if not 1 <= packet_id <= 200:
            # 201..205 impedance, 206..207 ASCII messages -- not EEG, ignore.
            return []

        # Accel-OFF (normal here) packs 19-bit deltas with ids 100..199; the
        # accel-ON 18-bit mode (ids 1..99) is handled defensively but unused.
        width = 19 if packet_id >= 100 else 18
        continuous = self._is_continuous(packet_id)
        self._last_id = packet_id

        if not continuous:
            # A packet was lost: the incoming deltas are relative to a sample we
            # never saw, so applying them would smear a wrong step across the seam
            # (the pulse). Hold the last value flat for this packet instead. The
            # small DC step is removed by the app's high-pass; a couple of flat ms
            # is invisible next to a full-scale spike.
            return [self._emit(), self._emit()]

        # Two samples per packet, 4 channels each, signed deltas.
        deltas = _unpack_fields(payload, width, 2 * NUM_EEG)
        out: List[List[float]] = []
        for s in range(2):
            for ch in range(NUM_EEG):
                # OpenBCI convention: new = previous - delta. (Polarity only; if it
                # looks inverted versus the dongle, flip this one sign.) The DC
                # level is arbitrary and removed by the display's detrend/high-pass;
                # we deliberately do NOT clamp here -- clamping pins the board's
                # idle/floating frame to the rail and collapses the trace.
                self._running[ch] -= deltas[s * NUM_EEG + ch]
            out.append(self._emit())
        return out

    def _emit(self) -> List[float]:
        return [self._running[ch] * SCALE_UV_PER_COUNT for ch in range(NUM_EEG)]


# ---------------------------------------------------------------------------
# Subprocess BLE worker (runs in a child process; CoreBluetooth-legal on macOS)
# ---------------------------------------------------------------------------
def _ble_worker_main(address, timeout, data_q, ctrl_q, status_q) -> None:  # pragma: no cover - needs hardware
    """Child-process entry point: connect, stream, decode, ship chunks home."""
    import asyncio

    async def run() -> None:
        try:
            from bleak import BleakClient
        except Exception as exc:  # import guard
            status_q.put(("error", f"bleak not available: {exc}"))
            return

        decoder = GanglionDecoder()

        def on_notify(_handle, data: bytearray) -> None:
            # Never let a decode hiccup propagate into bleak's callback machinery
            # (an unhandled exception there can tear down the notification loop).
            try:
                samples = decoder.decode(bytes(data))
                if not samples:
                    return
                arr = np.asarray(samples, dtype=np.float64).T          # (NUM_EEG, k)
                ts = np.full(arr.shape[1], time.time(), dtype=np.float64)
                data_q.put(np.vstack([arr, ts]))                       # (NUM_EEG+1, k)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("native decode/notify error: %s", exc)

        def on_disconnect(_client) -> None:
            status_q.put(("disconnected", ""))

        client = BleakClient(address, timeout=timeout, disconnected_callback=on_disconnect)
        try:
            await client.connect()
        except Exception as exc:
            status_q.put(("error", f"connect failed: {exc}"))
            return
        status_q.put(("connected", ""))

        notifying = False
        try:
            while True:
                try:
                    kind, arg = ctrl_q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.005)  # let bleak service notifications
                    continue
                if kind == "start":
                    if not notifying:
                        await client.start_notify(NOTIFY_UUID, on_notify)
                        notifying = True
                    await client.write_gatt_char(WRITE_UUID, b"b", response=False)
                elif kind == "stop":
                    await client.write_gatt_char(WRITE_UUID, b"s", response=False)
                    if notifying:
                        await client.stop_notify(NOTIFY_UUID)
                        notifying = False
                elif kind == "cmd":
                    await client.write_gatt_char(WRITE_UUID, arg, response=False)
                elif kind == "release":
                    break
        finally:
            try:
                if notifying:
                    await client.stop_notify(NOTIFY_UUID)
            except Exception:
                pass
            try:
                await client.disconnect()
            except Exception:
                pass

    asyncio.run(run())


class NativeGanglionClient:
    """BrainFlow-``BoardShim``-shaped facade over the custom BLE driver.

    Implements only the methods :class:`BoardManager` calls: ``prepare_session``,
    ``start_stream``, ``stop_stream``, ``get_board_data``, ``config_board``,
    ``insert_marker``, ``release_session``. Data is returned in the same
    ``(num_rows, n)`` row layout BrainFlow uses, so every downstream consumer is
    unchanged.
    """

    def __init__(
        self,
        address: str,
        num_rows: int,
        eeg_channels: List[int],
        timestamp_channel: int,
        marker_channel: int,
        timeout: float = 20.0,
    ) -> None:
        if not address:
            raise ValueError("a BLE address/UUID is required for native connection")
        self.address = address
        self.num_rows = num_rows
        self.eeg_channels = list(eeg_channels)[:NUM_EEG]
        self.timestamp_channel = timestamp_channel
        self.marker_channel = marker_channel
        self.timeout = timeout

        ctx = mp.get_context("spawn")  # required on macOS; safe everywhere
        self._data_q: mp.Queue = ctx.Queue()
        self._ctrl_q: mp.Queue = ctx.Queue()
        self._status_q: mp.Queue = ctx.Queue()
        self._ctx = ctx
        self._proc: Optional[mp.process.BaseProcess] = None
        self._pending_marker: Optional[float] = None
        self._disconnected = False

    # -- lifecycle ----------------------------------------------------------
    def prepare_session(self) -> None:
        self._proc = self._ctx.Process(
            target=_ble_worker_main,
            args=(self.address, self.timeout, self._data_q, self._ctrl_q, self._status_q),
            name="ganglion-ble",
            daemon=True,
        )
        self._proc.start()
        # Block until the child reports the connection result (or times out).
        try:
            kind, detail = self._status_q.get(timeout=self.timeout + 5.0)
        except queue.Empty:
            self._terminate()
            raise RuntimeError("Ganglion BLE connection timed out")
        if kind != "connected":
            self._terminate()
            raise RuntimeError(f"Ganglion BLE connection failed: {detail}")
        logger.info("Native Ganglion connected: %s", self.address)

    def start_stream(self, _buffer_size: int = 0) -> None:
        self._ctrl_q.put(("start", None))

    def stop_stream(self) -> None:
        self._ctrl_q.put(("stop", None))

    def release_session(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._ctrl_q.put(("release", None))
            self._proc.join(timeout=3.0)
        self._terminate()

    def _terminate(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)
        self._proc = None

    # -- commands -----------------------------------------------------------
    def config_board(self, command: str) -> str:
        self._ctrl_q.put(("cmd", command.encode("ascii", "ignore")))
        return ""  # BrainFlow returns the board's reply string; we don't read it

    def insert_marker(self, value: float) -> None:
        # Stamp the marker on the most recent sample of the next get_board_data.
        self._pending_marker = float(value)

    @property
    def disconnected(self) -> bool:
        """True once the BLE link dropped or the worker process died."""
        if self._disconnected:
            return True
        # Drain any status the worker reported (e.g. an unexpected disconnect).
        while True:
            try:
                kind, _ = self._status_q.get_nowait()
            except queue.Empty:
                break
            if kind in ("disconnected", "error"):
                self._disconnected = True
        if self._proc is not None and not self._proc.is_alive():
            self._disconnected = True
        return self._disconnected

    # -- data ---------------------------------------------------------------
    def get_board_data(self) -> np.ndarray:
        chunks: List[np.ndarray] = []
        while True:
            try:
                chunks.append(self._data_q.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return np.zeros((self.num_rows, 0))
        block = np.concatenate(chunks, axis=1)  # (NUM_EEG+1, n): eeg rows + ts
        return self._to_matrix(block)

    def _to_matrix(self, block: np.ndarray) -> np.ndarray:
        """Map a (NUM_EEG+1, n) eeg+timestamp block onto the BrainFlow row layout."""
        n = block.shape[1]
        out = np.zeros((self.num_rows, n), dtype=np.float64)
        for i, row in enumerate(self.eeg_channels):
            out[row] = block[i]
        out[self.timestamp_channel] = block[NUM_EEG]
        if self._pending_marker is not None and n:
            out[self.marker_channel, -1] = self._pending_marker
            self._pending_marker = None
        return out
