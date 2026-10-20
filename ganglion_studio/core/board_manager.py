"""BrainFlow board wrapper for the OpenBCI Ganglion (native BLE) + demo mode.

The :class:`BoardManager` owns the BrainFlow ``BoardShim`` and a single ring
buffer that always holds the most recent ``buffer_seconds`` of *raw* data for
every BrainFlow row (EEG, accel, resistance, marker, timestamp, ...).

A single producer (the :meth:`poll` method, driven by a dedicated acquisition
thread) pulls new samples with ``get_board_data`` (which removes them from
BrainFlow's internal buffer) and appends them to the ring buffer and, when
recording, to a record buffer. All visualization widgets are pure consumers
that read recent slices. A lock serializes the (thread) producer against the
(GUI thread) consumers.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from brainflow.board_shim import (
    BoardIds,
    BoardShim,
    BrainFlowInputParams,
    BrainFlowError,
)
from brainflow.exit_codes import BrainFlowExitCodes

from . import board_config as cfg
from .impedance import ImpedanceReading, impedance_scale
from .impedance_check import EEG_STOP, IMPEDANCE_START, IMPEDANCE_END, EEG_RESUME, CHECK_MARKERS

logger = logging.getLogger(__name__)

# Number of EEG channels exposed for a Ganglion. Even in demo mode (synthetic
# board with 16 channels) we only surface the first 4 so the UI matches real
# hardware.
GANGLION_CHANNELS = 4


@dataclass
class BoardManager:
    """Manage a BrainFlow session and a shared ring buffer of recent data."""

    demo: bool = False
    mac_address: str = ""
    serial_number: str = ""
    serial_port: str = ""  # set -> connect via the BLED112 dongle, not native BLE
    firmware: str = "3"  # "3" (default), "2", or "auto"
    # For native Bluetooth, use our own bleak driver instead of BrainFlow's
    # native backend (BrainFlow's injects a once-per-second pulse). Ignored for
    # demo and dongle modes. See core/native_ganglion.py.
    use_custom_native: bool = True
    # Native-BLE sample encoding: "delta" (firmware <= 2.x) or "msb" (firmware
    # 3.0.2+, which sends absolute MSB-truncated samples). Custom native only.
    decode_mode: str = "delta"
    buffer_seconds: float = 30.0
    acquisition_mode: str = "eeg"
    impedance_units: str = "kohms"
    impedance_history_limit: int = 50000

    # --- runtime state (not constructor args) ---
    # Either a BrainFlow BoardShim or a NativeGanglionClient -- both expose the
    # same small method set this class uses (prepare_session, start_stream,
    # stop_stream, get_board_data, config_board, insert_marker, release_session).
    board: Optional[object] = field(default=None, init=False)
    board_id: int = field(default=0, init=False)
    sampling_rate: int = field(default=200, init=False)
    num_rows: int = field(default=0, init=False)
    eeg_channels: List[int] = field(default_factory=list, init=False)
    resistance_channels: List[int] = field(default_factory=list, init=False)
    marker_channel: int = field(default=0, init=False)
    timestamp_channel: int = field(default=0, init=False)
    package_num_channel: int = field(default=0, init=False)

    channel_active: List[bool] = field(default_factory=list, init=False)
    channel_names: List[str] = field(default_factory=list, init=False)
    channel_types: List[str] = field(default_factory=list, init=False)
    electrodes: List[str] = field(default_factory=list, init=False)
    placements: List[str] = field(default_factory=list, init=False)
    streaming: bool = field(default=False, init=False)
    impedance_mode: bool = field(default=False, init=False)

    _ring: Optional[np.ndarray] = field(default=None, init=False)
    _filled: int = field(default=0, init=False)
    _buffer_len: int = field(default=0, init=False)
    # Circular-buffer write cursor: index of the next column to write. The most
    # recent sample lives at (_write - 1) % _buffer_len.
    _write: int = field(default=0, init=False)

    recording: bool = field(default=False, init=False)
    _record_chunks: List[np.ndarray] = field(default_factory=list, init=False)
    _record_count: int = field(default=0, init=False)
    _record_started_at: Optional[float] = field(default=None, init=False)
    _record_stopped_at: Optional[float] = field(default=None, init=False)
    _record_marker_requests: list = field(default_factory=list, init=False)
    _record_marker_events: list = field(default_factory=list, init=False)

    # BLE packet-loss tracking (native driver only; stays 0 for demo/dongle). The
    # loss ring is 1-D and aligned column-for-column with the data ring, so a drop
    # can be placed exactly on the trace and saved alongside a recording.
    _loss_ring: Optional[np.ndarray] = field(default=None, init=False)
    _dropped_total: int = field(default=0, init=False)
    _record_loss_idx: List[int] = field(default_factory=list, init=False)

    # demo impedance simulation state
    _demo_imp: Optional[np.ndarray] = field(default=None, init=False)
    _demo_imp_next: float = field(default=0.0, init=False)
    _impedance_history: deque = field(default_factory=deque, init=False)
    _impedance_latest: dict = field(default_factory=dict, init=False)
    _record_impedance: list = field(default_factory=list, init=False)
    _check_phase: str = field(default="idle", init=False)
    _check_started_at: Optional[float] = field(default=None, init=False)
    _check_gap_started_at: Optional[float] = field(default=None, init=False)
    _check_duration_s: Optional[float] = field(default=None, init=False)
    _check_impedance_started: bool = field(default=False, init=False)
    _check_impedance_ended: bool = field(default=False, init=False)
    _check_gap_samples: int = field(default=0, init=False)
    _check_gap_record_start: Optional[int] = field(default=None, init=False)
    _check_events: list = field(default_factory=list, init=False)
    _record_eeg_gaps: list = field(default_factory=list, init=False)
    _record_check_readings: list = field(default_factory=list, init=False)
    _check_last_stop_at: Optional[float] = field(default=None, init=False)

    # Serializes the acquisition thread (producer) against GUI-thread readers.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    # Data draining and recording transitions must share one producer boundary.
    _poll_lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    # Background acquisition thread.
    acquisition_hz: float = 60.0
    _acq_thread: Optional[threading.Thread] = field(default=None, init=False)
    _acq_stop: threading.Event = field(default_factory=threading.Event, init=False)
    _acq_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _lifecycle_lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _releasing: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.acquisition_mode not in ("eeg", "impedance"):
            raise ValueError("Acquisition mode must be 'eeg' or 'impedance'")
        impedance_scale(self.impedance_units)
        if self.impedance_history_limit < 1:
            raise ValueError("Impedance history limit must be positive")
        self._impedance_history = deque(maxlen=self.impedance_history_limit)

    @property
    def impedance_channel_names(self) -> List[str]:
        return [(self.channel_names[i] if i < len(self.channel_names) else f"Ch{i + 1}")
                for i in range(4)] + ["REF"]

    @property
    def impedance_raw_unit_basis(self) -> str:
        return self.impedance_units

    @property
    def impedance_check_active(self) -> bool:
        return self._check_phase != "idle"

    @property
    def impedance_check_status(self) -> str:
        return self._check_phase

    @property
    def impedance_check_started_at(self) -> Optional[float]:
        return self._check_started_at

    @property
    def impedance_check_duration_s(self) -> Optional[float]:
        return self._check_duration_s

    def impedance_check_events(self) -> list:
        with self._lock:
            return [dict(event) for event in self._check_events]

    def recorded_eeg_gaps(self) -> list:
        with self._lock:
            return [dict(gap) for gap in self._record_eeg_gaps]

    def recorded_impedance_checks(self) -> List[ImpedanceReading]:
        with self._lock:
            return list(self._record_check_readings)

    # ------------------------------------------------------------------ setup
    def _resolve_board_id(self) -> int:
        if self.demo:
            return BoardIds.SYNTHETIC_BOARD.value
        if self.serial_port:
            return BoardIds.GANGLION_BOARD.value  # BLED112 dongle (serial port)
        return BoardIds.GANGLION_NATIVE_BOARD.value

    def _build_params(self) -> BrainFlowInputParams:
        params = BrainFlowInputParams()
        if not self.demo:
            if self.serial_port:
                params.serial_port = self.serial_port  # BLED112 dongle
            if self.mac_address:
                params.mac_address = self.mac_address
            if self.serial_number:
                params.serial_number = self.serial_number
            # FW3 is the default in BrainFlow >=5.21. Only override otherwise.
            if self.firmware and self.firmware != "3":
                params.other_info = f"fw:{self.firmware}"
            params.timeout = 20
        return params

    def _load_descriptor(self) -> None:
        descr = BoardShim.get_board_descr(self.board_id)
        self.sampling_rate = int(descr["sampling_rate"])
        self.num_rows = int(descr["num_rows"])
        self.eeg_channels = list(descr.get("eeg_channels", []))[:GANGLION_CHANNELS]
        self.resistance_channels = list(descr.get("resistance_channels", []))
        self.marker_channel = int(descr.get("marker_channel", 0))
        self.timestamp_channel = int(descr.get("timestamp_channel", 0))
        self.package_num_channel = int(descr.get("package_num_channel", 0))
        n = len(self.eeg_channels)
        self.channel_active = [True] * n
        self.channel_names = [f"Ch{i + 1}" for i in range(n)]
        self.channel_types = ["EEG"] * n
        self.electrodes = [cfg.ELECTRODES[0]] * n
        self.placements = ["None"] * n

    def set_channel_config(self, names, types, electrodes, placements) -> None:
        """Apply per-channel display names + type/electrode/placement metadata."""
        n = len(self.eeg_channels)
        self.channel_names = list(names)[:n]
        self.channel_types = list(types)[:n]
        self.electrodes = list(electrodes)[:n]
        self.placements = list(placements)[:n]

    def _use_custom_native(self) -> bool:
        """True when we should use our own bleak driver (native BLE, not demo/dongle)."""
        return self.use_custom_native and not self.demo and not self.serial_port

    def prepare(self) -> None:
        """Create the session and prepare the board. May block while BLE connects."""
        BoardShim.disable_board_logger()
        self.board_id = self._resolve_board_id()
        # The descriptor (row layout, sampling rate) always comes from BrainFlow's
        # static board map -- it needs no connection -- so every downstream reader
        # sees the same layout regardless of which backend streams the data.
        self._load_descriptor()
        if self._use_custom_native():
            from .native_ganglion import NativeGanglionClient
            self.board = NativeGanglionClient(
                address=self.mac_address,
                num_rows=self.num_rows,
                eeg_channels=self.eeg_channels,
                timestamp_channel=self.timestamp_channel,
                marker_channel=self.marker_channel,
                decode_mode=self.decode_mode,
                impedance_units=self.impedance_units,
            )
            self.board.prepare_session()
        else:
            params = self._build_params()
            self.board = BoardShim(self.board_id, params)
            self.board.prepare_session()

        self._buffer_len = max(1, int(self.buffer_seconds * self.sampling_rate))
        self._ring = np.zeros((self.num_rows, self._buffer_len), dtype=np.float64)
        self._loss_ring = np.zeros(self._buffer_len, dtype=np.float64)
        self._filled = 0
        self._write = 0
        self._dropped_total = 0
        self._demo_imp = np.random.uniform(3.0, 25.0, size=len(self.eeg_channels))
        self._demo_imp_next = 0.0
        with self._lock:
            self._impedance_history.clear()
            self._impedance_latest.clear()
        logger.info("Board prepared: id=%s sr=%s", self.board_id, self.sampling_rate)

    def start(self) -> None:
        with self._lifecycle_lock:
            if self.board is None:
                raise RuntimeError("Board not prepared")
            if self.streaming:
                return
            if not self.demo:
                # BrainFlow selects its stream commands through z/Z before
                # start_stream. Native uses the same contract and subscribes
                # before writing z, without briefly starting EEG first.
                self.board.config_board("z" if self.acquisition_mode == "impedance" else "Z")
            self.board.start_stream(450000)
            self.streaming = True
            self.impedance_mode = self.acquisition_mode == "impedance"
            # Disable the accelerometer so the Ganglion streams 19-bit EEG deltas
            # (with accel on it drops to 18-bit). We trade motion data for resolution.
            if self.acquisition_mode == "eeg":
                self._safe_config(cfg.ACCEL_DISABLE)

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self.board is not None and self.streaming:
                try:
                    self.board.stop_stream()
                except BrainFlowError as exc:  # pragma: no cover - hardware path
                    logger.warning("stop_stream failed: %s", exc)
            self.streaming = False
            self.impedance_mode = False

    def release(self) -> None:
        """Release only after acquisition exits; callers should run this off-GUI."""
        with self._lifecycle_lock:
            with self._acq_lock:
                self._releasing = True
            try:
                if not self.stop_acquisition(timeout=None):
                    raise RuntimeError("Cannot release a board from its acquisition thread")
                self.stop()
                if self.board is not None:
                    try:
                        self.board.release_session()
                    except BrainFlowError as exc:  # pragma: no cover
                        logger.warning("release_session failed: %s", exc)
                self.board = None
            finally:
                with self._acq_lock:
                    self._releasing = False

    # ----------------------------------------------------------- acquisition
    def start_acquisition(self) -> bool:
        """Run :meth:`poll` on a dedicated thread, decoupled from rendering."""
        with self._acq_lock:
            if self._releasing or self.board is None or not self.streaming:
                return False
            if self._acq_thread is not None and self._acq_thread.is_alive():
                return False
            self._acq_stop = threading.Event()
            self._acq_thread = threading.Thread(
                target=self._acq_loop, name="ganglion-acquisition", daemon=True
            )
            self._acq_thread.start()
            return True

    def stop_acquisition(self, timeout: Optional[float] = 2.0) -> bool:
        """Request a stop, retaining the producer until it has actually exited.

        A zero timeout lets GUI callers request shutdown without waiting; use
        ``None`` off the GUI thread when backend release must wait for the poll.
        """
        with self._acq_lock:
            self._acq_stop.set()
            thread = self._acq_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        stopped = thread is None or not thread.is_alive()
        if stopped:
            with self._acq_lock:
                if self._acq_thread is thread:
                    self._acq_thread = None
        return stopped

    def _acq_loop(self) -> None:
        period = 1.0 / max(1.0, self.acquisition_hz)
        while not self._acq_stop.is_set():
            try:
                self.poll()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("acquisition poll error: %s", exc)
            self._acq_stop.wait(period)

    # --------------------------------------------------------------- polling
    def poll(self) -> int:
        """Pull all new samples into the ring buffer. Returns #new samples."""
        with self._poll_lock:
            return self._poll_data()

    def _poll_data(self, *, force: bool = False) -> int:
        """Drain data while holding ``_poll_lock``; force also drains when paused."""
        if self.board is None or (not self.streaming and not force and not self.impedance_check_active):
            return 0
        if self.impedance_check_active:
            n = self._poll_impedance() if self._check_phase == "measuring" else 0
            self._append_check_gap(time.time())
            return n
        if self.acquisition_mode == "impedance":
            return self._poll_impedance()
        try:
            data = self.board.get_board_data()
        except BrainFlowError as exc:  # pragma: no cover
            logger.warning("get_board_data failed: %s", exc)
            return 0
        return self._accept_eeg_data(data)

    def _accept_eeg_data(self, data, loss=None) -> int:
        n = data.shape[1] if data.ndim == 2 else 0
        if n == 0:
            return 0

        if self.demo:
            data = self._inject_demo_resistance(data)

        # Per-sample BLE-loss flags aligned with `data` (native driver only).
        if loss is None:
            loss = self._loss_flags(n)

        with self._lock:
            self._dropped_total += int(loss.sum())
            if self.recording:
                keep = self._recording_sample_mask(data)
                recorded = data[:, keep]
                drops = np.flatnonzero(loss[keep])
                if drops.size:
                    self._record_loss_idx.extend((self._record_count + drops).tolist())
                if recorded.shape[1]:
                    self._record_chunks.append(recorded.copy())
                    self._record_count += recorded.shape[1]
            self._append_ring(data, loss)
        return n

    def _poll_impedance(self) -> int:
        """Consume each measurement event once; never record held row caches."""
        if self.demo:
            # Drain synthetic EEG so a long impedance-only demo cannot fill its
            # backend buffer. These samples never enter the EEG or record ring.
            self.board.get_board_data()
            now = time.time()
            readings = []
            if now >= self._demo_imp_next:
                self._demo_imp_next = now + 1.0
                if self._demo_imp is None or len(self._demo_imp) != 5:
                    self._demo_imp = np.random.uniform(3.0, 25.0, 5)
                self._demo_imp = np.clip(self._demo_imp + np.random.normal(0, 0.2, 5), 0, 120)
                readings = [ImpedanceReading(now, i, float(value) * 1000)
                            for i, value in enumerate(self._demo_imp)]
        elif hasattr(self.board, "get_impedance_data"):
            readings = self.board.get_impedance_data()
        else:
            try:
                data = self.board.get_board_data()
            except BrainFlowError as exc:
                logger.warning("get_board_data failed: %s", exc)
                return 0
            readings = []
            scale = impedance_scale(self.impedance_units)
            if data.ndim == 2:
                for column in range(data.shape[1]):
                    packet = data[self.package_num_channel, column]
                    if not np.isfinite(packet) or packet != int(packet) or not 201 <= packet <= 205:
                        continue
                    channel = int(packet) - 201
                    if channel >= len(self.resistance_channels):
                        continue
                    value = float(data[self.resistance_channels[channel], column])
                    timestamp = float(data[self.timestamp_channel, column])
                    try:
                        readings.append(ImpedanceReading(timestamp, channel, value * scale))
                    except ValueError:
                        logger.warning("Invalid impedance measurement ignored")
        with self._lock:
            for reading in readings:
                self._impedance_history.append(reading)
                self._impedance_latest[reading.channel] = reading
                if self.impedance_check_active:
                    if (self.recording and self._check_started_at is not None
                            and reading.timestamp >= self._check_started_at
                            and (self._record_stopped_at is None
                                 or reading.timestamp <= self._record_stopped_at)):
                        self._record_check_readings.append(reading)
                    continue
                if (self.recording and reading.timestamp >= self._record_started_at
                        and (self._record_stopped_at is None
                             or reading.timestamp <= self._record_stopped_at)):
                    self._record_impedance.append(reading)
                    self._record_count += 1
        return len(readings)

    def impedance_history(self, seconds: Optional[float] = None) -> List[ImpedanceReading]:
        """Independent bounded event history; repeated equal readings are fresh."""
        cutoff = None if seconds is None else time.time() - max(0.0, seconds)
        with self._lock:
            return [reading for reading in self._impedance_history
                    if cutoff is None or reading.timestamp >= cutoff]

    def latest_impedance_readings(self) -> List[ImpedanceReading]:
        with self._lock:
            return [self._impedance_latest[channel] for channel in sorted(self._impedance_latest)]

    def _recording_sample_mask(self, data: np.ndarray) -> np.ndarray:
        """Exclude queued pre-start/post-stop samples when timestamps exist."""
        keep = np.ones(data.shape[1], dtype=bool)
        row = self.timestamp_channel
        if (self._record_started_at is None or not 0 <= row < data.shape[0]
                or row == self.marker_channel or row in self.eeg_channels):
            return keep
        timestamps = data[row]
        valid = np.isfinite(timestamps) & (timestamps > 0)
        keep[valid] = timestamps[valid] >= self._record_started_at
        if self._record_stopped_at is not None:
            keep[valid] &= timestamps[valid] <= self._record_stopped_at
        return keep

    def _loss_flags(self, n: int) -> np.ndarray:
        """Loss-flag array (length n) for the just-polled chunk; zeros unless the
        native driver reported drops aligned with this exact chunk."""
        flags = getattr(self.board, "last_loss", None)
        if flags is None or len(flags) != n:
            return np.zeros(n, dtype=np.float64)
        return np.asarray(flags, dtype=np.float64)

    def _append_ring(self, data: np.ndarray, loss: Optional[np.ndarray] = None) -> None:
        """Append new columns to the circular ring buffer (O(n), no shifting).

        The 1-D ``loss`` array (defaulting to zeros) is written to ``_loss_ring`` at
        the same indices, so the two rings stay aligned and ``recent`` /
        ``recent_loss`` agree. Skipped when no loss ring is allocated.
        """
        assert self._ring is not None
        length = self._buffer_len
        rows = min(data.shape[0], self._ring.shape[0])
        n = data.shape[1]
        if n <= 0:
            return
        if loss is None:
            loss = np.zeros(n, dtype=np.float64)
        track = self._loss_ring is not None
        if n >= length:
            # New chunk alone overfills the buffer: keep only its last `length`.
            self._ring[:rows, :] = data[:rows, -length:]
            if track:
                self._loss_ring[:] = loss[-length:]
            self._write = 0
            self._filled = length
            return
        end = self._write + n
        if end <= length:
            self._ring[:rows, self._write:end] = data[:rows, :]
            if track:
                self._loss_ring[self._write:end] = loss
        else:  # chunk wraps past the buffer end
            first = length - self._write
            self._ring[:rows, self._write:] = data[:rows, :first]
            self._ring[:rows, : n - first] = data[:rows, first:]
            if track:
                self._loss_ring[self._write:] = loss[:first]
                self._loss_ring[: n - first] = loss[first:]
        self._write = end % length
        self._filled = min(length, self._filled + n)

    def _inject_demo_resistance(self, data: np.ndarray) -> np.ndarray:
        """Synthesize plausible impedance values for demo mode."""
        if self._demo_imp is None or not self.resistance_channels:
            return data
        n = data.shape[1]
        # slow random walk so the impedance plot looks alive
        self._demo_imp += np.random.normal(0, 0.2, size=self._demo_imp.shape)
        self._demo_imp = np.clip(self._demo_imp, 1.0, 120.0)
        for i, row in enumerate(self.resistance_channels[: len(self._demo_imp)]):
            if row < data.shape[0]:
                data[row, :] = (self._demo_imp[i] * 1000.0) + np.random.normal(0, 50, n)
        return data

    # --------------------------------------------------------------- readers
    def recent(self, seconds: float) -> np.ndarray:
        """Return the most recent ``seconds`` of the full row matrix (copy)."""
        with self._lock:
            if self._ring is None or self._filled == 0:
                return np.zeros((self.num_rows, 0))
            length = self._buffer_len
            count = min(self._filled, max(1, int(seconds * self.sampling_rate)))
            end = self._write  # one past the most recent sample
            start = (end - count) % length
            if start < end:
                return self._ring[:, start:end].copy()
            # Window wraps the buffer end: stitch tail + head in time order.
            return np.concatenate(
                (self._ring[:, start:], self._ring[:, :end]), axis=1
            )

    def recent_eeg(self, seconds: float) -> np.ndarray:
        data = self.recent(seconds)
        if data.shape[1] == 0:
            return np.zeros((len(self.eeg_channels), 0))
        return data[self.eeg_channels, :]

    def recent_markers(self, seconds: float):
        """Return (indices, codes) of markers within the recent window."""
        data = self.recent(seconds)
        if data.shape[1] == 0 or self.marker_channel >= data.shape[0]:
            return np.array([], dtype=int), np.array([])
        row = data[self.marker_channel, :]
        idx = np.flatnonzero(row != 0)
        return idx, row[idx]

    # ------------------------------------------------------- signal quality
    def recent_loss(self, seconds: float) -> np.ndarray:
        """BLE-loss flags (1.0 at each dropped packet) for the recent window,
        column-aligned with :meth:`recent`. Use ``flatnonzero`` for drop indices."""
        with self._lock:
            if self._loss_ring is None or self._filled == 0:
                return np.zeros(0, dtype=np.float64)
            length = self._buffer_len
            count = min(self._filled, max(1, int(seconds * self.sampling_rate)))
            end = self._write
            start = (end - count) % length
            if start < end:
                return self._loss_ring[start:end].copy()
            return np.concatenate((self._loss_ring[start:], self._loss_ring[:end]))

    def loss_rate(self, window: float = 10.0) -> float:
        """Dropped packets per second over the most recent ``window`` seconds.

        Divided by the nominal ``window`` (not the data actually buffered) so the
        reading converges up as the window fills instead of spiking to a false
        "poor" in the first second or two of a session.
        """
        return float(self.recent_loss(window).sum()) / max(window, 1e-6)

    def dropped_packets(self) -> int:
        """Total BLE packets lost this session (native driver only; 0 otherwise)."""
        return int(self._dropped_total)

    def recorded_loss_indices(self) -> List[int]:
        """Sample indices (within the current/last recording) where a packet was
        lost. Saved alongside the recording so the gaps can be handled offline."""
        with self._lock:
            return list(self._record_loss_idx)

    def latest_impedance_kohm(self) -> List[float]:
        """Latest impedance (kOhm) per EEG channel, or -1.0 if not measured yet.

        The Ganglion reports impedance on a slow per-channel cycle, so each
        resistance row is mostly zeros between updates. We take the most recent
        *non-zero* sample over a 2 s window -- averaging (as before) diluted the
        value toward ~0. The /1000 assumes BrainFlow returns the resistance in
        Ohms (confirm against a known resistor; see docs).
        """
        if self.acquisition_mode == "impedance":
            latest = {r.channel: r.ohms / 1000.0 for r in self.latest_impedance_readings()}
            return [latest.get(channel, -1.0) for channel in range(5)]
        if self.demo and self._demo_imp is not None:
            return [float(v) for v in self._demo_imp]
        result: List[float] = []
        data = self.recent(2.0)
        for i, _ch in enumerate(self.eeg_channels):
            val = -1.0
            if i < len(self.resistance_channels):
                row = self.resistance_channels[i]
                if data.shape[1] and row < data.shape[0]:
                    nonzero = data[row][data[row] != 0.0]
                    if nonzero.size:
                        val = float(nonzero[-1]) / 1000.0
            result.append(val)
        return result

    # ------------------------------------------------------------- commands
    def _safe_config(self, command: str) -> Optional[str]:
        """Send a config command, swallowing errors (synthetic board, etc.)."""
        if self.board is None:
            return None
        try:
            return self.board.config_board(command)
        except BrainFlowError as exc:
            logger.info("config_board('%s') not applied: %s", command, exc)
            return None

    def set_channel_active(self, index: int, active: bool) -> None:
        if index < 0 or index >= len(self.channel_active):
            return
        self.channel_active[index] = active
        cmd = cfg.CHANNEL_ON[index] if active else cfg.CHANNEL_OFF[index]
        self._safe_config(cmd)

    def start_impedance(self) -> None:
        self.set_acquisition_mode("impedance")
        if not self.streaming and self.board is not None:
            self.start()

    def stop_impedance(self) -> None:
        self.set_acquisition_mode("eeg")

    def set_acquisition_mode(self, mode: str) -> None:
        if mode not in ("eeg", "impedance"):
            raise ValueError("Acquisition mode must be 'eeg' or 'impedance'")
        with self._lifecycle_lock, self._poll_lock:
            if self.impedance_check_active:
                raise RuntimeError("Finish the temporary impedance check before changing mode")
            if mode == self.acquisition_mode:
                return
            if self.recording:
                raise RuntimeError("Stop recording before changing acquisition mode")
            was_streaming = self.streaming
            self.stop()
            self.acquisition_mode = mode
            with self._lock:
                self._filled = 0
                self._write = 0
            if was_streaming:
                self.start()

    def _emit_check_event(self, code: int, timestamp: float) -> None:
        with self._lock:
            self._check_events.append(dict(timestamp=timestamp, code=code, label=CHECK_MARKERS[code]))
            if self.recording:
                self._record_marker_requests.append((timestamp, float(code)))

    def _switch_check_hardware(self, mode: Optional[str]) -> float:
        """Off-GUI mode switch; callers hold lifecycle and poll locks."""
        self._check_last_stop_at = None
        if not self.demo and hasattr(self.board, "switch_acquisition_mode"):
            try:
                timestamp = self.board.switch_acquisition_mode(mode)
            finally:
                self._check_last_stop_at = getattr(self.board, "last_mode_switch_stopped_at", None)
            if mode == "impedance":
                self._drain_check_eeg_tail(self._check_last_stop_at or timestamp)
        else:
            try:
                # After a failed transition the hardware state is uncertain;
                # retry the stop even if our conservative flag is False.
                self.board.stop_stream()
            except BrainFlowError as exc:
                if exc.exit_code != BrainFlowExitCodes.STREAM_THREAD_IS_NOT_RUNNING.value:
                    raise
            else:
                self._check_last_stop_at = time.time()
                if mode == "impedance":
                    self._drain_check_eeg_tail(self._check_last_stop_at)
            self.streaming = False
            if mode is not None:
                if not self.demo:
                    self.board.config_board("z" if mode == "impedance" else "Z")
                self.board.start_stream(450000)
            timestamp = time.time()
        self.streaming = mode is not None
        self.impedance_mode = mode == "impedance"
        if mode == "eeg":
            self._safe_config(cfg.ACCEL_DISABLE)
        return timestamp

    def _drain_check_eeg_tail(self, stopped_at: float) -> None:
        """Keep pre-stop EEG already queued, excluding later transition packets."""
        try:
            data = self.board.get_board_data()
        except Exception as exc:
            # This read is separate from the acknowledged hardware transition.
            # A read failure must not falsely report that impedance never began.
            logger.warning("Could not drain final pre-check EEG data: %s", exc)
            return
        if data.ndim != 2 or not data.shape[1]:
            return
        timestamps = data[self.timestamp_channel]
        keep = np.isfinite(timestamps) & (timestamps <= stopped_at)
        self._accept_eeg_data(data[:, keep], self._loss_flags(data.shape[1])[keep])

    def begin_impedance_check(self, duration_s: Optional[float] = None) -> None:
        """Temporarily suspend EEG without ending its logical recording.

        Call off-GUI. The UI owns the timer and calls end_impedance_check.
        Missing EEG samples are NaN, with independent impedance events/provenance.
        """
        if duration_s is not None and (not np.isfinite(duration_s) or not 0.1 <= duration_s <= 3600):
            raise ValueError("Impedance check duration must be 0.1..3600 seconds")
        with self._lifecycle_lock, self._poll_lock:
            if self.acquisition_mode != "eeg" or self.board is None or not self.streaming:
                raise RuntimeError("An active EEG stream is required for an impedance check")
            if self.impedance_check_active:
                raise RuntimeError("An impedance check is already active")
            self._poll_data()
            now = time.time()
            self._check_phase = "starting"
            self._check_duration_s = duration_s
            self._check_started_at = None
            self._check_gap_started_at = now
            self._check_gap_samples = 0
            self._check_gap_record_start = self._record_count if self.recording else None
            self._check_impedance_started = False
            self._check_impedance_ended = False
            stop_event_emitted = False
            try:
                self._check_started_at = self._switch_check_hardware("impedance")
                self._check_gap_started_at = self._check_last_stop_at or now
                self._check_gap_record_start = self._record_count if self.recording else None
                self._emit_check_event(EEG_STOP, self._check_gap_started_at)
                stop_event_emitted = True
                self._check_impedance_started = True
                self._check_phase = "measuring"
                self._emit_check_event(IMPEDANCE_START, self._check_started_at)
                if not self.demo and not hasattr(self.board, "get_impedance_data"):
                    self._poll_impedance()
                if hasattr(self.board, "discard_pending_markers"):
                    self.board.discard_pending_markers()
                self._demo_imp_next = 0.0
                self._append_check_gap(time.time())
            except Exception:
                if not stop_event_emitted:
                    self._check_gap_started_at = self._check_last_stop_at or now
                    self._check_gap_record_start = self._record_count if self.recording else None
                    self._emit_check_event(EEG_STOP, self._check_gap_started_at)
                self.streaming = False
                self._check_phase = "restore_failed"
                try:
                    self.end_impedance_check()
                except Exception:
                    logger.exception("EEG could not be restored after impedance startup failed")
                raise

    def _append_check_gap(self, until: float) -> None:
        if self._check_gap_started_at is None:
            return
        if self._record_stopped_at is not None and self.recording:
            until = min(until, self._record_stopped_at)
        target = max(0, int(round((until - self._check_gap_started_at) * self.sampling_rate)))
        count = target - self._check_gap_samples
        if count <= 0:
            return
        data = np.zeros((self.num_rows, count), dtype=np.float64)
        data[self.eeg_channels] = np.nan
        data[self.timestamp_channel] = (self._check_gap_started_at
            + np.arange(self._check_gap_samples, target) / self.sampling_rate)
        with self._lock:
            for event in self._check_events:
                offset = int(np.ceil((event["timestamp"] - self._check_gap_started_at)
                                     * self.sampling_rate)) - self._check_gap_samples
                if 0 <= offset < count and data[self.marker_channel, offset] == 0:
                    data[self.marker_channel, offset] = event["code"]
            if self.recording and self._check_gap_record_start is not None:
                self._record_chunks.append(data.copy())
                self._record_count += count
            if self._ring is not None:
                self._append_ring(data)
            self._check_gap_samples = target

    def end_impedance_check(self, resume_eeg: bool = True) -> None:
        """End/retry a check; shutdown uses resume_eeg=False, never a false resume."""
        with self._lifecycle_lock, self._poll_lock:
            if not self.impedance_check_active:
                return
            if self._check_phase == "measuring":
                self._poll_impedance()
            self._check_phase = "restoring"
            now = time.time()
            if self._record_stopped_at is not None and self.recording:
                now = min(now, self._record_stopped_at)
            try:
                # Clear old EEG tail notifications before enabling a new EEG
                # stream; impedance data has its own separate queue.
                if hasattr(self.board, "get_impedance_data"):
                    self.board.get_board_data()
                ended = self._switch_check_hardware("eeg" if resume_eeg else None)
            except Exception:
                if (self._check_last_stop_at is not None and self._check_impedance_started
                        and not self._check_impedance_ended):
                    self._emit_check_event(IMPEDANCE_END, min(now, self._check_last_stop_at)
                                          if self._record_stopped_at is not None else self._check_last_stop_at)
                    self._check_impedance_ended = True
                self.streaming = False
                self.impedance_mode = False
                self._check_phase = "restore_failed"
                self._append_check_gap(time.time())
                raise
            if self._record_stopped_at is not None and self.recording:
                ended = min(ended, self._record_stopped_at)
            if self._check_impedance_started and not self._check_impedance_ended:
                stop_time = self._check_last_stop_at or now
                if self._record_stopped_at is not None and self.recording:
                    stop_time = min(stop_time, self._record_stopped_at)
                self._emit_check_event(IMPEDANCE_END, stop_time)
                self._check_impedance_ended = True
            self._append_check_gap(ended)
            if resume_eeg:
                self._emit_check_event(EEG_RESUME, ended)
            with self._lock:
                if self._check_gap_record_start is not None and self._check_gap_samples:
                    self._record_eeg_gaps.append(dict(
                        start_sample=self._check_gap_record_start,
                        length=self._check_gap_samples, reason="impedance_check",
                        started_at=self._check_gap_started_at, ended_at=ended))
            self._check_phase = "idle"

    def is_disconnected(self) -> bool:
        """True if a native-BLE link dropped mid-session (custom driver only)."""
        return bool(getattr(self.board, "disconnected", False))

    def insert_marker(self, value: float, timestamp: Optional[float] = None) -> None:
        """Insert an event and retain its exact request time while recording."""
        if self.acquisition_mode == "impedance":
            raise RuntimeError("EEG event markers are unavailable in impedance mode")
        event_time = time.time() if timestamp is None else float(timestamp)
        if not np.isfinite(event_time):
            raise ValueError("Marker timestamp must be finite")
        with self._poll_lock:
            if self.board is not None and self.streaming and not self.impedance_check_active:
                try:
                    if hasattr(self.board, "discard_pending_markers"):
                        self.board.insert_marker(float(value), timestamp=event_time)
                    else:
                        self.board.insert_marker(float(value))
                except BrainFlowError as exc:
                    logger.warning("insert_marker failed: %s", exc)
            with self._lock:
                if self.recording:
                    self._record_marker_requests.append((event_time, float(value)))

    def send_raw_command(self, command: str) -> Optional[str]:
        command = command.strip()
        if command == "get_firmware_version":
            return self._safe_config(command)
        if command == "b":
            self.set_acquisition_mode("eeg")
            self.start()
            return ""
        if command == "s":
            self.stop()
            return ""
        if command in ("z", "Z"):
            self.set_acquisition_mode("impedance" if command == "z" else "eeg")
            return ""
        if any(control in command for control in "bzZs"):
            raise ValueError("Send acquisition mode commands separately")
        return self._safe_config(command)

    # ------------------------------------------------------------ recording
    def start_recording(self) -> None:
        with self._poll_lock:
            if self.recording:
                return
            if self.impedance_check_active:
                raise RuntimeError("Finish the impedance check before starting a recording")
            clear = getattr(self.board, "discard_pending_markers", None)
            if clear is not None:
                clear()
            with self._lock:
                self._record_chunks = []
                self._record_count = 0
                self._record_loss_idx = []
                self._record_marker_requests = []
                self._record_marker_events = []
                self._record_impedance = []
                self._record_eeg_gaps = []
                self._record_check_readings = []
                self._record_started_at = time.time()
                self._record_stopped_at = None
                self.recording = True

    def stop_recording(self):
        with self._lifecycle_lock, self._poll_lock:
            with self._lock:
                if not self.recording:
                    return [] if self.acquisition_mode == "impedance" else np.zeros((self.num_rows, 0))
                self._record_stopped_at = time.time()
            closed_check = self.impedance_check_active
            if closed_check:
                try:
                    self.end_impedance_check(resume_eeg=False)
                except Exception:
                    self._record_stopped_at = None
                    raise
            # Capture the final available samples before freezing the recording.
            if not closed_check:
                self._poll_data(force=True)
            with self._lock:
                self.recording = False
                if self.acquisition_mode == "impedance":
                    readings = self._record_impedance
                    self._record_impedance = []
                    return readings
                chunks = self._record_chunks
                self._record_chunks = []
                raw = (np.concatenate(chunks, axis=1) if chunks
                       else np.zeros((self.num_rows, 0)))
                self._align_recorded_markers(raw)
            clear = getattr(self.board, "discard_pending_markers", None)
            if clear is not None:
                clear()
            return raw

    def _align_recorded_markers(self, raw: np.ndarray) -> None:
        """Keep every event, including collisions and events at the stop edge.

        Host arrival timestamps select the first in-range sample at/after an
        event. If no later sample arrived before stop, use the final sample;
        an empty recording keeps the event with ``sample_index=None``.
        """
        n = raw.shape[1]
        requests = self._record_marker_requests
        self._record_marker_events = []
        if not requests:
            # Preserve markers delivered directly by a backend as well.
            if n and 0 <= self.marker_channel < raw.shape[0]:
                for sample in np.flatnonzero(raw[self.marker_channel]):
                    timestamp = (float(raw[self.timestamp_channel, sample])
                                 if 0 <= self.timestamp_channel < raw.shape[0] else 0.0)
                    self._record_marker_events.append(
                        (int(sample), float(raw[self.marker_channel, sample]), timestamp))
            return
        if n:
            if 0 <= self.marker_channel < raw.shape[0]:
                raw[self.marker_channel] = 0.0
            timestamps = (raw[self.timestamp_channel]
                          if 0 <= self.timestamp_channel < raw.shape[0] else np.zeros(n))
            valid = np.flatnonzero(np.isfinite(timestamps) & (timestamps > 0))
            # Arrival clocks normally increase; tolerate a host clock correction
            # without requiring binary search on a non-monotonic array.
            clock = np.maximum.accumulate(timestamps[valid]) if valid.size else None
        for timestamp, code in requests:
            if (self._record_started_at is not None and timestamp < self._record_started_at
                    or self._record_stopped_at is not None and timestamp > self._record_stopped_at):
                continue
            sample = None
            if n:
                if valid.size:
                    position = min(int(np.searchsorted(clock, timestamp)), len(valid) - 1)
                    sample = int(valid[position])
                else:
                    offset = (timestamp - self._record_started_at) * self.sampling_rate
                    sample = min(n - 1, max(0, int(np.ceil(offset))))
                if 0 <= self.marker_channel < raw.shape[0] and raw[self.marker_channel, sample] == 0:
                    raw[self.marker_channel, sample] = code
            self._record_marker_events.append((sample, code, timestamp))

    def recorded_marker_events(self) -> list:
        """Finalized ``(sample_index or None, code, insertion_timestamp)`` events.

        Call after :meth:`stop_recording`. Same-sample collisions remain separate
        entries; the raw marker row can represent only the first one.
        """
        with self._lock:
            return list(self._record_marker_events)

    def recorded_sample_count(self) -> int:
        return self._record_count
