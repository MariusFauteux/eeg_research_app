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
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from brainflow.board_shim import (
    BoardIds,
    BoardShim,
    BrainFlowInputParams,
    BrainFlowError,
)

from . import board_config as cfg

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
    firmware: str = "3"  # "3" (default), "2", or "auto"
    buffer_seconds: float = 30.0

    # --- runtime state (not constructor args) ---
    board: Optional[BoardShim] = field(default=None, init=False)
    board_id: int = field(default=0, init=False)
    sampling_rate: int = field(default=200, init=False)
    num_rows: int = field(default=0, init=False)
    eeg_channels: List[int] = field(default_factory=list, init=False)
    resistance_channels: List[int] = field(default_factory=list, init=False)
    marker_channel: int = field(default=0, init=False)
    timestamp_channel: int = field(default=0, init=False)

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

    # demo impedance simulation state
    _demo_imp: Optional[np.ndarray] = field(default=None, init=False)

    # Serializes the acquisition thread (producer) against GUI-thread readers.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # Background acquisition thread.
    acquisition_hz: float = 60.0
    _acq_thread: Optional[threading.Thread] = field(default=None, init=False)
    _acq_stop: threading.Event = field(default_factory=threading.Event, init=False)

    # ------------------------------------------------------------------ setup
    def _resolve_board_id(self) -> int:
        if self.demo:
            return BoardIds.SYNTHETIC_BOARD.value
        return BoardIds.GANGLION_NATIVE_BOARD.value

    def _build_params(self) -> BrainFlowInputParams:
        params = BrainFlowInputParams()
        if not self.demo:
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

    def prepare(self) -> None:
        """Create the session and prepare the board. May block while BLE connects."""
        BoardShim.disable_board_logger()
        self.board_id = self._resolve_board_id()
        self._load_descriptor()
        params = self._build_params()
        self.board = BoardShim(self.board_id, params)
        self.board.prepare_session()

        self._buffer_len = max(1, int(self.buffer_seconds * self.sampling_rate))
        self._ring = np.zeros((self.num_rows, self._buffer_len), dtype=np.float64)
        self._filled = 0
        self._write = 0
        self._demo_imp = np.random.uniform(3.0, 25.0, size=len(self.eeg_channels))
        logger.info("Board prepared: id=%s sr=%s", self.board_id, self.sampling_rate)

    def start(self) -> None:
        if self.board is None:
            raise RuntimeError("Board not prepared")
        self.board.start_stream(450000)
        self.streaming = True
        # Disable the accelerometer so the Ganglion streams 19-bit EEG deltas
        # (with accel on it drops to 18-bit). We trade motion data for resolution.
        self._safe_config(cfg.ACCEL_DISABLE)

    def stop(self) -> None:
        if self.board is not None and self.streaming:
            try:
                self.board.stop_stream()
            except BrainFlowError as exc:  # pragma: no cover - hardware path
                logger.warning("stop_stream failed: %s", exc)
        self.streaming = False

    def release(self) -> None:
        self.stop_acquisition()
        self.stop()
        if self.board is not None:
            try:
                self.board.release_session()
            except BrainFlowError as exc:  # pragma: no cover
                logger.warning("release_session failed: %s", exc)
        self.board = None

    # ----------------------------------------------------------- acquisition
    def start_acquisition(self) -> None:
        """Run :meth:`poll` on a dedicated thread, decoupled from rendering."""
        if self._acq_thread is not None and self._acq_thread.is_alive():
            return
        self._acq_stop.clear()
        self._acq_thread = threading.Thread(
            target=self._acq_loop, name="ganglion-acquisition", daemon=True
        )
        self._acq_thread.start()

    def stop_acquisition(self) -> None:
        self._acq_stop.set()
        thread = self._acq_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._acq_thread = None

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
        if self.board is None or not self.streaming:
            return 0
        try:
            data = self.board.get_board_data()
        except BrainFlowError as exc:  # pragma: no cover
            logger.warning("get_board_data failed: %s", exc)
            return 0
        n = data.shape[1] if data.ndim == 2 else 0
        if n == 0:
            return 0

        if self.demo:
            data = self._inject_demo_resistance(data)

        with self._lock:
            if self.recording:
                self._record_chunks.append(data.copy())
                self._record_count += n
            self._append_ring(data)
        return n

    def _append_ring(self, data: np.ndarray) -> None:
        """Append new columns to the circular ring buffer (O(n), no shifting)."""
        assert self._ring is not None
        length = self._buffer_len
        rows = min(data.shape[0], self._ring.shape[0])
        n = data.shape[1]
        if n <= 0:
            return
        if n >= length:
            # New chunk alone overfills the buffer: keep only its last `length`.
            self._ring[:rows, :] = data[:rows, -length:]
            self._write = 0
            self._filled = length
            return
        end = self._write + n
        if end <= length:
            self._ring[:rows, self._write:end] = data[:rows, :]
        else:  # chunk wraps past the buffer end
            first = length - self._write
            self._ring[:rows, self._write:] = data[:rows, :first]
            self._ring[:rows, : n - first] = data[:rows, first:]
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

    def latest_impedance_kohm(self) -> List[float]:
        """Return latest impedance (kOhm) per EEG channel, or -1 if unknown."""
        if self.demo and self._demo_imp is not None:
            return [float(v) for v in self._demo_imp]
        result: List[float] = []
        data = self.recent(0.5)
        for i, _ch in enumerate(self.eeg_channels):
            if i < len(self.resistance_channels):
                row = self.resistance_channels[i]
                if data.shape[1] and row < data.shape[0]:
                    val = float(np.mean(data[row, -min(data.shape[1], 25):]))
                    result.append(val / 1000.0)
                    continue
            result.append(-1.0)
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
        self.impedance_mode = True
        self._safe_config(cfg.IMPEDANCE_START)

    def stop_impedance(self) -> None:
        self.impedance_mode = False
        self._safe_config(cfg.IMPEDANCE_STOP)

    def insert_marker(self, value: float) -> None:
        if self.board is not None and self.streaming:
            try:
                self.board.insert_marker(float(value))
            except BrainFlowError as exc:
                logger.warning("insert_marker failed: %s", exc)

    def send_raw_command(self, command: str) -> Optional[str]:
        return self._safe_config(command)

    # ------------------------------------------------------------ recording
    def start_recording(self) -> None:
        with self._lock:
            self._record_chunks = []
            self._record_count = 0
            self.recording = True

    def stop_recording(self) -> np.ndarray:
        with self._lock:
            self.recording = False
            chunks = self._record_chunks
            self._record_chunks = []
        if not chunks:
            return np.zeros((self.num_rows, 0))
        return np.concatenate(chunks, axis=1)

    def recorded_sample_count(self) -> int:
        return self._record_count
