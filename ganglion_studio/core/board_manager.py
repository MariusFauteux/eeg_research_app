"""BrainFlow board wrapper for the OpenBCI Ganglion (native BLE) + demo mode.

The :class:`BoardManager` owns the BrainFlow ``BoardShim`` and a single ring
buffer that always holds the most recent ``buffer_seconds`` of *raw* data for
every BrainFlow row (EEG, accel, resistance, marker, timestamp, ...).

A single producer (the :meth:`poll` method, driven by a Qt timer) pulls new
samples with ``get_board_data`` (which removes them from BrainFlow's internal
buffer) and appends them to the ring buffer and, when recording, to a record
buffer. All visualization widgets are pure consumers that read recent slices.
"""

from __future__ import annotations

import logging
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
    accel_channels: List[int] = field(default_factory=list, init=False)
    resistance_channels: List[int] = field(default_factory=list, init=False)
    marker_channel: int = field(default=0, init=False)
    timestamp_channel: int = field(default=0, init=False)

    channel_active: List[bool] = field(default_factory=list, init=False)
    streaming: bool = field(default=False, init=False)
    impedance_mode: bool = field(default=False, init=False)
    accel_enabled: bool = field(default=True, init=False)

    _ring: Optional[np.ndarray] = field(default=None, init=False)
    _filled: int = field(default=0, init=False)
    _buffer_len: int = field(default=0, init=False)

    recording: bool = field(default=False, init=False)
    _record_chunks: List[np.ndarray] = field(default_factory=list, init=False)

    # demo impedance simulation state
    _demo_imp: Optional[np.ndarray] = field(default=None, init=False)

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
        self.accel_channels = list(descr.get("accel_channels", []))
        self.resistance_channels = list(descr.get("resistance_channels", []))
        self.marker_channel = int(descr.get("marker_channel", 0))
        self.timestamp_channel = int(descr.get("timestamp_channel", 0))
        self.channel_active = [True] * len(self.eeg_channels)

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
        self._demo_imp = np.random.uniform(3.0, 25.0, size=len(self.eeg_channels))
        logger.info("Board prepared: id=%s sr=%s", self.board_id, self.sampling_rate)

    def start(self) -> None:
        if self.board is None:
            raise RuntimeError("Board not prepared")
        self.board.start_stream(450000)
        self.streaming = True
        # Ganglion ships with accel on by delta-compression; make explicit.
        if self.accel_enabled:
            self._safe_config(cfg.ACCEL_START)

    def stop(self) -> None:
        if self.board is not None and self.streaming:
            try:
                self.board.stop_stream()
            except BrainFlowError as exc:  # pragma: no cover - hardware path
                logger.warning("stop_stream failed: %s", exc)
        self.streaming = False

    def release(self) -> None:
        self.stop()
        if self.board is not None:
            try:
                self.board.release_session()
            except BrainFlowError as exc:  # pragma: no cover
                logger.warning("release_session failed: %s", exc)
        self.board = None

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

        if self.recording:
            self._record_chunks.append(data.copy())

        self._append_ring(data)
        return n

    def _append_ring(self, data: np.ndarray) -> None:
        assert self._ring is not None
        n = data.shape[1]
        if n >= self._buffer_len:
            self._ring[:, :] = data[:, -self._buffer_len:]
            self._filled = self._buffer_len
        else:
            self._ring[:, :-n] = self._ring[:, n:]
            self._ring[:, -n:] = data[: self._ring.shape[0], :]
            self._filled = min(self._buffer_len, self._filled + n)

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
        if self._ring is None or self._filled == 0:
            return np.zeros((self.num_rows, 0))
        n = min(self._filled, max(1, int(seconds * self.sampling_rate)))
        return self._ring[:, -n:].copy()

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

    def set_accel(self, enabled: bool) -> None:
        self.accel_enabled = enabled
        self._safe_config(cfg.ACCEL_START if enabled else cfg.ACCEL_STOP)

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
        self._record_chunks = []
        self.recording = True

    def stop_recording(self) -> np.ndarray:
        self.recording = False
        if not self._record_chunks:
            return np.zeros((self.num_rows, 0))
        return np.concatenate(self._record_chunks, axis=1)

    def recorded_sample_count(self) -> int:
        return int(sum(c.shape[1] for c in self._record_chunks))
