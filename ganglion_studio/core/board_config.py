"""Static configuration and command maps for the OpenBCI Ganglion.

All ASCII commands are taken from the OpenBCI Ganglion SDK and are sent to the
board through BrainFlow's ``BoardShim.config_board``.
"""

from __future__ import annotations

# --- Ganglion ASCII command protocol (OpenBCI Ganglion SDK) ---------------
# Turn the respective channel [1-4] ON  -> the channel streams ADC values.
CHANNEL_ON = {0: "!", 1: "@", 2: "#", 3: "$"}
# Turn the respective channel [1-4] OFF -> the channel reads 0.00.
CHANNEL_OFF = {0: "1", 1: "2", 2: "3", 3: "4"}

# LeadOff impedance test.
IMPEDANCE_START = "z"
IMPEDANCE_STOP = "Z"

# Disable the accelerometer. With accel OFF the Ganglion streams EEG as 19-bit
# deltas; with it ON the EEG drops to 18-bit to make room for 3-axis accel data.
# This app keeps the accelerometer off to maximize EEG resolution, so only the
# disable command is ever sent (once, on stream start).
ACCEL_DISABLE = "N"

# Streaming.
STREAM_START = "b"
STREAM_STOP = "s"

# Default electrode labels for the 4 Ganglion channels (10-20 friendly).
DEFAULT_CHANNEL_NAMES = ["Ch1", "Ch2", "Ch3", "Ch4"]

# Channel signal types and electrode materials used for setup/analysis.
# Single source of truth: core.analysis re-exports these so the live Channel
# Setup dialog and the Processing Lab always offer identical options.
CHANNEL_TYPES = ["EEG", "ECG", "EMG", "MISC"]
ELECTRODES = ["Ag/AgCl (wet)", "Ag/AgCl (dry)", "PEDOT:PSS", "PEDOT", "Other"]

# Standard 10-20 electrode placements (plus references / unset).
TEN_TWENTY = [
    "None",
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
    "A1", "A2", "M1", "M2", "Custom",
]

# A small palette used consistently across the time series / PSD / panels so a
# channel keeps the same colour everywhere.
CHANNEL_COLORS = [
    "#4f8ef7",  # blue
    "#f7766f",  # red
    "#5fd38d",  # green
    "#e2c044",  # yellow
]

# Impedance quality thresholds (kOhm). Below "good" is excellent contact.
IMPEDANCE_GOOD_KOHM = 10.0
IMPEDANCE_OK_KOHM = 50.0


def impedance_color(kohm: float) -> str:
    """Return a colour string for an impedance value (kOhm)."""
    if kohm < 0:
        return "#888888"
    if kohm <= IMPEDANCE_GOOD_KOHM:
        return "#5fd38d"  # green - good
    if kohm <= IMPEDANCE_OK_KOHM:
        return "#e2c044"  # yellow - ok
    return "#f7766f"  # red - bad
