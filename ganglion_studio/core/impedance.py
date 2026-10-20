"""Timestamped Ganglion measurements, independent of the EEG sample matrix."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ImpedanceReading:
    """One fresh measurement; UTC epoch seconds, channels 0..3 and 4 (REF)."""

    timestamp: float
    channel: int
    ohms: float

    def __post_init__(self):
        if not math.isfinite(self.timestamp):
            raise ValueError("Impedance timestamp must be finite")
        if type(self.channel) is not int or not 0 <= self.channel <= 4:
            raise ValueError("Impedance channel must be 0..4")
        if not math.isfinite(self.ohms) or self.ohms < 0:
            raise ValueError("Impedance must be a finite nonnegative resistance")


def impedance_scale(units: str) -> float:
    """Convert transmitted impedance units explicitly; never guess by value."""
    if units not in ("ohms", "kohms"):
        raise ValueError("Impedance units must be 'ohms' or 'kohms'")
    return 1000.0 if units == "kohms" else 1.0
