"""Enumerate serial ports for the OpenBCI BLED112 Bluetooth dongle.

Kept separate so the optional ``pyserial`` dependency is only needed for the
dongle connection mode -- if it is missing the app still runs and native
Bluetooth is unaffected (the list just comes back empty).
"""

from __future__ import annotations

from typing import List, Tuple


def list_serial_ports() -> List[Tuple[str, str]]:
    """Return ``[(device, description), ...]`` for attached serial ports.

    Empty list if ``pyserial`` is not installed. The OpenBCI BLED112 dongle is a
    Silicon Labs CP210x USB-serial device (e.g. ``/dev/cu.usbserial-*`` on macOS,
    ``/dev/ttyUSB*`` on Linux, ``COMx`` on Windows) -- NOT a generic Bluetooth
    adapter. The description includes manufacturer + USB VID:PID so a real dongle
    is easy to tell apart from macOS system stubs, and real USB devices (those
    with a vendor id) are listed first.
    """
    try:
        from serial.tools import list_ports
    except Exception:
        return []
    ports = list(list_ports.comports())
    ports.sort(key=lambda p: (getattr(p, "vid", None) is None, p.device))
    out: List[Tuple[str, str]] = []
    for p in ports:
        bits = [p.description or p.device]
        if getattr(p, "manufacturer", None):
            bits.append(p.manufacturer)
        if getattr(p, "vid", None) is not None and getattr(p, "pid", None) is not None:
            bits.append(f"{p.vid:04x}:{p.pid:04x}")
        out.append((p.device, " - ".join(bits)))
    return out
