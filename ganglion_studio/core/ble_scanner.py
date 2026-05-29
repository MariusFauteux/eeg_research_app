"""Native Bluetooth discovery of OpenBCI Ganglion boards using ``bleak``.

Returns lightweight dictionaries so the UI layer never needs to import bleak.
``bleak`` is optional at import time: if it is unavailable (or BLE is not
supported on the machine) the helpers raise :class:`BleUnavailable` which the
dashboard turns into a friendly message + Demo-mode suggestion.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List


class BleUnavailable(RuntimeError):
    """Raised when BLE scanning cannot be performed on this machine."""


@dataclass
class BleDevice:
    name: str
    address: str
    rssi: int

    @property
    def is_ganglion(self) -> bool:
        return "ganglion" in (self.name or "").lower()


async def _scan(timeout: float) -> List[BleDevice]:
    try:
        from bleak import BleakScanner
    except Exception as exc:  # pragma: no cover - import guard
        raise BleUnavailable(f"bleak is not installed: {exc}") from exc

    try:
        discovered = await BleakScanner.discover(timeout=timeout, return_adv=True)
    except Exception as exc:  # pragma: no cover - platform/adapter errors
        raise BleUnavailable(f"BLE scan failed: {exc}") from exc

    devices: List[BleDevice] = []
    for _addr, (dev, adv) in discovered.items():
        name = dev.name or (adv.local_name if adv else None) or "Unknown"
        rssi = getattr(adv, "rssi", 0) if adv else 0
        devices.append(BleDevice(name=name, address=dev.address, rssi=int(rssi or 0)))
    return devices


def scan(timeout: float = 8.0, ganglion_only: bool = False) -> List[BleDevice]:
    """Synchronously scan for BLE devices. Intended to run in a worker thread."""
    try:
        devices = asyncio.run(_scan(timeout))
    except BleUnavailable:
        raise
    except RuntimeError as exc:
        # e.g. "asyncio.run() cannot be called from a running event loop"
        raise BleUnavailable(str(exc)) from exc

    devices.sort(key=lambda d: (not d.is_ganglion, -d.rssi))
    if ganglion_only:
        devices = [d for d in devices if d.is_ganglion]
    return devices
