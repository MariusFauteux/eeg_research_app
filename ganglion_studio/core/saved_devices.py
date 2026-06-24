"""Persist a small list of known Ganglion devices.

Lets the user reconnect to a board without scanning every time. Stored as a tiny
JSON file in the user's home config dir (not the project), so it survives across
sessions and checkouts. A "device" is just a friendly name + the BLE address/UUID
the native driver connects with.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

# Module-level so tests can point it at a temp file via monkeypatch.
PATH = os.path.expanduser(os.path.join("~", ".ganglion_studio", "devices.json"))


@dataclass
class SavedDevice:
    name: str
    address: str


def load() -> List[SavedDevice]:
    """Return saved devices (empty list if none / file missing / unreadable)."""
    try:
        with open(PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    out: List[SavedDevice] = []
    for d in data if isinstance(data, list) else []:
        address = str(d.get("address", "")).strip()
        if address:
            out.append(SavedDevice(name=str(d.get("name") or address), address=address))
    return out


def _write(devices: List[SavedDevice]) -> None:
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w", encoding="utf-8") as fh:
        json.dump([{"name": d.name, "address": d.address} for d in devices], fh, indent=2)


def add(name: str, address: str) -> List[SavedDevice]:
    """Save (or update) a device, de-duplicated by address. Returns the new list."""
    address = address.strip()
    if not address:
        return load()
    devices = [d for d in load() if d.address != address]
    devices.append(SavedDevice(name=name.strip() or address, address=address))
    _write(devices)
    return devices


def remove(address: str) -> List[SavedDevice]:
    """Delete the device with this address. Returns the remaining list."""
    devices = [d for d in load() if d.address != address]
    _write(devices)
    return devices
