"""Native Bluetooth discovery of OpenBCI Ganglion boards using ``bleak``.

Scanning is run in an **isolated subprocess** by default. This is deliberate:

* On macOS the CoreBluetooth backend must run on a process's *main* thread; doing
  it on a Qt worker thread crashes the app. The subprocess's main thread
  satisfies this requirement.
* A native crash/abort in the BLE stack (segfault, CoreBluetooth assertion,
  BlueZ/D-Bus error) cannot be caught by a Python ``try/except`` - if it happened
  in-process it would take the whole GUI down. In a subprocess it just yields a
  non-zero exit code that we surface as :class:`BleUnavailable`.

An in-process fallback is kept for environments where the helper module cannot be
spawned (e.g. frozen builds). The UI layer never imports bleak directly.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
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


# --------------------------------------------------------------------------- #
# In-process scan (used inside the subprocess, or as a fallback)
# --------------------------------------------------------------------------- #
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
        name = dev.name or (getattr(adv, "local_name", None) if adv else None) or "Unknown"
        rssi = getattr(adv, "rssi", 0) if adv else 0
        devices.append(BleDevice(name=name, address=dev.address, rssi=int(rssi or 0)))
    return devices


def _post(devices: List[BleDevice], ganglion_only: bool) -> List[BleDevice]:
    devices.sort(key=lambda d: (not d.is_ganglion, -d.rssi))
    if ganglion_only:
        devices = [d for d in devices if d.is_ganglion]
    return devices


def _inprocess_scan(timeout: float, ganglion_only: bool) -> List[BleDevice]:
    try:
        devices = asyncio.run(_scan(timeout))
    except BleUnavailable:
        raise
    except RuntimeError as exc:
        # e.g. "asyncio.run() cannot be called from a running event loop"
        raise BleUnavailable(str(exc)) from exc
    return _post(devices, ganglion_only)


# --------------------------------------------------------------------------- #
# Subprocess-isolated scan
# --------------------------------------------------------------------------- #
def _subprocess_scan(timeout: float, ganglion_only: bool) -> List[BleDevice]:
    # Make the package importable in the child regardless of the caller's cwd.
    pkg_parent = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_parent + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "ganglion_studio.core.ble_scanner",
           "--timeout", str(timeout)]
    if ganglion_only:
        cmd.append("--ganglion-only")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=timeout + 25,
        )
    except subprocess.TimeoutExpired as exc:
        raise BleUnavailable("BLE scan timed out") from exc
    except Exception as exc:
        raise BleUnavailable(f"could not start BLE scan helper: {exc}") from exc

    if proc.returncode != 0:
        msg = (proc.stderr or "").strip() or f"scan helper exited with code {proc.returncode}"
        raise BleUnavailable(msg)

    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise BleUnavailable(f"invalid scan output: {exc}") from exc
    return [BleDevice(name=d["name"], address=d["address"], rssi=int(d["rssi"]))
            for d in payload]


def scan(timeout: float = 8.0, ganglion_only: bool = False) -> List[BleDevice]:
    """Scan for BLE devices. Safe to call from a Qt worker thread.

    Runs in an isolated subprocess; falls back to in-process scanning only if the
    helper module cannot be spawned.
    """
    try:
        return _subprocess_scan(timeout, ganglion_only)
    except BleUnavailable as exc:
        msg = str(exc).lower()
        if "no module named" in msg or "could not start" in msg:
            return _inprocess_scan(timeout, ganglion_only)
        raise


# --------------------------------------------------------------------------- #
# CLI entry point (run inside the subprocess)
# --------------------------------------------------------------------------- #
def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ganglion BLE scan helper")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--ganglion-only", action="store_true")
    args = parser.parse_args()

    try:
        devices = _inprocess_scan(args.timeout, args.ganglion_only)
    except BleUnavailable as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"unexpected scan error: {exc}", file=sys.stderr)
        sys.exit(3)

    print(json.dumps([
        {"name": d.name, "address": d.address, "rssi": d.rssi} for d in devices
    ]))


if __name__ == "__main__":
    _cli()
