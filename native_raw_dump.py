"""Dump RAW Ganglion BLE notification bytes (no decoding) for protocol debugging.

The custom decoder is producing railed/flat output, which means the assumed
packet format is wrong. This tool connects with bleak, starts the stream, and
records the raw 20-byte notifications exactly as they arrive -- so we can inspect
the real on-wire format and fix the decoder against ground truth.

Runs bleak on the process main thread (CLI), which is CoreBluetooth-legal on
macOS -- no subprocess needed.

Usage
-----
    .venv/bin/python native_raw_dump.py <ble-address-or-uuid> [--seconds 6] [--accel-off]

It prints the byte[0] (packet-id) histogram and the first 40 packets in hex, and
saves every captured packet to native_raw_dump.txt (one hex packet per line).
Paste the printed output back.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter

NOTIFY_UUID = "2d30c082-f39f-4ce6-923f-3484ea480596"
WRITE_UUID = "2d30c083-f39f-4ce6-923f-3484ea480596"


async def run(address: str, seconds: float, accel_off: bool) -> None:
    from bleak import BleakClient

    packets: list[bytes] = []

    def on_notify(_handle, data: bytearray) -> None:
        packets.append(bytes(data))

    print(f"Connecting to {address} ...")
    async with BleakClient(address, timeout=20.0) as client:
        print("Connected. Subscribing + starting stream ('b')...")
        await client.start_notify(NOTIFY_UUID, on_notify)
        await client.write_gatt_char(WRITE_UUID, b"b", response=False)
        if accel_off:
            await asyncio.sleep(0.2)
            await client.write_gatt_char(WRITE_UUID, b"N", response=False)
        await asyncio.sleep(seconds)
        await client.write_gatt_char(WRITE_UUID, b"s", response=False)
        await client.stop_notify(NOTIFY_UUID)

    print(f"\nCaptured {len(packets)} notifications.")
    if not packets:
        print("Nothing arrived -- check the address and that the board is on.")
        return

    lengths = Counter(len(p) for p in packets)
    print(f"packet lengths: {dict(lengths)}")
    ids = Counter(p[0] for p in packets if p)
    lo, hi = min(ids), max(ids)
    print(f"byte[0] (packet id) range: {lo}..{hi}, {len(ids)} distinct values")
    print(f"byte[0] histogram (id:count, sorted): "
          f"{dict(sorted(ids.items()))}")

    print("\nfirst 40 packets (hex):")
    for i, p in enumerate(packets[:40]):
        print(f"  {i:3d} id={p[0]:3d} len={len(p):2d}  {p.hex()}")

    with open("native_raw_dump.txt", "w", encoding="utf-8") as fh:
        for p in packets:
            fh.write(p.hex() + "\n")
    print("\nSaved all packets to native_raw_dump.txt")


def main() -> None:
    ap = argparse.ArgumentParser(description="Raw Ganglion BLE packet dumper")
    ap.add_argument("address", help="BLE address/UUID (from the app's Scan list)")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--accel-off", action="store_true",
                    help="also send 'N' (accel off -> 19-bit) after start")
    args = ap.parse_args()
    asyncio.run(run(args.address, args.seconds, args.accel_off))


if __name__ == "__main__":
    main()
