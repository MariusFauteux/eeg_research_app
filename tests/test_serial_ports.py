"""serial_ports.list_serial_ports must work, and degrade gracefully to [] when
pyserial is not installed (the dongle dependency is optional)."""

import builtins

from ganglion_studio.core import serial_ports


def test_returns_list_of_tuples():
    out = serial_ports.list_serial_ports()
    assert isinstance(out, list)
    for item in out:
        assert isinstance(item, tuple) and len(item) == 2


def test_graceful_without_pyserial(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("serial"):
            raise ImportError("pyserial not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert serial_ports.list_serial_ports() == []
