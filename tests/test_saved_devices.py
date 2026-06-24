"""Tests for the saved-devices persistence helper."""

from __future__ import annotations

import ganglion_studio.core.saved_devices as sd


def test_add_update_remove(tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "PATH", str(tmp_path / "devices.json"))
    assert sd.load() == []

    sd.add("My Ganglion", "UUID-1")
    sd.add("Lab board", "UUID-2")
    assert [d.address for d in sd.load()] == ["UUID-1", "UUID-2"]

    # de-dupe by address: re-adding UUID-1 updates the name, no duplicate
    sd.add("Renamed", "UUID-1")
    devs = sd.load()
    assert [d.address for d in devs].count("UUID-1") == 1
    assert any(d.name == "Renamed" and d.address == "UUID-1" for d in devs)

    sd.remove("UUID-2")
    assert [d.address for d in sd.load()] == ["UUID-1"]


def test_blank_address_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "PATH", str(tmp_path / "devices.json"))
    sd.add("no address", "   ")
    assert sd.load() == []


def test_load_missing_file_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "PATH", str(tmp_path / "nope.json"))
    assert sd.load() == []
