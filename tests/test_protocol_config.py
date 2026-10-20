"""Tests for the protocol-parameter persistence helper."""

from __future__ import annotations

import json

import pytest

import ganglion_studio.core.protocol_config as pc


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "PATH", str(tmp_path / "protocols.json"))
    assert pc.load() == {}

    pc.save("blinks", {"sets": 4, "blinks_per_set": 6})
    pc.save("eyes", {"conditions": 3})
    data = pc.load()
    assert data["blinks"]["sets"] == 4
    assert data["eyes"]["conditions"] == 3

    # saving the same key again replaces just that entry
    pc.save("blinks", {"sets": 2})
    data = pc.load()
    assert data["blinks"] == {"sets": 2}
    assert data["eyes"] == {"conditions": 3}


def test_load_missing_file_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "PATH", str(tmp_path / "nope.json"))
    assert pc.load() == {}


def test_load_ignores_entries_without_parameter_objects(tmp_path, monkeypatch):
    path = tmp_path / "protocols.json"
    monkeypatch.setattr(pc, "PATH", str(path))
    path.write_text(json.dumps({"blinks": 3, "rest": None, "eyes": {"conditions": 4}}))
    assert pc.load() == {"eyes": {"conditions": 4}}


@pytest.mark.parametrize("failure_at", ["directory", "file"])
def test_save_is_best_effort_when_preferences_unwritable(tmp_path, monkeypatch, failure_at):
    monkeypatch.setattr(pc, "PATH", str(tmp_path / "protocols.json"))

    def denied(*args, **kwargs):
        raise PermissionError("read-only config directory")

    if failure_at == "directory":
        monkeypatch.setattr(pc.os, "makedirs", denied)
    else:
        monkeypatch.setattr(pc, "open", denied, raising=False)
    assert pc.save("blinks", {"sets": 4}) == {"blinks": {"sets": 4}}
