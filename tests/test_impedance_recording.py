"""Impedance persistence retains real update timing, units, and every reading."""

import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ganglion_studio.core import impedance_recording as store
from ganglion_studio.core.impedance import ImpedanceReading
from ganglion_studio.core.session import SessionConfig


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(store, "datetime", FixedDatetime)
    recorder = store.ImpedanceRecorder(SessionConfig(name="electrode test"), str(tmp_path))
    recorder.begin()
    return recorder


def test_round_trip_preserves_precision_names_reference_and_repeated_updates(recorder):
    start = recorder.started_at.timestamp()
    names = ["Cz, central", "Électrode 2", "O1", "O2", "Reference"]
    readings = [
        ImpedanceReading(start + 2.1234567, 0, 2500.125),
        ImpedanceReading(start + 1.1234567, 4, 0.0),
        ImpedanceReading(start + 2.1234567, 1, 140000.0),
        ImpedanceReading(start + 2.1234567, 0, 2500.125),
    ]

    paths = recorder.save(readings, names, metadata={"operator": "Marius", "units": "extra only"})
    loaded = store.load_impedance_recording(paths[0])

    assert loaded.readings == sorted(readings, key=lambda reading: reading.timestamp)
    assert loaded.channel_names == names
    assert loaded.started_at == start
    assert loaded.duration_s == pytest.approx(readings[0].timestamp - start)
    assert loaded.metadata["units"] == "ohms"
    assert loaded.metadata["payload_units"] == recorder.config.impedance_units
    assert loaded.metadata["extra"]["operator"] == "Marius"
    assert loaded.metadata["n_readings"] == 4
    assert loaded.metadata["elapsed_origin"] == "recording_start"
    assert "sampling_rate" not in loaded.metadata
    assert datetime.fromisoformat(loaded.metadata["started_at_utc"]).timestamp() == start
    assert datetime.fromisoformat(loaded.metadata["started_at_local"]).timestamp() == start
    with open(paths[0], newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert float(rows[1]["impedance_kohm"]) == 2.500125
    assert float(rows[0]["elapsed_s"]) == readings[1].timestamp - start
    assert float(rows[1]["timestamp"]) == readings[0].timestamp


def test_metadata_reports_observed_per_channel_cadence(recorder):
    start = recorder.started_at.timestamp()
    readings = [ImpedanceReading(start + second, 0, 1000.0) for second in (1, 2, 4)]
    readings.append(ImpedanceReading(start + 1.5, 4, 3000.0))

    _, meta_path = recorder.save(readings, ["A", "B", "C", "D"])
    meta = json.loads(Path(meta_path).read_text())

    ch0 = meta["cadence"]["per_channel"]["0"]
    assert ch0["n_readings"] == 3
    assert ch0["min_interval_s"] == 1.0
    assert ch0["max_interval_s"] == 2.0
    assert ch0["mean_interval_s"] == 1.5
    assert ch0["observed_update_rate_hz"] == pytest.approx(2 / 3)
    assert meta["cadence"]["per_channel"]["4"]["observed_update_rate_hz"] is None


def test_empty_recording_writes_reopenable_header_and_metadata(recorder):
    paths = recorder.save([], ["A", "B", "C", "D"])

    loaded = store.load_impedance_recording(paths[0])

    assert loaded.readings == []
    assert loaded.channel_names == ["A", "B", "C", "D", "REF"]
    assert loaded.started_at == recorder.started_at.timestamp()
    assert loaded.duration_s == 0
    assert loaded.metadata["n_readings"] == 0
    with open(paths[0], newline="", encoding="utf-8") as fh:
        assert list(csv.reader(fh)) == [list(store.CSV_COLUMNS)]


@pytest.mark.parametrize("timestamp,channel,ohms", [
    (math.nan, 0, 1000), (math.inf, 0, 1000), (1, -1, 1000),
    (1, 5, 1000), (1, 1.5, 1000), (1, True, 1000),
    (1, 0, math.nan), (1, 0, math.inf), (1, 0, -1),
])
def test_invalid_measurements_raise_without_publishing_files(recorder, timestamp, channel, ohms):
    invalid = SimpleNamespace(timestamp=timestamp, channel=channel, ohms=ohms)

    with pytest.raises(store.ImpedanceRecordingError):
        recorder.save([invalid], ["A", "B", "C", "D"])

    assert list(Path(recorder.out_dir).iterdir()) == []


def test_nonfinite_extra_metadata_raises_without_publishing_files(recorder):
    with pytest.raises(store.ImpedanceRecordingError, match="valid JSON"):
        recorder.save([], [], metadata={"invalid": math.nan})

    assert list(Path(recorder.out_dir).iterdir()) == []


def test_unique_atomic_directories_prevent_same_second_collisions(recorder):
    def begin(_):
        another = store.ImpedanceRecorder(recorder.config, recorder.base_dir)
        return another.begin()

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(begin, range(8)))

    assert len(set(paths)) == 8
    assert recorder.out_dir not in paths
    assert all("_impedance" in Path(path).name for path in paths)


def test_failed_publication_retries_same_paths_and_spares_earlier_recording(recorder, monkeypatch):
    start = recorder.started_at.timestamp()
    prior_paths = recorder.save([ImpedanceReading(start, 0, 1000)], ["A"])
    prior_bytes = [Path(path).read_bytes() for path in prior_paths]
    another = store.ImpedanceRecorder(recorder.config, recorder.base_dir)
    another.begin()
    original_replace = store.os.replace
    replacements = []

    def fail_second(source, target):
        replacements.append(target)
        if len(replacements) == 2:
            raise OSError("disk full")
        original_replace(source, target)

    reading = ImpedanceReading(start + 2, 4, 2500)
    monkeypatch.setattr(store.os, "replace", fail_second)
    with pytest.raises(store.ImpedanceRecordingError, match="disk full"):
        another.save([reading], ["A"])
    monkeypatch.setattr(store.os, "replace", original_replace)

    paths = another.save([reading], ["A"])

    assert paths == replacements
    assert store.load_impedance_recording(paths[0]).readings == [reading]
    assert [Path(path).read_bytes() for path in prior_paths] == prior_bytes
    assert not list(Path(another.out_dir).glob(".impedance-*"))


def test_successful_snapshot_is_idempotent_and_cannot_be_overwritten(recorder):
    reading = ImpedanceReading(recorder.started_at.timestamp(), 0, 1000)
    paths = recorder.save([reading], ["A"])
    before = [Path(path).read_bytes() for path in paths]

    assert recorder.save([reading], ["A"]) == paths
    with pytest.raises(store.ImpedanceRecordingError, match="already saved"):
        recorder.save([reading, reading], ["A"])

    assert [Path(path).read_bytes() for path in paths] == before
    recorder.begin()
    assert recorder.save([reading, reading], ["A"]) != paths


def test_csv_can_be_reopened_without_optional_metadata(recorder):
    reading = ImpedanceReading(recorder.started_at.timestamp() + 1.125, 4, 1234.5)
    csv_path, meta_path = recorder.save([reading], ["A", "B", "C", "D", "Reference"])
    Path(meta_path).unlink()

    loaded = store.load_impedance_recording(csv_path)

    assert loaded.readings == [reading]
    assert loaded.started_at == recorder.started_at.timestamp()
    assert loaded.channel_names[4] == "Reference"
    assert loaded.metadata == {}


def test_modified_csv_is_rejected_when_metadata_checksum_exists(recorder):
    csv_path, _ = recorder.save([ImpedanceReading(recorder.started_at.timestamp(), 0, 1000)], ["A"])
    path = Path(csv_path)
    path.write_text(path.read_text().replace("1000.0", "2000.0"))

    with pytest.raises(store.ImpedanceRecordingError, match="checksum"):
        store.load_impedance_recording(csv_path)


@pytest.mark.parametrize("change,error", [
    ({"timestamp": "nan"}, "finite"),
    ({"channel_index": "5"}, "channel index"),
    ({"channel_index": "0.5"}, "channel index"),
    ({"impedance_ohm": "-1"}, "negative"),
    ({"impedance_kohm": "2"}, "ohm/kOhm"),
    ({"channel_name": ""}, "channel name"),
])
def test_corrupt_standalone_csv_has_clear_errors(recorder, change, error):
    csv_path, meta_path = recorder.save([ImpedanceReading(recorder.started_at.timestamp(), 0, 1000)], ["A"])
    Path(meta_path).unlink()
    with open(csv_path, newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    row.update(change)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=store.CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(store.ImpedanceRecordingError, match=error):
        store.load_impedance_recording(csv_path)


def test_corrupt_metadata_and_missing_files_have_clear_errors(recorder):
    csv_path, meta_path = recorder.save([], [])
    Path(meta_path).write_text("{broken")

    with pytest.raises(store.ImpedanceRecordingError, match="Could not read"):
        store.load_impedance_recording(csv_path)
    with pytest.raises(store.ImpedanceRecordingError, match="Could not read"):
        store.load_impedance_recording(str(Path(recorder.out_dir) / "missing.csv"))
