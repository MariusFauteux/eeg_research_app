"""Regression: the Ganglion advertises as 'Simblee', so the scan filter must
recognize both names (otherwise it only shows up under 'all devices')."""

from ganglion_studio.core.ble_scanner import BleDevice


def test_simblee_and_ganglion_names_match():
    assert BleDevice("Simblee", "AA", -50).is_ganglion
    assert BleDevice("SIMBLEE 0x42", "AA", -50).is_ganglion
    assert BleDevice("Ganglion-1234", "BB", -50).is_ganglion


def test_unrelated_names_do_not_match():
    assert not BleDevice("AirPods", "CC", -50).is_ganglion
    assert not BleDevice("", "DD", -50).is_ganglion
    assert not BleDevice(None, "EE", -50).is_ganglion  # name may be None
