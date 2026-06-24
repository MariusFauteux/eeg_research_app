"""Regression: channel-type / electrode lists must stay a single source.

board_config.ELECTRODES had drifted from analysis.ELECTRODES and was missing
"PEDOT:PSS", so the live Channel Setup dialog could not offer it even though the
Processing Lab and README could/did.
"""

from ganglion_studio.core import analysis, board_config


def test_lists_are_the_same_objects():
    assert analysis.CHANNEL_TYPES is board_config.CHANNEL_TYPES
    assert analysis.ELECTRODES is board_config.ELECTRODES


def test_pedot_pss_is_available():
    assert "PEDOT:PSS" in board_config.ELECTRODES


def test_expected_members():
    assert set(board_config.CHANNEL_TYPES) == {"EEG", "ECG", "EMG", "MISC"}
    for electrode in ("Ag/AgCl (wet)", "Ag/AgCl (dry)", "PEDOT:PSS", "PEDOT", "Other"):
        assert electrode in board_config.ELECTRODES
