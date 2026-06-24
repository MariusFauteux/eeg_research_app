"""Regression: colours live in one place (ganglion_studio.palette).

Guards against the old "colour sprawl" where the same hex was duplicated across
many widgets and could drift apart.
"""

from ganglion_studio import palette
from ganglion_studio.core import analysis, board_config


def test_palette_exposes_expected_names():
    for name in ("BG", "FG", "ACCENT", "GOOD", "OK", "BAD", "MUTED",
                 "CHANNEL_COLORS", "BAND_COLORS", "ELECTRODE_COLORS", "MARKER"):
        assert hasattr(palette, name), f"palette is missing {name}"


def test_channel_and_band_palettes():
    assert len(palette.CHANNEL_COLORS) >= 4
    assert len(palette.BAND_COLORS) == 5  # delta..gamma
    assert all(c.startswith("#") for c in palette.CHANNEL_COLORS)


def test_quality_colors_are_distinct():
    assert len({palette.GOOD, palette.OK, palette.BAD}) == 3


def test_consumers_reference_the_palette():
    # Single source of truth: same objects, not copies that could drift.
    assert board_config.CHANNEL_COLORS is palette.CHANNEL_COLORS
    assert analysis._ELECTRODE_COLORS is palette.ELECTRODE_COLORS


def test_electrode_colors_cover_all_materials():
    for material in board_config.ELECTRODES:
        assert material in palette.ELECTRODE_COLORS, material
