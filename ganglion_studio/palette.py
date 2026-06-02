"""Central colour palette for Ganglion EEG Studio.

Every colour the app uses -- window chrome, plot traces, contact-quality flags,
electrode materials, EEG-band bars -- is named here so a visual change happens in
ONE place instead of being hunted across a dozen widgets.

Values are plain hex strings that work everywhere we draw: PyQt6 stylesheets and
``QColor``, pyqtgraph pens/brushes, and matplotlib. This module deliberately has
no dependencies, so both ``core`` and ``ui`` can import it
(``from ganglion_studio import palette``).
"""

from __future__ import annotations

# --- Window chrome (dark theme) -------------------------------------------- #
BG = "#1b1d23"        # window background
BG_ALT = "#23262e"    # panels, inputs, tab strip
FG = "#e6e6e6"        # primary text
ACCENT = "#4f8ef7"    # highlight / selection (also the blue trace colour)
WHITE = "#ffffff"

# --- Semantic status colours ----------------------------------------------- #
GOOD = "#5fd38d"      # green  - good contact / pass
OK = "#e2c044"        # yellow - marginal
BAD = "#f7766f"       # red    - bad contact / railed / fail
MUTED = "#9aa0aa"     # secondary / hint text
DISABLED = "#666666"  # greyed-out values
NEUTRAL = "#888888"   # unknown / not-applicable

# A soft violet reused by the band bars and the Bland-Altman scatter.
VIOLET = "#b48ef7"

# --- Plot colours ---------------------------------------------------------- #
# Per-channel trace palette (channel i -> CHANNEL_COLORS[i % len]). Kept stable so
# a given channel is the same colour in every view.
CHANNEL_COLORS = [ACCENT, BAD, GOOD, OK]

# EEG band bars, in ``dsp.EEG_BANDS`` order: Delta, Theta, Alpha, Beta, Gamma.
BAND_COLORS = [VIOLET, ACCENT, GOOD, OK, BAD]

# Electrode-material colours used by the analysis / characterization figures.
ELECTRODE_COLORS = {
    "Ag/AgCl (wet)": ACCENT,
    "Ag/AgCl (dry)": GOOD,
    "PEDOT:PSS": BAD,
    "PEDOT": "#e2722c",
    "Other": MUTED,
}

# In the two-material comparison plots, PEDOT is drawn red-ish, Ag/AgCl blue.
MATERIAL_PEDOT = BAD
MATERIAL_AGAGCL = ACCENT

# Vertical event/marker lines on the time-series plot.
MARKER = BAD

# A pink for categorical cycling.
PINK = "#f79edb"

# Distinct colours cycled when assigning a colour to a new marker type.
MARKER_PALETTE = [GOOD, ACCENT, OK, BAD, VIOLET, PINK]
