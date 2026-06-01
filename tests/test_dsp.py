"""DSP unit tests, including a regression for the Ganglion full-scale constant.

The railing detector previously used +/- 187500 uV (the *Cyton's* range), which a
real Ganglion signal can never reach -- so railing was dead code. These tests lock
in the correct +/- 15686 uV MCP3912 full-scale.
"""

import numpy as np

from ganglion_studio.core import dsp


def test_full_scale_constant_is_ganglion_not_cyton():
    # MCP3912 (Vref 1.2 V, gain 51): 1.2e6 / (1.5 * 51) ~= 15686 uV.
    assert abs(dsp.GANGLION_FULLSCALE_UV - 1.2e6 / (1.5 * 51)) < 1.0
    # Guard against regressing to the Cyton's 187500 uV.
    assert dsp.GANGLION_FULLSCALE_UV < 20000.0


def test_is_railed_at_full_scale():
    sr = 200
    railed = np.full(sr, dsp.GANGLION_FULLSCALE_UV, dtype=np.float64)
    assert dsp.is_railed(railed) is True
    assert dsp.signal_quality(railed, sr)["railed"] is True


def test_clean_signal_not_railed():
    sr = 200
    t = np.arange(sr) / sr
    clean = 50.0 * np.sin(2 * np.pi * 10 * t)  # 50 uV, 10 Hz
    assert dsp.is_railed(clean) is False
    assert dsp.signal_quality(clean, sr)["railed"] is False


def test_is_railed_empty():
    assert dsp.is_railed(np.array([])) is False


def test_quality_label_boundaries():
    assert dsp.quality_label({"railed": True}) == "bad"
    assert dsp.quality_label({"ptp": 50.0, "line_ratio": 0.10}) == "good"
    assert dsp.quality_label({"ptp": 300.0, "line_ratio": 0.10}) == "ok"
    assert dsp.quality_label({"ptp": 50.0, "line_ratio": 0.30}) == "ok"
    assert dsp.quality_label({"ptp": 1500.0, "line_ratio": 0.10}) == "bad"
    assert dsp.quality_label({"ptp": 50.0, "line_ratio": 0.60}) == "bad"


def test_dominant_frequency_recovers_tone():
    sr = 200
    t = np.arange(4 * sr) / sr
    sig = 30.0 * np.sin(2 * np.pi * 12.0 * t)
    assert abs(dsp.dominant_frequency(sig, sr) - 12.0) < 1.0
