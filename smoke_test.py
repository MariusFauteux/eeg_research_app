"""Headless smoke + performance test for Ganglion EEG Studio (demo mode).

Validates that:
* acquisition runs on its background thread (no manual polling needed),
* every tab renders without error and within a sane time budget,
* heavy tabs are throttled (update_plot runs far less often than the GUI tick),
* recording captures continuous data and markers work.
"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMessageBox

# Avoid modal dialogs blocking the headless run.
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)

from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.dsp import FilterSettings
from ganglion_studio.core.session import SessionConfig
from ganglion_studio.ui.session_view import SessionView

app = QApplication([])

config = SessionConfig(name="smoke test", demo=True, notch_freq=50)
mgr = BoardManager(demo=True)
mgr.prepare()
mgr.start()
sr = mgr.sampling_rate
print("prepared:", mgr.board_id, "sr", sr, "eeg", mgr.eeg_channels)

# Constructing the view starts the background acquisition thread.
view = SessionView(mgr, config)
view.resize(1400, 900)
assert mgr._acq_thread is not None and mgr._acq_thread.is_alive(), "acquisition thread not running"

# The thread should fill the ring buffer on its own (no manual poll calls).
time.sleep(2.0)
app.processEvents()
print("ring filled samples (thread):", mgr._filled)
assert mgr._filled > sr, "background acquisition did not fill the buffer"

# Time each tab's render; nothing should block for long.
max_dt = 0.0
for i in range(view.tabs.count()):
    view.tabs.setCurrentIndex(i)
    w = view.tabs.currentWidget()
    if hasattr(w, "_last_render"):
        w._last_render = 0.0
    t0 = time.perf_counter()
    w.update_plot(view._settings, view._active)
    dt = (time.perf_counter() - t0) * 1000.0
    max_dt = max(max_dt, dt)
    print(f"tab ok: {view.tabs.tabText(i):<18} {dt:6.1f} ms")
    assert dt < 500.0, f"{view.tabs.tabText(i)} render too slow: {dt:.0f} ms"
print(f"slowest single render: {max_dt:.1f} ms")

# Throttling: drive _tick in a tight loop for ~1s on the spectrogram tab and
# confirm its expensive update_plot ran only a handful of times.
spec_idx = [view.tabs.tabText(i) for i in range(view.tabs.count())].index("Spectrogram / FFT")
view.tabs.setCurrentIndex(spec_idx)
spec = view.tabs.currentWidget()
calls = {"n": 0}
_orig = spec.update_plot
spec.update_plot = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), _orig(*a, **k))[1]
spec._last_render = 0.0
end = time.time() + 1.0
ticks = 0
while time.time() < end:
    view._tick()
    ticks += 1
    time.sleep(0.005)
spec.update_plot = _orig
print(f"spectrogram throttle: {calls['n']} renders over {ticks} ticks in ~1s")
assert calls["n"] <= 8, f"spectrogram not throttled ({calls['n']} renders)"

# Channel + filter controls.
view.channel_panel._checks[1].setChecked(False)
print("active after toggle:", view._active)
assert view._active[1] is False
view._on_filters_changed(FilterSettings(bp_low=2.0, bp_high=40.0, notch_freq=60))
view.time_series.update_plot(view._settings, view._active)
print("filters applied")

# Impedance mode.
view.impedance._on_toggle(True)
time.sleep(0.3)
view.impedance.update_plot(view._settings, view._active)
print("impedance kohm:", [round(x, 1) for x in mgr.latest_impedance_kohm()])
view.impedance._on_toggle(False)

# Recording + markers (data captured by the acquisition thread).
view.record_btn.setChecked(True)
time.sleep(1.2)
view._on_marker(3, "Stimulus", time.time())
view.marker_panel.fire(view.marker_panel._types[0])
view.record_btn.setChecked(False)
rec = mgr.recorded_sample_count()
print("recorded samples:", rec)
assert rec > sr, "recording did not capture continuous data"

# Signal statistics panel.
view.stats_panel.update_stats(view._settings, view._active)
print("stats general:", view.stats_panel.general.text())
assert view.stats_panel.table.item(0, 1).text() not in ("", "-"), "stats not populated"

# Channel setup: set names/types/electrodes/placements and relabel widgets.
mgr.set_channel_config(
    ["Fp1", "Fp2", "C3", "C4"],
    ["EEG", "EEG", "EEG", "ECG"],
    ["PEDOT:PSS", "Ag/AgCl (wet)", "PEDOT:PSS", "Other"],
    ["Fp1", "Fp2", "C3", "None"],
)
for _w in (view.time_series, view.psd, view.spectrogram, view.impedance,
           view.stats_panel, view.channel_panel):
    _w.set_channel_names(mgr.channel_names)
assert view.stats_panel.table.item(0, 0).text() == "Fp1", "stats relabel failed"
assert view.channel_panel._checks[0].text() == "Fp1", "channel panel relabel failed"
assert view.spectrogram.ch_combo.itemText(0) == "Fp1", "spectrogram relabel failed"
print("channel relabel ok:", mgr.channel_names)

# Recording review + export.
import tempfile
import numpy as np
from ganglion_studio.ui.review_window import ReviewWindow
from ganglion_studio.core import exporter

raw = mgr.recent(5.0)
# Inject a couple of markers directly into the recorded marker row for review.
mch = mgr.marker_channel
raw[mch, 100] = 1
raw[mch, 400] = 3
meta = {
    "sampling_rate": sr,
    "eeg_channels": mgr.eeg_channels,
    "channel_names": ["Ch1", "Ch2", "Ch3", "Ch4"][: len(mgr.eeg_channels)],
    "marker_channel": mch,
    "notch_freq": 50,
}
review = ReviewWindow(raw, meta, code_labels={1: "Eyes Open", 3: "Stimulus"},
                      marker_types=view.marker_panel._types, title="smoke")
n0 = len(review._markers)
print("review markers extracted:", n0)
assert n0 >= 2, "marker extraction failed"

# Add then remove a marker.
review._click_time = 1.0
review._on_add_marker()
assert len(review._markers) == n0 + 1, "add marker failed"
review.table.selectRow(0)
review._on_remove_marker()
assert len(review._markers) == n0, "remove marker failed"
print("marker edit ok ->", len(review._markers))

# Exports.
print("available formats:", exporter.available_formats())
tmp = tempfile.mkdtemp()
for fmt, ext in [("fif", ".fif"), ("set", ".set"), ("edf", ".edf")]:
    path = os.path.join(tmp, f"rec{ext}")
    out = exporter.export(path, fmt, raw, meta, review._markers)
    assert os.path.exists(out), f"{fmt} not written"
    print(f"exported {fmt}: {os.path.basename(out)} ({os.path.getsize(out)} bytes)")

# GDF should fail gracefully without BioSig.
try:
    exporter.export(os.path.join(tmp, "rec.gdf"), "gdf", raw, meta, review._markers)
    print("gdf: written (BioSig present)")
except exporter.ExportError as e:
    print("gdf gracefully unavailable:", str(e).splitlines()[0])

# --- Processing Lab: loader, pipeline, AAS, and window recompute ---
from ganglion_studio.core import processing as P
from ganglion_studio.core.recording_loader import load_recording
from ganglion_studio.ui.processing_window import ProcessingWindow
from ganglion_studio.core.session import SessionRecorder, SessionConfig as SC

# Write a real recording file via the recorder, then load it back.
rec_raw = mgr.recent(8.0)
sr2 = mgr.sampling_rate
rmeta = {
    "sampling_rate": sr2,
    "eeg_channels": mgr.eeg_channels,
    "channel_names": list(mgr.channel_names),
    "channel_types": list(mgr.channel_types),
    "electrodes": list(mgr.electrodes),
    "placements": list(mgr.placements),
    "marker_channel": mgr.marker_channel,
}
rec = SessionRecorder(SC(name="proc-lab-test", demo=True))
rec.begin()
written = rec.save(rec_raw, rmeta)
csv_path = [w for w in written if w.endswith("_raw.csv")][0]
loaded = load_recording(csv_path)
print("loaded:", loaded.n_channels, "ch", loaded.sampling_rate, "Hz", loaded.n_samples, "samp")
assert loaded.n_channels == len(mgr.eeg_channels) and loaded.n_samples > 0
# Channel config persisted into meta and read back by the loader.
assert loaded.channel_names[0] == "Fp1", "channel name not persisted"
assert loaded.electrodes and loaded.electrodes[0] == "PEDOT:PSS", "electrode not persisted"
assert loaded.channel_types and loaded.channel_types[3] == "ECG", "type not persisted"
print("channel config persisted:", loaded.channel_types, loaded.electrodes)

# AAS: synthesize an ECG-like reference with known R-peaks + artifact, verify reduction.
n2 = sr2 * 20
tt = np.arange(n2) / sr2
eeg2 = np.random.randn(4, n2) * 8 + 15 * np.sin(2 * np.pi * 10 * tt)
beats = list(range(sr2 // 2, n2 - sr2, sr2))  # ~1 Hz
for p in beats:
    eeg2[0, p - 3:p + 3] += 300 * np.hanning(6)   # ECG ref channel
    eeg2[1, p - 3:p + 3] += 100 * np.hanning(6)   # cardiac artifact in EEG
peaks = P.detect_rpeaks(eeg2[0], sr2)
print("R-peaks detected:", len(peaks), "expected ~", len(beats))
assert abs(len(peaks) - len(beats)) <= 3, "R-peak detection off"
aas_cfg = P.AasStepConfig(enabled=True, ref_channel=0, pre_ms=200, post_ms=400)
cleaned, msg = P.apply_aas(eeg2, sr2, aas_cfg)
art_before = np.ptp(eeg2[1])
art_after = np.ptp(cleaned[1])
print(f"AAS: {msg} | ch1 ptp {art_before:.0f} -> {art_after:.0f}")
assert art_after < art_before, "AAS did not reduce the cardiac artifact"

# Full pipeline incl ASR (meegkit) if available.
cfg2 = P.ProcessingConfig(reref_car=True, detrend="linear")
cfg2.wavelet.enabled = True
cfg2.asr.enabled = P.available_methods()["asr"]
cfg2.aas.enabled = True
cfg2.aas.ref_channel = 0
out2, msgs2 = P.apply_pipeline(eeg2, sr2, ["Ch1", "Ch2", "Ch3", "Ch4"], cfg2)
assert out2.shape == eeg2.shape and np.isfinite(out2).all()
print("pipeline steps:", " | ".join(msgs2))

# Processing window: load a file and run one synchronous recompute.
win = ProcessingWindow()
win._load(csv_path)
# Loaded metadata should pre-fill the channel type/electrode combos.
assert win._elec_combos[0].currentText() == "PEDOT:PSS", "lab did not pre-fill electrode"
win.wavelet_box.setChecked(True)
win._recompute()
win._worker.wait(20000)
app.processEvents()
assert win._processed is not None and win._processed.shape == loaded.eeg.shape
print("processing window recompute ok")

# Deactivate a channel: excluded from CAR active set and from analysis metas.
win._enable_checks[1].setChecked(False)
dcfg = win._build_config()
assert 1 not in (dcfg.active_channels or []), "deactivated channel still in CAR set"
enabled_metas = win.channel_metas(enabled_only=True)
assert all(m.index != 1 for m in enabled_metas), "deactivated channel still in analysis metas"
print("deactivation ok; active channels:", dcfg.active_channels)

# --- Analysis report: channel typing, figures, comparison stats, save ---
import matplotlib
matplotlib.use("Agg")
from ganglion_studio.core import analysis as A
from ganglion_studio.ui.analysis_window import AnalysisWindow

# Synthetic recording: 2 PEDOT EEG, 2 Ag/AgCl EEG, 1 ECG.
na = sr2 * 30
ta = np.arange(na) / sr2
aeeg = np.random.randn(5, na) * 8 + 20 * np.sin(2 * np.pi * 10 * ta)
aeeg[4] = 0.0
for p in range(sr2 // 2, na, sr2):
    aeeg[4, max(0, p - 3):p + 3] += 300 * np.hanning(6)
ametas = [
    A.ChannelMeta(0, "P1", "EEG", "PEDOT:PSS"),
    A.ChannelMeta(1, "P2", "EEG", "PEDOT:PSS"),
    A.ChannelMeta(2, "A1", "EEG", "Ag/AgCl (wet)"),
    A.ChannelMeta(3, "A2", "EEG", "Ag/AgCl (dry)"),
    A.ChannelMeta(4, "ECG", "ECG", "Other"),
]
assert A.comparison_available(ametas)
agm = A.pair_agreement(aeeg[0], aeeg[2], sr2)
for k in ("r", "spearman", "rmse", "nrmse", "mean_coherence_1_30", "ba_bias"):
    assert np.isfinite(agm[k]), f"stat {k} not finite"
assert set(agm["band_coherence"].keys()) == {"Delta", "Theta", "Alpha", "Beta", "Gamma"}
gstats = A.group_band_stats(aeeg, sr2, ametas)
assert "PEDOT" in gstats["groups"] and "Ag/AgCl" in gstats["groups"]
print(f"analysis: r={agm['r']:.2f} spearman={agm['spearman']:.2f} coh={agm['mean_coherence_1_30']:.2f} "
      f"groups={ {k: v['n'] for k, v in gstats['groups'].items()} } pvals={len(gstats.get('pvals', []))}")

awin = AnalysisWindow(aeeg, aeeg.copy(), sr2, ametas, title="analysis-test")
tab_titles = [awin.tabs.tabText(i) for i in range(awin.tabs.count())]
assert "Compare channels" in tab_titles, tab_titles
assert "Material groups" in tab_titles, tab_titles
n_panels = len(awin._panels)
print("analysis tabs:", tab_titles, "| panels:", n_panels)
assert n_panels >= 8
# Switch the A/B comparison pair and confirm a rebuild.
awin.chan_b_combo.setCurrentIndex(1)
app.processEvents()
# Save one figure and switch source without error.
atmp = tempfile.mkdtemp()
awin._panels[0][1].figure().savefig(os.path.join(atmp, "fig0.png"), dpi=90)
assert os.path.getsize(os.path.join(atmp, "fig0.png")) > 0
awin.source_combo.setCurrentText("Original")
app.processEvents()
print("analysis window ok; figure saved")

import shutil
shutil.rmtree("recordings", ignore_errors=True)

view.shutdown()
mgr.release()
assert not (mgr._acq_thread and mgr._acq_thread.is_alive()), "acquisition thread not stopped"
print("SMOKE TEST PASSED")
