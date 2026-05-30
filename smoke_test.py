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

view.shutdown()
mgr.release()
assert not (mgr._acq_thread and mgr._acq_thread.is_alive()), "acquisition thread not stopped"
print("SMOKE TEST PASSED")
