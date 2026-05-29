"""Headless smoke test for Ganglion EEG Studio in demo mode."""
import os
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMessageBox
# Avoid modal dialogs blocking the headless run.
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
from ganglion_studio.core.board_manager import BoardManager
from ganglion_studio.core.session import SessionConfig
from ganglion_studio.ui.session_view import SessionView

app = QApplication([])

config = SessionConfig(name="smoke test", demo=True, notch_freq=50)
mgr = BoardManager(demo=True)
mgr.prepare()
mgr.start()
print("prepared:", mgr.board_id, "sr", mgr.sampling_rate, "eeg", mgr.eeg_channels)

# Let the synthetic board produce data.
for _ in range(8):
    time.sleep(0.25)
    app.processEvents()
    mgr.poll()
print("ring filled samples:", mgr._filled)

view = SessionView(mgr, config)
view.resize(1400, 900)

# Exercise every tab's update_plot.
for i in range(view.tabs.count()):
    view.tabs.setCurrentIndex(i)
    mgr.poll()
    w = view.tabs.currentWidget()
    w.update_plot(view._settings, view._active)
    print("tab ok:", view.tabs.tabText(i))

# Channel toggle.
view.channel_panel._on_toggle(1, False)
print("active after toggle:", view._active)

# Filter change.
from ganglion_studio.core.dsp import FilterSettings
view._on_filters_changed(FilterSettings(bp_low=2.0, bp_high=40.0, notch_freq=60))
view.time_series.update_plot(view._settings, view._active)
print("filters applied")

# Impedance mode.
view.impedance._on_toggle(True)
mgr.poll()
view.impedance.update_plot(view._settings, view._active)
print("impedance kohm:", [round(x, 1) for x in mgr.latest_impedance_kohm()])
view.impedance._on_toggle(False)

# Recording + markers.
view.record_btn.setChecked(True)
for _ in range(6):
    time.sleep(0.2)
    mgr.poll()
view._on_marker(3, "Stimulus", time.time())
view.marker_panel.fire(view.marker_panel._types[0])
view.record_btn.setChecked(False)
print("recorded samples:", mgr.recorded_sample_count())

mgr.release()
print("SMOKE TEST PASSED")
