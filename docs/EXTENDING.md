# Extending the app

Three common "I want to add…" recipes. Read `docs/ARCHITECTURE.md` first for the
overall data flow. After any change, run the checks at the bottom.

---

## 1. Add a live plot tab

A plot tab is a `QWidget` that implements the **`PlotTab`** contract (documented
at the top of `ui/session_view.py`):

```python
def update_plot(self, settings, active) -> None: ...   # required: redraw
refresh_hz = 8.0                                        # optional: max redraws/sec
def set_channel_names(self, names) -> None: ...         # optional: relabel traces
```

Steps:

1. Create `ui/plots/my_widget.py` (plot tabs live in `ui/plots/`; side-column
   controls live in `ui/panels/`). Read recent data from the manager and draw:

   ```python
   import pyqtgraph as pg
   from PyQt6.QtWidgets import QVBoxLayout, QWidget
   from ganglion_studio import palette
   from ganglion_studio.core.board_manager import BoardManager
   from ganglion_studio.core.dsp import FilterSettings

   class MyWidget(QWidget):
       refresh_hz = 8.0   # the session view throttles redraws to this

       def __init__(self, manager: BoardManager) -> None:
           super().__init__()
           self._manager = manager
           root = QVBoxLayout(self)
           self._plot = pg.PlotWidget()
           self._curve = self._plot.plot(pen=pg.mkPen(palette.ACCENT, width=1))
           root.addWidget(self._plot)

       def update_plot(self, settings: FilterSettings, active) -> None:
           data = self._manager.recent_eeg(4.0)      # (channels, samples), uV
           if data.shape[1]:
               self._curve.setData(data[0])
   ```

2. Register it in `ui/session_view.py` → `_build_tabs`:

   ```python
   from ganglion_studio.ui.plots.my_widget import MyWidget
   ...
   self.my_widget = MyWidget(self._manager)
   self.tabs.addTab(self.my_widget, "My View")
   ```

3. (Optional) If it should relabel on montage changes, add `set_channel_names`
   and append it to the relabel loop in `_open_channel_setup`.

That's it — the timer renders it automatically when it's the visible tab. Use
colours from `palette.py`, not raw hex.

---

## 2. Add a processing step (Processing Lab)

Steps run in a fixed order inside `core/processing.py` → `apply_pipeline`.

1. Add its settings to `ProcessingConfig` (a dataclass), e.g. a nested config:

   ```python
   @dataclass
   class MyStepConfig:
       enabled: bool = False
       strength: float = 1.0
   ```
   and a field on `ProcessingConfig`: `my_step: MyStepConfig = field(default_factory=MyStepConfig)`.

2. Write the transform — take and return `(channels, samples)` microvolts, and
   **never raise** (degrade gracefully, return a message):

   ```python
   def apply_my_step(eeg, sampling_rate, cfg) -> tuple[np.ndarray, str]:
       try:
           out = eeg * cfg.strength
           return out, f"My step: x{cfg.strength}"
       except Exception as exc:
           return eeg, f"My step skipped: {exc}"
   ```

3. Call it in `apply_pipeline` where it belongs in the order, guarded by its flag:

   ```python
   if config.my_step.enabled:
       data, msg = apply_my_step(data, sampling_rate, config.my_step)
       messages.append(msg)
   ```

4. Add the UI: a `_build_my_step_box()` in `ui/processing_window.py` (copy an
   existing `_build_*_box`; use `self._dspin(...)` for numeric inputs), add it in
   `_build_config_panel`, and read the controls in `_build_config`.

If the step needs an optional library, gate it like ASR/AAS do (disable the box
with a tooltip when the import is missing — see `available_methods`).

---

## 3. Add an analysis-report figure

Report figures are plain functions returning a matplotlib `Figure` in
`core/analysis.py`.

1. Write the builder (reuse `compute_psd`, `_band_powers`, `palette`, etc.):

   ```python
   def fig_my_metric(eeg, sr, metas) -> Figure:
       fig = _new_fig()
       ax = fig.add_subplot(111)
       for m in eeg_metas(metas):
           ax.plot(eeg[m.index], color=_ch_color(m), label=m.name)
       ax.set_title("My metric")
       ax.legend(fontsize=8)
       return fig
   ```

2. Register it in `ui/analysis_window.py` where the other `fig_*` calls add tabs
   /panels, so it shows up in the report.

---

## Always verify

```bash
QT_QPA_PLATFORM=offscreen python smoke_test.py   # full UI + pipeline + export
python -m pytest tests/ -q                        # unit tests
```
