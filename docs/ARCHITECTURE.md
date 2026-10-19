# How Ganglion EEG Studio works

A map of the app for someone who wants to understand or change it. It is written
for a researcher, not a software engineer: each section says *what a part does*
and *where it lives*.

## The big picture

```
                 ┌──────────────┐   start session    ┌────────────────────┐
   main.py  ───▶ │  MainWindow  │ ─────────────────▶ │    SessionView     │
 (entry point)   │ (dashboard ⇄ │                    │  toolbar + tabs +  │
                 │  session)    │ ◀───── back ─────── │  side panels       │
                 └──────┬───────┘                     └─────────┬──────────┘
                        │ owns                                  │ reads recent data
                        ▼                                       ▼
                 ┌──────────────────────────────────────────────────────┐
   OpenBCI       │                   BoardManager                        │
   Ganglion ───▶ │  BrainFlow BoardShim  +  circular ring buffer         │
   (Bluetooth)   │  background thread calls poll() ──▶ newest samples    │
                 └──────────────┬───────────────────────────────────────┘
                                │ when recording
                                ▼
                 ┌──────────────────────────┐      ┌──────────────────────┐
                 │     SessionRecorder      │      │     ReviewWindow      │
                 │ raw CSV + meta + markers │ ───▶ │ browse / edit markers │
                 │ (+ EDF)                  │      │ export (exporter.py)  │
                 └──────────────────────────┘      └──────────────────────┘

  Offline (no board needed):
    recording_loader.py ──▶ ProcessingWindow ──(pipeline)──▶ AnalysisWindow
                                                                (figures from analysis.py)
```

## Layers

The code is split into **`core/`** (no GUI: hardware, signal processing, files)
and **`ui/`** (PyQt6 windows and widgets). `core` never imports `ui`.

- `ganglion_studio/palette.py` — every colour the app uses, named in one place.
- `ganglion_studio/main.py` — starts Qt, applies the theme, opens `MainWindow`.

### Acquisition (`core/board_manager.py`)
`BoardManager` wraps BrainFlow's `BoardShim` and owns a **ring buffer** that
always holds the last `buffer_seconds` of every data row (EEG, resistance,
markers, timestamp…). A dedicated background thread calls `poll()` (the single
*producer*); every widget is a *consumer* that copies a recent slice via
`recent()` / `recent_eeg()`. A lock keeps the producer and the GUI readers safe.
The accelerometer is disabled on start so the Ganglion streams 19-bit EEG.

### Live view (`ui/session_view.py` + `ui/plots/` + `ui/panels/`)
`SessionView` builds the toolbar, the side **panels** (`ui/panels/`: channels,
filters, live stats, markers) and the central **plot tabs** (`ui/plots/`:
time series, PSD, spectrogram, impedance, band power). A `QTimer` tick renders
only the *visible* tab, throttled to that tab's `refresh_hz`. Every tab follows
the small `PlotTab` contract (`update_plot`, optional `refresh_hz` /
`set_channel_names`) documented at the top of `session_view.py`.

### Display vs. recording (important)
Filters in the UI are **display only**. Recording always stores the *raw*
samples. `SessionRecorder` (`core/session.py`) writes, per session:
`*_raw.csv` (BrainFlow), `*_meta.json` (channel map + settings),
`*_markers.csv`, and `*_raw.edf` if MNE is installed.

Marker logs now include sample positions, preserving multiple events at one
sample; the numeric raw marker row is a compatibility view that holds one event
per sample. Legacy logs without sample positions continue to load. Native FIF
exports embed packet-loss intervals as reserved `BAD_ganglion_packet_loss`
annotations, so provenance follows the recording through cropping and copying.
These annotations are kept separate from experimental event markers in the UI.

The Dashboard also offers an independent impedance-only session. The native
driver decodes impedance packets into timestamped channel events, while the
BrainFlow path extracts the freshly updated channel from each packet ID. These
events have a bounded live history and a separate recording buffer; they never
enter the EEG ring. `ImpedanceSessionView` plots the events and `ImpedanceRecorder`
saves/reloads their irregular time series as CSV plus metadata. See
[IMPEDANCE.md](IMPEDANCE.md) for firmware units and mode-switching constraints.

EEG sessions also support managed timed impedance checks. `SessionView` retains a
mode-switch QThread until it finishes, times the measurement phase and waits for
transitions before shutdown. `BoardManager` keeps the logical EEG recording open,
collects impedance events separately, emits reserved transition markers 90–93
and inserts NaN EEG placeholders during the interruption. Its `eeg_gaps` metadata
and reserved `BAD_impedance_check` annotations preserve elapsed time through
CSV/FIF/EDF, independently of packet-loss metadata. `SessionRecorder` writes a
linked impedance CSV and metadata in the same session directory. DSP operates on
continuous finite runs, keeping these gaps missing throughout display and
processing. See `core/impedance_check.py` and `core/eeg_gaps.py`.

### Offline analysis
`core/recording_loader.py` loads a saved recording (CSV+meta or any MNE format)
into a uniform `LoadedRecording` (always microvolts). `ProcessingWindow`
(`ui/processing_window.py`) shows original vs. processed and runs the pipeline in
`core/processing.py`:

`CAR re-reference → detrend → band-pass/notch filters → wavelet denoise →
ASR → ECG removal (R-peak-locked AAS)`

`AnalysisWindow` (`ui/analysis_window.py`) renders the report figures built in
`core/analysis.py`. `core/exporter.py` writes `.fif/.set/.edf` (and best-effort
`.gdf`).

Offline sampling rates stay floating point through import, processing and export.
Welch PSD uses SciPy density scaling, and EEG band powers include exact band edges
by interpolation. SciPy filters handle rates that BrainFlow's integer API cannot
represent. This preserves timing and physical spectral-power units.

### Signal processing (`core/dsp.py`)
Shared DSP helpers used everywhere: `apply_filters`, `compute_psd`,
`compute_fft`, `compute_spectrogram`, `compute_band_powers`, plus the
contact-quality metrics (`signal_quality`, `is_railed`, `quality_label`).

## Where do I change…?

| I want to change…                         | Edit…                                   |
|-------------------------------------------|-----------------------------------------|
| Any colour (traces, good/ok/bad, bands)   | `palette.py`                            |
| EEG band edges (delta…gamma)              | `dsp.py` → `EEG_BANDS`                  |
| Available filter types                    | `dsp.py` → `FILTER_TYPES`              |
| The "railed/clipping" threshold           | `dsp.py` → `GANGLION_FULLSCALE_UV`     |
| Impedance good/ok thresholds              | `board_config.py` → `IMPEDANCE_*_KOHM` |
| Channel types / electrode materials       | `board_config.py` (single source)      |
| Ganglion ASCII commands                   | `board_config.py`                      |
| How long the live buffer keeps data       | `BoardManager(buffer_seconds=...)`     |
| Add a plot tab / processing step / figure | see `docs/EXTENDING.md`                |

## Running it

```bash
pip install -r requirements.txt
python -m ganglion_studio.main          # real app
QT_QPA_PLATFORM=offscreen python smoke_test.py   # headless end-to-end check
python -m pytest tests/ -q              # unit tests
```
