# Ganglion EEG Studio

An interactive desktop application to fully control and visualize an **OpenBCI Ganglion**
EEG board over **native Bluetooth**, built on [BrainFlow](https://brainflow.org/),
PyQt6 and pyqtgraph.

Scan for your Ganglion, name a recording session, and open a rich session view with
live time series, PSD, spectrogram/FFT, live impedance and band power -
all with full control over board, channels, filters and markers.

## Documentation
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - how the app is organised and
  how data flows through it (start here to understand the code).
- **[docs/EXTENDING.md](docs/EXTENDING.md)** - recipes for adding a plot tab, a
  processing step, or an analysis-report figure.

## Features

### Dashboard
- **Native Bluetooth scan** for nearby BLE devices (via `bleak`), with Ganglion
  auto-highlighting and RSSI. No BLED112 dongle required (`GANGLION_NATIVE_BOARD`).
- **Named sessions**, optional explicit MAC address, firmware selector
  (FW3 default / FW2 legacy / auto) and mains-notch region (50/60 Hz).
- **Demo mode** (BrainFlow synthetic board) to explore the whole app with no hardware.

### Channel setup (recording view)
- A **Channel setup** dialog lets you set each channel's signal **type** (EEG/EMG/ECG/MISC),
  **electrode** (Ag/AgCl wet/dry, PEDOT:PSS, PEDOT, Other) and **10-20 placement**.
  The placement becomes the channel's display name across all plots, the config is
  written into each recording's `_meta.json`, and montages can be saved/loaded as
  reusable JSON presets.

### Live signal statistics
- A left-column panel updates continuously with per-channel **RMS, peak-to-peak,
  std, dominant frequency, mains-noise fraction and a contact-quality flag**
  (good/ok/bad), plus general info (sampling rate, window, buffer fill, mean RMS).

### Processing Lab (offline)
- Open the **Processing Lab** any time (button on the dashboard, or in the session
  toolbar) to load a saved recording (`.csv`+meta / `.fif` / `.edf` / `.set` / `.gdf`)
  and experiment with a preprocessing pipeline. Configuration is on the left;
  the **original** signal is shown top-right and the **processed** signal
  bottom-right (time or PSD view, linked time axis).
- Pipeline (each step independently toggleable): common-average re-reference ->
  detrend -> filters (band-pass + notch) -> wavelet denoising (BrainFlow) ->
  ASR (Artifact Subspace Reconstruction, `meegkit`) -> ECG removal by
  **R-peak-locked median AAS** (R-peaks detected with `neurokit2`). Heavy steps
  run on a worker thread so the UI stays responsive. Steps whose optional backend
  is missing are disabled with a tooltip.
- **Channel typing**: set each channel's type (EEG/ECG/EMG/MISC) and electrode
  material (Ag/AgCl wet/dry, PEDOT:PSS, PEDOT, Other), pre-filled from the loaded
  recording's metadata. An ECG-typed channel auto-fills the AAS reference. Each
  channel also has an **On/Off** toggle that excludes it from re-referencing,
  the plots, and the analysis report.

### Analysis & electrode-characterization report
- **Generate analysis plots** (Processing Lab) opens a tabbed, descriptive report:
  - *EEG analysis*: Welch PSD (log-log, bands shaded), relative band-power bars,
    per-channel quality/noise table.
  - *Electrode characterization*: RMS noise and 50/60 Hz line-noise bars and a PSD
    overlay, coloured by electrode material.
  - *Compare channels*: pick any two channels (A vs B) and compare them with a
    statistics table (Pearson + Spearman r, RMSE/NRMSE, per-band coherence,
    Bland-Altman bias/LoA, per-channel RMS/line-noise/alpha-ratio), overlaid time series,
    PSD overlay, coherence, correlation scatter, cross-correlation, amplitude
    histograms, and band-power comparison.
  - *Material groups* (shown when both PEDOT and Ag/AgCl are present): group-mean
    PSD +/- std and band-power comparison with t-tests.
- Switch the report source between processed and original signal. Each figure has
  its own Save (PNG/SVG/PDF) plus a Save-all option.

### Recording review & export
- When you stop a recording, a **Review window** opens: browse the whole
  recording (scroll, window/amplitude, optional display filter), see all markers,
  and **add/remove** markers (click to place, choose a marker type).
- **Save / Export** to research formats: **.fif** (MNE native, lossless),
  **.set** (EEGLAB), **.edf** (European Data Format). Markers are written as
  annotations. **.gdf** is offered best-effort via the optional BioSig toolkit
  (MNE/standard Python libraries cannot write GDF); EDF is the recommended open
  alternative. A lossless CSV + metadata + marker backup is always written too.

### Session view
- **Live time series** - one trace per channel, with controls for window length,
  amplitude scale, auto-scale and marker overlays.
- **PSD** - Welch power spectral density with log/linear Y, window length and max-freq.
- **Spectrogram / FFT** - rolling spectrogram plus instantaneous FFT per channel.
- **Live impedance** - per-electrode bars colour-coded good/ok/bad with an
  impedance-over-time history (electrode characterization).
- **Band power** - delta / theta / alpha / beta / gamma averaged over active channels.

### Full board & plot control
- **Channels**: enable/disable each of the 4 Ganglion channels. Toggling sends the
  real Ganglion ASCII command (`!@#$` on / `1234` off) and hides/shows the trace.
- **Filters (display only)**: band-pass (low/high/order/type), mains notch (50/60 Hz)
  and detrend. Filters never alter recorded raw data.
- **Plot controls**: refresh rate, window, amplitude, frequency range, channel select.
- **Impedance test**: start/stop the Ganglion LeadOff impedance test (`z`/`Z`).

### Markers / annotations
- Define marker types (label + numeric code + colour). Fire them with a button **or a
  number-key hotkey** to time experiment protocols.
- Each marker is embedded in the recorded BrainFlow marker channel
  (`insert_marker`) and logged with a timestamp. Export the marker log to CSV.

### Recording & export
- Records **raw, unfiltered** data to a timestamped folder under `recordings/`:
  - `*_raw.csv` - BrainFlow native format
  - `*_meta.json` - session metadata + channel map
  - `*_markers.csv` - annotation log
  - `*_raw.edf` - optional, written automatically if `mne` is installed

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

This installs everything (including the optional export/processing libraries).
Alternatively, install the package itself to also get the `ganglion-studio`
command, and pick optional extras explicitly:

```bash
pip install -e .                       # core only, editable
pip install -e ".[export,processing]"  # + research export + Processing Lab algorithms
```

On Linux you may also need Qt/BLE system libraries, e.g.:

```bash
sudo apt-get install -y libegl1 libgl1 libxkbcommon0 libdbus-1-3
```

## Running

```bash
python -m ganglion_studio.main
```

1. Click **Scan Bluetooth** and select your Ganglion (or tick **Demo mode**).
2. Enter a **session name** and choose firmware / notch region.
3. Click **Start Session**.

### Using a real Ganglion
- Make sure the board is powered and within range, and that BLE is enabled on your
  computer (no dongle needed for the native board).
- On macOS 12.3+ you may need to grant the terminal/app Bluetooth permission. On
  Linux, running with appropriate BLE permissions (or `sudo`) may be required.
- If your board has firmware v2, choose **2 (legacy)** in the firmware selector
  (this sets `BrainFlowInputParams.other_info = "fw:2"`).

## Troubleshooting Bluetooth
- Scanning runs in an isolated helper subprocess (`python -m ganglion_studio.core.ble_scanner`).
  This is required on macOS (CoreBluetooth must run on a process main thread) and
  ensures a BLE-stack crash can never take down the app - you get a "scan failed"
  message instead. You can run that command directly to debug discovery.
- **macOS**: grant the terminal/app Bluetooth permission when prompted (System
  Settings -> Privacy & Security -> Bluetooth); use 12.3+.
- **Linux**: BlueZ must be running (`systemctl status bluetooth`); the
  `org.bluez ... not provided` error means the Bluetooth service is not up.
- **Windows**: 10.0.19041.0+ works without extra setup.
- If scanning is unavailable, you can still enter the board MAC manually or use Demo mode.

## Notes
- The accelerometer is kept **disabled** so the Ganglion streams EEG as 19-bit
  deltas; enabling it would drop the EEG to 18-bit. There is no motion/accel view
  by design (the `N` disable command is sent once on stream start).
- The cloud/CI environment used to develop this app has no Bluetooth radio, so real
  hardware streaming must be validated on your own machine. The full UI and data
  pipeline are validated in Demo mode (`smoke_test.py`).
- Marker number-key hotkeys are application-wide while a session is open; avoid using
  the number keys while editing a numeric control if you do not intend to drop a marker.

## Project layout

```
ganglion_studio/
  main.py                  # entry point (python -m ganglion_studio.main)
  palette.py               # every colour the app uses, named in one place
  core/                    # no GUI: hardware, signal processing, files
    board_manager.py       # BrainFlow BoardShim wrapper, ring buffer, commands, recording
    board_config.py        # Ganglion ASCII commands, channel/electrode types, impedance thresholds
    native_ganglion.py     # custom bleak native-BLE driver (delta/MSB decode, packet loss)
    ble_scanner.py         # native BLE discovery (bleak), run in a helper subprocess
    serial_ports.py        # BLED112 dongle serial-port discovery
    saved_devices.py       # remembered device list (name + address)
    dsp.py                 # filters, PSD, FFT, spectrogram, band powers, quality metrics
    session.py             # session config + live recorder (raw CSV / meta / markers / EDF)
    recording_loader.py    # load a saved recording (CSV+meta or MNE format) -> LoadedRecording
    processing.py          # offline pipeline (CAR, detrend, filters, wavelet, ASR, ECG-AAS)
    analysis.py            # report figures + channel/pair metrics (matplotlib)
    exporter.py            # save/export .fif / .set / .edf (+ best-effort .gdf)
  ui/                      # PyQt6 windows and widgets (core never imports ui)
    main_window.py         # dashboard <-> session stack, async board prepare
    dashboard.py           # scan + session setup
    session_view.py        # toolbar, panels, plot tabs, refresh timer
    channel_setup_dialog.py# per-channel type / electrode / 10-20 placement + montage presets
    review_window.py       # post-recording browse + marker edit + export
    processing_window.py   # Processing Lab (original vs processed, pipeline controls)
    analysis_window.py     # tabbed analysis / electrode-characterization report
    theme.py               # dark theme
    plots/                 # plot tabs (PlotTab contract): time_series, psd, spectrogram,
                           #   impedance, band_power
    panels/                # side-column controls: channel, filter, stats, marker

tools/                     # hardware / debug scripts (diagnose_artifact, native_raw_dump,
                           #   native_stream_check) -- not needed to run the app
tests/                     # unit tests (python -m pytest tests/ -q)
smoke_test.py              # headless end-to-end check (QT_QPA_PLATFORM=offscreen)
docs/                      # ARCHITECTURE.md, EXTENDING.md
```

## Ideas / possible add-ons
- Session replay using BrainFlow's `PLAYBACK_FILE_BOARD`.
- Focus/relaxation metric via BrainFlow `MLModel`.
- Per-channel artifact rejection and automatic contact-quality scoring report.
- LSL output stream for integration with other acquisition tools.
