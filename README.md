# Ganglion EEG Studio

An interactive desktop application to fully control and visualize an **OpenBCI Ganglion**
EEG board over **native Bluetooth**, built on [BrainFlow](https://brainflow.org/),
PyQt6 and pyqtgraph.

Scan for your Ganglion, name a recording session, and open a rich session view with
live time series, PSD, spectrogram/FFT, live impedance, band power and motion plots -
all with full control over board, channels, filters and markers.

## Features

### Dashboard
- **Native Bluetooth scan** for nearby BLE devices (via `bleak`), with Ganglion
  auto-highlighting and RSSI. No BLED112 dongle required (`GANGLION_NATIVE_BOARD`).
- **Named sessions**, optional explicit MAC address, firmware selector
  (FW3 default / FW2 legacy / auto) and mains-notch region (50/60 Hz).
- **Demo mode** (BrainFlow synthetic board) to explore the whole app with no hardware.

### Session view
- **Live time series** - one trace per channel, with controls for window length,
  amplitude scale, auto-scale and marker overlays.
- **PSD** - Welch power spectral density with log/linear Y, window length and max-freq.
- **Spectrogram / FFT** - rolling spectrogram plus instantaneous FFT per channel.
- **Live impedance** - per-electrode bars colour-coded good/ok/bad with an
  impedance-over-time history (electrode characterization).
- **Band power** - delta / theta / alpha / beta / gamma averaged over active channels.
- **Accelerometer / motion** - X/Y/Z traces for movement-artifact monitoring.

### Full board & plot control
- **Channels**: enable/disable each of the 4 Ganglion channels. Toggling sends the
  real Ganglion ASCII command (`!@#$` on / `1234` off) and hides/shows the trace.
- **Filters (display only)**: band-pass (low/high/order/type), mains notch (50/60 Hz)
  and detrend. Filters never alter recorded raw data.
- **Plot controls**: refresh rate, window, amplitude, frequency range, channel select.
- **Impedance test**: start/stop the Ganglion LeadOff impedance test (`z`/`Z`).
- **Accelerometer**: enable/disable (`n`/`N`).

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
- The cloud/CI environment used to develop this app has no Bluetooth radio, so real
  hardware streaming must be validated on your own machine. The full UI and data
  pipeline are validated in Demo mode (`smoke_test.py`).
- Marker number-key hotkeys are application-wide while a session is open; avoid using
  the number keys while editing a numeric control if you do not intend to drop a marker.

## Project layout

```
ganglion_studio/
  main.py                 # entry point
  core/
    board_manager.py      # BrainFlow BoardShim wrapper, ring buffer, commands, recording
    board_config.py       # Ganglion ASCII command map, colours, impedance thresholds
    ble_scanner.py        # native BLE discovery (bleak)
    dsp.py                # filters, PSD, FFT, spectrogram, band powers, quality metrics
    session.py            # session config + recorder/exporter
  ui/
    main_window.py        # dashboard <-> session stack, async board prepare
    dashboard.py          # scan + session setup
    session_view.py       # toolbar, panels, plot tabs, refresh timer
    theme.py              # dark theme
    widgets/              # time_series, psd, spectrogram, impedance, band_power,
                          # accel, channel_panel, filter_panel, marker_panel
```

## Ideas / possible add-ons
- Session replay using BrainFlow's `PLAYBACK_FILE_BOARD`.
- Focus/relaxation metric via BrainFlow `MLModel`.
- Per-channel artifact rejection and automatic contact-quality scoring report.
- LSL output stream for integration with other acquisition tools.
