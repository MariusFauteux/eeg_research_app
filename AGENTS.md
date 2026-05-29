# AGENTS.md

## Cursor Cloud specific instructions

### Product

Single-process **desktop app** (PyQt6 + BrainFlow). No web server, database, Docker, or separate backend. End-to-end dev/test is **Demo mode** (synthetic board); real OpenBCI Ganglion over BLE requires hardware and is not available in typical cloud VMs.

### Dependencies

- **Python 3.10+** with `pip install -r requirements.txt` from repo root.
- **Linux GUI** (when not using headless smoke test): besides packages in README, install `libxcb-cursor0` and related XCB libs if Qt fails to load the `xcb` platform plugin (`qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`). Example: `sudo apt-get install -y libegl1 libgl1 libxkbcommon0 libdbus-1-3 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxkbcommon-x11-0`.

### Commands (see README)

| Action | Command |
|--------|---------|
| Headless validation | `python3 smoke_test.py` (sets `QT_QPA_PLATFORM=offscreen`) |
| Run GUI | `python3 -m ganglion_studio.main` |
| Lint / unit tests | Not configured in repo |

### Running the GUI in Cloud Agent VMs

- Use a display (`DISPLAY` is usually set). Do **not** set `QT_QPA_PLATFORM=offscreen` for interactive runs.
- **Demo mode** on the dashboard: check “Demo mode (synthetic board, no hardware)”, enter a session name, click **Start Session**. No Bluetooth required.
- `smoke_test.py` is the reliable CI-style check; it exercises board streaming, all plot tabs, filters, impedance, and recording without a display.

### Gotchas

- Scripts launched outside the repo root need `PYTHONPATH=/workspace` (or run from `/workspace`) so `ganglion_studio` imports resolve.
- Pip installs to `~/.local/bin`; extend `PATH` if you need CLI tools like `mne` in the shell.
- After `pip install`, no extra service startup is required—only the desktop process.
