# Codebase review — 2026-10-19

This review addressed concrete correctness and data-preservation issues before
new feature work. Existing uncommitted changes were preserved.

## Architecture

The application separates hardware, recording, processing and file handling in
`ganglion_studio/core/` from its PyQt6 interface in `ganglion_studio/ui/`.
`BoardManager` acquires BrainFlow data or data from the custom native BLE
subprocess, maintains a locked circular buffer for live views, and collects raw
recording chunks. `SessionRecorder` writes the CSV and metadata backups;
review/editor windows export research formats. Offline recordings pass through
`recording_loader.py` into the Processing Lab, whose background worker runs the
processing pipeline before analysis figures are generated. Display filters are
separate from recorded raw samples. See [ARCHITECTURE.md](ARCHITECTURE.md) for the
broader component map.

## Implemented fixes

- **Recording lifecycle:** session/app shutdown saves active recordings;
  declining session exit keeps the session open. Failed saves retain the captured
  samples for retry and prevent teardown. Recordings started within the same
  second reserve separate directories instead of overwriting files.
- **Native BLE decoding:** only exact supported counter wraps are continuous.
  Missing packets near a wrap now follow the loss-handling path.
- **Recording interchange:** signal channel types and saved placements survive
  the relevant export/reload paths. Cropped FIF annotations account for the
  recording's first sample offset. EDF export preserves voltage amplitudes for
  channels labelled MISC. Invalid or unreadable CSV metadata JSON raises the
  application's `LoadError` so callers can report the failure.
- **Processing correctness:** Chebyshev filters use SciPy second-order sections
  with positive ripple. An explicitly empty CAR channel selection stays empty;
  analysis refuses an empty channel selection. Processing results and errors
  from an earlier loaded recording are ignored using a recording-generation
  guard.
- **Protocol reliability:** unwritable preferences do not block protocol startup,
  and malformed saved parameters fall back to defaults. Startup respects a
  session's synchronous cancellation; shutdown resets the runner and controls
  without emitting a save-triggering stop signal. A pending recording save
  blocks new protocols and further additions to its frozen marker log.
- **Installation:** Matplotlib is now a core dependency. The application imports
  its analysis UI at startup, so the documented core-only installation needs it
  even when optional export/processing packages are omitted.

## Follow-up resolutions

1. **Marker preservation and timing:** native events use a locked timestamped
   queue. Recording finalization aligns every event to recorded host-notification
   timestamps and preserves collisions in a separate event list. The numeric raw
   marker row holds the first event at a sample for compatibility; the marker CSV
   now includes an authoritative `sample_index` column, and review/FIF export
   retain every event. Older CSVs without that column still use their raw marker
   row and label log. Pending events cannot leak into the next recording. Events
   after the last available sample map to that final sample; an empty recording
   retains its events in the log with a blank sample position.
2. **Worker shutdown coordination:** timed-out acquisition stops retain the live
   producer, and backend release waits for it. Qt windows retain preparation and
   processing workers until completion and defer closure without blocking the
   event loop. Session teardown quiesces the interface, joins acquisition and
   captures its final recording snapshot off the GUI thread before saving or
   releasing hardware. Failed saves leave a usable session for retry.
3. **Offline numerical precision:** import, processing, editing and export retain
   fractional sampling rates. Fractional-rate filters use SciPy rather than
   rounding rates for BrainFlow. PSD uses density-scaled SciPy Welch at all rates;
   canonical EEG band integration interpolates exact band boundaries. Tests
   validate known signal power, frequency axes and channel weighting. Absolute
   PSD values differ from earlier reports: the old BrainFlow Hann-window output
   underestimated power (a 3 uV sine measured 1.6875 instead of 4.5 uV² in the
   checked case). Relative bands also now use the advertised 0.5–4 Hz Delta band.
   File-format precision still applies: FIF sampling-rate headers use float32,
   so a value such as 199.8 Hz reloads near 199.800003 Hz rather than as an integer.
4. **Packet-loss provenance through FIF:** native files embed reserved
   `BAD_ganglion_packet_loss` annotations, separate from experimental markers.
   Gap starts and lengths survive export/reload without companion files; the
   editor and external MNE cropping preserve partial gaps at crop boundaries.
   Processing uses the saved lengths, while legacy CSV loss logs default to
   two samples per packet. Rebuilt FIFs also carry this provenance.

Marker times remain based on host notification arrival and host event clocks.
These fixes preserve events and their best available sample alignment; precise
hardware synchronization still requires measurements with the real board.

## Validation scope

The complete suite, including the subsequent impedance features, passes **310
tests**, and the full headless demo smoke test passes through recording, timed
impedance checks, impedance-only capture, protocols, export, processing and analysis. Regression
checks cover marker collisions and recording boundaries, slow-worker shutdown,
calibrated spectral power and fractional timing, and loss intervals through
standalone FIF files and cropped recordings. Real BLE hardware behavior was not
validated during this review.
