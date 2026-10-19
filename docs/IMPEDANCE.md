# Impedance acquisition and EEG timing

## Capture and review

On the Dashboard, choose **Capture → Impedance only**, select the board and
firmware, then start the session. The dedicated view shows the four input
channels and **REF**, their latest values in kΩ, measurement age, and an impedance
time series. Each graph point is a received measurement; refreshing the screen
does not create extra samples. Channel visibility and the plot window affect
only the display.

Use **Record** to capture changes over time and **Stop recording** to save.
Recordings live in a separate impedance session directory under `recordings/`:

- `*_impedance.csv`: one row per channel measurement, with its host timestamp,
  elapsed time, channel index/name, and resistance in Ω and kΩ.
- `*_meta.json`: channel mapping, session configuration, selected payload units,
  start/save times and observed measurement cadence.

Use **Open impedance recording…** on the Dashboard or in an impedance session to
inspect a saved CSV as a time series. Impedance is an asynchronous series of
channel readings, so it has no assumed EEG sampling rate. Equal consecutive
values are retained when they came from separate measurements. Missing values
remain missing rather than being written as zero or repeated at the screen's
refresh rate. Demo mode provides simulated impedance readings for trying the
workflow without a board.

## Timed checks during EEG recording

In an EEG session, start **Record**, choose the duration beside **Get impedance**
(default 10 seconds), then click the button. The app stops EEG, measures impedance
for the selected duration and resumes EEG automatically. The **Impedance** tab
opens with current values and a graph of received readings. **Resume EEG now**
ends a check early. The four transitions have reserved event codes:

| Event | Code | Meaning |
| --- | --- | --- |
| EEG_STOP | 90 | EEG streaming stopped |
| IMPEDANCE_START | 91 | Impedance measurement started |
| IMPEDANCE_END | 92 | Impedance measurement stopped |
| EEG_RESUME | 93 | EEG streaming successfully resumed |

The selected duration counts measurement time; switching modes takes additional
time. Markers use host times at acknowledged mode changes for the native driver,
not precise ADC timestamps. Short checks may finish before all five electrodes
report. There is no assumed one-second channel cadence.

The logical recording stays active. Manual and protocol markers retain their
times during the interruption. Missing EEG is represented by NaN samples, with
an `eeg_gaps` interval in metadata and a reserved `BAD_impedance_check` annotation
in exported recordings. The elapsed timeline remains intact, and these gaps are
separate from Bluetooth packet loss. Display filters and the processing pipeline
operate on continuous EEG sections without filtering across a gap. Analysis
reports require a continuous interval cropped in the recording editor.

The EEG session directory also contains `*_impedance.csv` and
`*_impedance_meta.json`, linked from its EEG metadata. **View impedance** in the
recording review opens the saved measurements. FIF preserves missing EEG values;
EDF stores zeros only in explicitly annotated gaps, which the app restores to
missing values when reopening.

If EEG cannot resume, the app keeps the interruption open and offers **Retry EEG
resume**. It records EEG_RESUME only after success. Leaving the session during a
check saves the captured EEG, gap, impedance readings and completed transitions;
it does not invent a resume event. Mode changes run in the background, and closing
waits for any in-progress transition before finalizing the recording.

## Payload units

Choose **Modern firmware (kΩ readings)** for the currently published 3.0.2
firmware, or **Legacy firmware (Ω readings)** for firmware that reports ohms.
The app converts either selection to canonical Ω and plots kΩ. The selection is
saved with the recording; values are never used to guess their own units.

This explicit choice is necessary because the public documentation describes
ohms, but the current firmware divides its calculated resistance by 1,000 before
sending it. BrainFlow retains the numeric payload rather than resolving that
unit discrepancy. Confirm the scale with a known resistor when validating a
particular board/firmware combination. [OpenBCI data format](https://docs.openbci.com/Ganglion/GanglionDataFormat/),
[firmware measurement implementation](https://github.com/OpenBCI/OpenBCI_Ganglion_Library/blob/master/OpenBCI_Ganglion_Library.cpp),
[BrainFlow Ganglion driver](https://github.com/brainflow-dev/brainflow/blob/master/src/board_controller/openbci/ganglion_native.cpp).

## Can EEG and impedance be interleaved at 100 Hz?

The stock protocol switches acquisition modes. `z` starts an impedance sweep,
stops EEG streaming, and repeatedly measures channels 1–4 plus REF; `Z` ends the
sweep. Dedicated sessions and timed checks keep these readings separate from EEG
samples. [OpenBCI impedance protocol](https://docs.openbci.com/Ganglion/GanglionDataFormat/#impedance-testing).

The documented sample-rate commands range from 200 Hz upward; there is no stock
100 Hz setting. Downsampling received EEG to 100 Hz can reduce stored data after
proper anti-alias filtering, but it does not alter the board's measurement mode
or create hardware time for impedance tests. [Ganglion sample-rate commands](https://docs.openbci.com/Ganglion/GanglionSDK/#sample-rate).

The published implementation measures one electrode at a time, using a test
current waveform and calibration. Its timing constants imply a lower bound of
roughly 0.7 seconds per electrode, or more than 3.5 seconds for all five, before
additional processing/communication delays. This is an inference from source,
not a measured guarantee; the app saves actual arrival times and observed
cadence. A full fresh set every second is therefore not supported by that
implementation. [Measurement loop](https://github.com/OpenBCI/OpenBCI_Ganglion_Library/blob/master/OpenBCI_Ganglion_Library.cpp),
[timing definitions](https://github.com/OpenBCI/OpenBCI_Ganglion_Library/blob/master/Definitions_Ganglion.h),
[calibration timing](https://github.com/OpenBCI/OpenBCI_Ganglion_Library/blob/master/OpenBCI_Ganglion_Library.h).

The timed button provides **EEG blocks → impedance check → EEG recovery**, with
explicit interruptions. It does not add a repeating schedule or a hidden
impedance sample between EEG samples. There is currently no extra settling
interval after EEG resumes. Establishing one and validating test-current artifacts
requires a real board and known resistors.

## Validation

Automated tests cover packet decoding, explicit unit scaling, asynchronous
channel updates, repeated values, four transition markers, timed resume, failed
resume and retry, closing during mode changes, persistence and crop/export
roundtrips. The headless demo smoke test covers EEG with a timed check and the
dedicated impedance workflow. Real-board measurement scale, cadence, mode
transitions and post-check settling still need hardware validation.
