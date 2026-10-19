"""Reserved transition markers for temporary impedance checks inside EEG."""

EEG_STOP = 90
IMPEDANCE_START = 91
IMPEDANCE_END = 92
EEG_RESUME = 93

CHECK_MARKERS = {
    EEG_STOP: "EEG_STOP",
    IMPEDANCE_START: "IMPEDANCE_START",
    IMPEDANCE_END: "IMPEDANCE_END",
    EEG_RESUME: "EEG_RESUME",
}
