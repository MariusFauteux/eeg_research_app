"""EEG protocol timelines (blinks, eyes open/closed, resting state).

A *protocol* is a guided experiment the researcher runs during a recording: the
app cues each step (visually + audibly) and drops the matching annotation so the
data is labelled without any manual clicking.

This module is pure Python (no Qt) so the timelines are easy to unit-test. Each
builder turns a small parameter dataclass into an ordered list of
:class:`ProtocolStep` plus the total duration. The UI (``ui/panels/protocol_panel``)
walks that list with a timer, shows the pop-out cue window, plays a tone, and
inserts a marker whenever a step carries one.

Marker codes are reused from the manual Markers panel so protocol annotations and
hand-dropped ones share one vocabulary (see ``ui/panels/marker_panel.DEFAULT_MARKERS``):
Eyes Open = 1, Eyes Closed = 2, Movement = 4, Rest = 5. Blinks are annotated as
Movement (the eye/muscle movement of a blink), as requested.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Tuple

# Marker codes (kept in sync with ui/panels/marker_panel.DEFAULT_MARKERS). Defined
# here as plain ints so ``core`` never has to import anything from ``ui``.
MARK_EYES_OPEN = 1
MARK_EYES_CLOSED = 2
MARK_MOVEMENT = 4
MARK_REST = 5

# Protocol-specific codes (extend the shared 1..5 above). These are NOT added to the
# manual Markers panel (which is limited to hotkeys 1-9 and would get cluttered) --
# instead their labels live in PROTOCOL_MARKERS below, which SessionView merges into
# the review/export code->label map. Distinct codes keep conditions cleanly epochable.
MARK_STANDARD = 6
MARK_TARGET = 7
MARK_LOOK_LEFT = 8
MARK_LOOK_RIGHT = 9
MARK_LOOK_UP = 10
MARK_LOOK_DOWN = 11
MARK_TASK = 13
MARK_JAW_CLENCH = 14
MARK_SWALLOW = 15
MARK_TEETH_GRIND = 16
MARK_HEAD_TURN = 17
MARK_EYEBROW_RAISE = 18

PROTOCOL_MARKERS = {
    MARK_STANDARD: "Standard",
    MARK_TARGET: "Target",
    MARK_LOOK_LEFT: "Look Left",
    MARK_LOOK_RIGHT: "Look Right",
    MARK_LOOK_UP: "Look Up",
    MARK_LOOK_DOWN: "Look Down",
    MARK_TASK: "Task",
    MARK_JAW_CLENCH: "Jaw clench",
    MARK_SWALLOW: "Swallow",
    MARK_TEETH_GRIND: "Teeth grind",
    MARK_HEAD_TURN: "Head turn",
    MARK_EYEBROW_RAISE: "Eyebrow raise",
}

# Cued-movement / artifact actions the user can pick from (label -> marker code).
CUED_ACTIONS = {
    "Jaw clench": MARK_JAW_CLENCH,
    "Swallow": MARK_SWALLOW,
    "Teeth grind": MARK_TEETH_GRIND,
    "Head turn": MARK_HEAD_TURN,
    "Eyebrow raise": MARK_EYEBROW_RAISE,
}

# Eye-movement directions: (param attribute, marker label, cue text, marker code).
_EYE_DIRECTIONS = [
    ("left", "Look Left", "LOOK LEFT", MARK_LOOK_LEFT),
    ("right", "Look Right", "LOOK RIGHT", MARK_LOOK_RIGHT),
    ("up", "Look Up", "LOOK UP", MARK_LOOK_UP),
    ("down", "Look Down", "LOOK DOWN", MARK_LOOK_DOWN),
]


@dataclass
class ProtocolStep:
    """One cue in a protocol timeline.

    at            seconds from protocol start when this step becomes active
    instruction   big text shown in the cue window (e.g. "EYES CLOSED", "BLINK")
    state         short key used to pick the cue colour / behaviour, one of
                  "lead_in" | "blink" | "eyes_open" | "eyes_closed" | "rest" | "done"
    marker_code   annotation code to drop when this step fires (None = no marker)
    marker_label  human label for that marker (paired with marker_code)
    tone          audio cue to play on entry: "go" | "stop" | "tick" | "end" | None
    progress      one-line progress caption (e.g. "Set 2 / 3 · blink 1 / 5")
    fixation_cross  show a big steady "+" in the cue window for this step, so the
                  subject has something to fixate on (keeps the eyes still)
    """

    at: float
    instruction: str
    state: str
    marker_code: Optional[int] = None
    marker_label: Optional[str] = None
    tone: Optional[str] = None
    progress: str = ""
    fixation_cross: bool = False


# --------------------------------------------------------------------------- #
# Parameters (one dataclass per protocol; defaults double as the UI defaults)
# --------------------------------------------------------------------------- #
@dataclass
class BlinkParams:
    lead_in_s: float = 5.0        # quiet time before the first blink
    blinks_per_set: int = 5       # number of blinks in one set
    sets: int = 3                 # number of sets in the recording
    inter_set_s: float = 10.0     # rest between the end of a set and the next set
    inter_blink_s: float = 2.0    # spacing between blinks within a set


@dataclass
class EyesParams:
    conditions: int = 2               # how many alternating EO/EC blocks
    seconds_per_condition: float = 60.0
    start_with: str = "EO"            # "EO" (eyes open first) or "EC"
    lead_in_s: float = 5.0            # delayed start: quiet time before the first block
    fixation_cross: bool = True       # show a "+" to fixate on during Eyes Open


@dataclass
class RestParams:
    duration_s: float = 300.0     # length of the rest block
    lead_in_s: float = 5.0        # quiet "get ready" time before rest starts


@dataclass
class OddballParams:
    n_trials: int = 100               # total stimuli
    target_prob: float = 0.2          # fraction that are rare "target" tones
    isi_s: float = 1.0                # base inter-stimulus interval
    jitter_s: float = 0.2             # +/- random jitter added to each ISI
    lead_in_s: float = 5.0
    min_standards_before_target: int = 2   # standards required before/between targets
    seed: int = 0                     # 0 = fresh random each run; else reproducible


@dataclass
class EyeMovementParams:
    left: bool = True
    right: bool = True
    up: bool = True
    down: bool = True
    hold_s: float = 2.0               # hold each look
    center_s: float = 2.0             # return-to-centre (baseline) between looks
    repeats: int = 5                  # cycles through the enabled directions
    lead_in_s: float = 5.0
    randomize: bool = False           # shuffle the direction order each cycle


@dataclass
class CuedMovementParams:
    action: str = "Jaw clench"        # which artifact/movement to cue (see CUED_ACTIONS)
    lead_in_s: float = 5.0
    reps_per_set: int = 5
    sets: int = 3
    inter_set_s: float = 10.0
    inter_rep_s: float = 3.0


@dataclass
class CognitiveParams:
    blocks: int = 4                   # alternating rest/task blocks
    seconds_per_block: float = 30.0
    start_with: str = "REST"          # "REST" or "TASK"
    task_label: str = "Serial-7 subtraction"


# --------------------------------------------------------------------------- #
# Builders: params -> (ordered steps, total seconds)
# --------------------------------------------------------------------------- #
def build_blinks(p: BlinkParams) -> Tuple[List[ProtocolStep], float]:
    """Lead-in, then ``sets`` sets of ``blinks_per_set`` blinks each.

    Every blink drops a Movement marker and cues a "go" tone; a short rest prompt
    is shown between sets.
    """
    steps: List[ProtocolStep] = [
        ProtocolStep(0.0, "Get ready…", "lead_in", progress="Blinks — starting")
    ]
    t = float(p.lead_in_s)
    for s in range(p.sets):
        for k in range(p.blinks_per_set):
            steps.append(
                ProtocolStep(
                    at=t,
                    instruction="BLINK",
                    state="blink",
                    marker_code=MARK_MOVEMENT,
                    marker_label="Movement",
                    tone="go",
                    progress=f"Set {s + 1} / {p.sets} · blink {k + 1} / {p.blinks_per_set}",
                )
            )
            if k < p.blinks_per_set - 1:
                t += p.inter_blink_s
        # ``t`` now sits on the last blink of this set.
        if s < p.sets - 1:
            rest_at = t + min(1.0, p.inter_set_s * 0.5)  # let the last BLINK show briefly
            steps.append(
                ProtocolStep(
                    at=rest_at,
                    instruction="Rest — next set coming up",
                    state="rest",
                    tone="tick",
                    progress=f"Finished set {s + 1} / {p.sets}",
                )
            )
            t += p.inter_set_s  # advance to the first blink of the next set
    total = t + 1.5
    steps.append(ProtocolStep(total, "Done", "done", tone="end", progress="Protocol complete"))
    return steps, total


def build_eyes(p: EyesParams) -> Tuple[List[ProtocolStep], float]:
    """Alternating Eyes-Open / Eyes-Closed blocks, one marker per block onset.

    ``lead_in_s`` is a delayed start: a quiet "get ready" step at t=0 that pushes the
    first block (and every block after it) back by that many seconds, so the
    researcher has time to settle the subject / step out of view. Eyes-Open blocks
    carry ``fixation_cross`` so the cue window shows a steady "+" to stare at.
    """
    order = ("EO", "EC") if p.start_with == "EO" else ("EC", "EO")
    steps: List[ProtocolStep] = []
    if p.lead_in_s > 0:
        steps.append(ProtocolStep(0.0, "Get ready…", "lead_in",
                                  progress="Eyes Open / Closed — starting"))
    for i in range(p.conditions):
        at = float(p.lead_in_s) + i * float(p.seconds_per_condition)
        if order[i % 2] == "EO":
            steps.append(
                ProtocolStep(at, "EYES OPEN", "eyes_open", MARK_EYES_OPEN, "Eyes Open", "go",
                             progress=f"Block {i + 1} / {p.conditions}",
                             fixation_cross=p.fixation_cross)
            )
        else:
            steps.append(
                ProtocolStep(at, "EYES CLOSED", "eyes_closed", MARK_EYES_CLOSED, "Eyes Closed", "stop",
                             progress=f"Block {i + 1} / {p.conditions}")
            )
    total = float(p.lead_in_s + p.conditions * p.seconds_per_condition)
    steps.append(ProtocolStep(total, "Done", "done", tone="end", progress="Protocol complete"))
    return steps, total


def build_rest(p: RestParams) -> Tuple[List[ProtocolStep], float]:
    """A single fixed-duration rest block with a Rest marker at its onset."""
    steps: List[ProtocolStep] = []
    if p.lead_in_s > 0:
        steps.append(ProtocolStep(0.0, "Get ready…", "lead_in", progress="Rest — starting"))
    steps.append(
        ProtocolStep(float(p.lead_in_s), "REST — stay still", "rest",
                     MARK_REST, "Rest", "go", progress="Resting")
    )
    total = float(p.lead_in_s + p.duration_s)
    steps.append(ProtocolStep(total, "Done", "done", tone="end", progress="Protocol complete"))
    return steps, total


def _oddball_sequence(p: OddballParams, rng: random.Random) -> List[str]:
    """A list of "standard"/"target" of length n_trials.

    Targets are ~``target_prob`` of trials, never appear before
    ``min_standards_before_target`` standards, and are spaced so at least that many
    standards separate consecutive targets.
    """
    gap = p.min_standards_before_target
    candidates = list(range(gap, p.n_trials))  # earliest a target may appear is index ``gap``
    rng.shuffle(candidates)
    n_targets = round(p.n_trials * p.target_prob)
    chosen: List[int] = []
    for pos in candidates:
        if len(chosen) >= n_targets:
            break
        if all(abs(pos - c) > gap for c in chosen):  # keep >= gap standards between targets
            chosen.append(pos)
    picked = set(chosen)
    return ["target" if i in picked else "standard" for i in range(p.n_trials)]


def build_oddball(p: OddballParams) -> Tuple[List[ProtocolStep], float]:
    """Auditory oddball (P300): frequent standards + rare targets, count the targets.

    The cue window shows a constant fixation ("+") so the target is revealed only by
    the tone, never visually. Each stimulus drops a Standard/Target marker; ISI is
    jittered. Onset timing is software-scheduled (fine for the late P300, not for
    precise early components).
    """
    rng = random.Random(p.seed or None)
    seq = _oddball_sequence(p, rng)
    steps: List[ProtocolStep] = [
        ProtocolStep(0.0, "Get ready — silently count the high (target) beeps",
                     "lead_in", progress="Oddball — starting")
    ]
    t = float(p.lead_in_s)
    for i, kind in enumerate(seq):
        if kind == "target":
            steps.append(ProtocolStep(t, "+", "fixation", MARK_TARGET, "Target", "target",
                                      progress=f"Trial {i + 1} / {p.n_trials}"))
        else:
            steps.append(ProtocolStep(t, "+", "fixation", MARK_STANDARD, "Standard", "standard",
                                      progress=f"Trial {i + 1} / {p.n_trials}"))
        t += max(0.05, p.isi_s + rng.uniform(-p.jitter_s, p.jitter_s))
    total = t
    steps.append(ProtocolStep(total, "Done — how many did you count?", "done",
                              tone="end", progress="Protocol complete"))
    return steps, total


def build_eye_movements(p: EyeMovementParams) -> Tuple[List[ProtocolStep], float]:
    """Cue look left/right/up/down with a return-to-centre baseline between each."""
    dirs = [d for d in _EYE_DIRECTIONS if getattr(p, d[0])]
    steps: List[ProtocolStep] = [
        ProtocolStep(0.0, "Get ready…", "lead_in", progress="Eye movements — starting")
    ]
    t = float(p.lead_in_s)
    rng = random.Random() if p.randomize else None
    for r in range(p.repeats):
        order = list(dirs)
        if rng is not None:
            rng.shuffle(order)
        for _attr, label, cue, code in order:
            steps.append(ProtocolStep(t, cue, "look", code, label, "go",
                                      progress=f"Repeat {r + 1} / {p.repeats} · {label}"))
            t += p.hold_s
            steps.append(ProtocolStep(t, "CENTRE", "center", tone="tick",
                                      progress=f"Repeat {r + 1} / {p.repeats} · back to centre"))
            t += p.center_s
    total = t
    steps.append(ProtocolStep(total, "Done", "done", tone="end", progress="Protocol complete"))
    return steps, total


def build_cued_movement(p: CuedMovementParams) -> Tuple[List[ProtocolStep], float]:
    """Cue a repeated movement/artifact (jaw clench, swallow, ...) in sets.

    Generalises :func:`build_blinks`: one Movement-style marker per cued action so the
    recording is labelled with a clean artifact library.
    """
    code = CUED_ACTIONS.get(p.action, MARK_MOVEMENT)
    cue = p.action.upper()
    steps: List[ProtocolStep] = [
        ProtocolStep(0.0, "Get ready…", "lead_in", progress=f"{p.action} — starting")
    ]
    t = float(p.lead_in_s)
    for s in range(p.sets):
        for k in range(p.reps_per_set):
            steps.append(ProtocolStep(t, cue, "action", code, p.action, "go",
                                      progress=f"Set {s + 1} / {p.sets} · {k + 1} / {p.reps_per_set}"))
            if k < p.reps_per_set - 1:
                t += p.inter_rep_s
        if s < p.sets - 1:
            rest_at = t + min(1.0, p.inter_set_s * 0.5)
            steps.append(ProtocolStep(rest_at, "Rest — next set coming up", "rest", tone="tick",
                                      progress=f"Finished set {s + 1} / {p.sets}"))
            t += p.inter_set_s
    total = t + 1.5
    steps.append(ProtocolStep(total, "Done", "done", tone="end", progress="Protocol complete"))
    return steps, total


def build_cognitive(p: CognitiveParams) -> Tuple[List[ProtocolStep], float]:
    """Alternating Rest / active-task blocks (e.g. mental arithmetic)."""
    order = ("REST", "TASK") if p.start_with == "REST" else ("TASK", "REST")
    steps: List[ProtocolStep] = []
    for i in range(p.blocks):
        at = i * float(p.seconds_per_block)
        if order[i % 2] == "REST":
            steps.append(ProtocolStep(at, "REST", "rest", MARK_REST, "Rest", "stop",
                                      progress=f"Block {i + 1} / {p.blocks}"))
        else:
            steps.append(ProtocolStep(at, p.task_label, "task", MARK_TASK, "Task", "go",
                                      progress=f"Block {i + 1} / {p.blocks}"))
    total = float(p.blocks * p.seconds_per_block)
    steps.append(ProtocolStep(total, "Done", "done", tone="end", progress="Protocol complete"))
    return steps, total


# --------------------------------------------------------------------------- #
# Registry: one entry per protocol so the panel and tests enumerate in one place
# --------------------------------------------------------------------------- #
@dataclass
class Protocol:
    key: str                 # stable id used for saving params
    label: str               # shown in the protocol dropdown
    params_cls: type         # the parameter dataclass
    build: Callable          # build(params) -> (steps, total_s)


# NOTE: the panel builds one config page per entry in THIS order and maps the
# dropdown index straight to the page stack, so keep panel pages in the same order.
PROTOCOLS: List[Protocol] = [
    Protocol("blinks", "Blinks", BlinkParams, build_blinks),
    Protocol("eyes", "Eyes Open / Closed", EyesParams, build_eyes),
    Protocol("rest", "Resting state", RestParams, build_rest),
    Protocol("oddball", "Auditory oddball (P300)", OddballParams, build_oddball),
    Protocol("eyes_move", "Guided eye movements", EyeMovementParams, build_eye_movements),
    Protocol("cued", "Cued movement / artifact", CuedMovementParams, build_cued_movement),
    Protocol("cognitive", "Cognitive task blocks", CognitiveParams, build_cognitive),
]

_BY_KEY = {p.key: p for p in PROTOCOLS}


def get(key: str) -> Protocol:
    """Look up a protocol by key (raises KeyError if unknown)."""
    return _BY_KEY[key]


def default_params(key: str) -> dict:
    """Default parameter values for a protocol, as a plain dict."""
    return asdict(_BY_KEY[key].params_cls())


def build_steps(key: str, params) -> Tuple[List[ProtocolStep], float]:
    """Build the timeline for ``key`` from a params dataclass instance."""
    return _BY_KEY[key].build(params)
