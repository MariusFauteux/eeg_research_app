"""Tests for the pure-Python protocol timeline builders."""

from __future__ import annotations

from ganglion_studio.core import protocols


def test_blinks_timeline():
    p = protocols.BlinkParams(lead_in_s=5, blinks_per_set=3, sets=2,
                              inter_set_s=10, inter_blink_s=2)
    steps, total = protocols.build_blinks(p)

    blinks = [s for s in steps if s.marker_code == protocols.MARK_MOVEMENT]
    assert len(blinks) == 3 * 2                      # blinks_per_set * sets
    assert all(s.marker_label == "Movement" for s in blinks)
    assert blinks[0].at == 5                         # first blink after the lead-in
    assert [s.at for s in blinks] == [5, 7, 9, 19, 21, 23]

    ats = [s.at for s in steps]
    assert ats == sorted(ats)                        # monotonically increasing
    assert total == 24.5
    assert steps[-1].state == "done"


def test_eyes_alternates_and_marks_blocks():
    p = protocols.EyesParams(conditions=4, seconds_per_condition=30, start_with="EO",
                             lead_in_s=0)
    steps, total = protocols.build_eyes(p)

    blocks = [s for s in steps if s.marker_code is not None]
    assert [s.at for s in blocks] == [0, 30, 60, 90]
    assert [s.marker_code for s in blocks] == [
        protocols.MARK_EYES_OPEN, protocols.MARK_EYES_CLOSED,
        protocols.MARK_EYES_OPEN, protocols.MARK_EYES_CLOSED,
    ]
    assert total == 120


def test_eyes_can_start_with_closed():
    p = protocols.EyesParams(conditions=2, seconds_per_condition=10, start_with="EC")
    steps, _ = protocols.build_eyes(p)
    blocks = [s for s in steps if s.marker_code is not None]
    assert [s.marker_code for s in blocks] == [
        protocols.MARK_EYES_CLOSED, protocols.MARK_EYES_OPEN]


def test_eyes_delayed_start_offsets_every_block():
    p = protocols.EyesParams(conditions=3, seconds_per_condition=20, start_with="EO",
                             lead_in_s=10)
    steps, total = protocols.build_eyes(p)
    assert steps[0].at == 0 and steps[0].state == "lead_in"   # "get ready" comes first
    assert steps[0].marker_code is None                       # and is not annotated
    blocks = [s for s in steps if s.marker_code is not None]
    assert [s.at for s in blocks] == [10, 30, 50]
    assert total == 70


def test_eyes_fixation_cross_only_on_eyes_open():
    on, _ = protocols.build_eyes(protocols.EyesParams(conditions=2, fixation_cross=True))
    assert [s.fixation_cross for s in on if s.state == "eyes_open"] == [True]
    assert [s.fixation_cross for s in on if s.state == "eyes_closed"] == [False]

    off, _ = protocols.build_eyes(protocols.EyesParams(conditions=2, fixation_cross=False))
    assert not any(s.fixation_cross for s in off)


def test_rest_marks_onset_after_lead_in():
    p = protocols.RestParams(duration_s=300, lead_in_s=5)
    steps, total = protocols.build_rest(p)
    rest = [s for s in steps if s.marker_code == protocols.MARK_REST]
    assert len(rest) == 1
    assert rest[0].at == 5
    assert total == 305


def test_registry_dispatch_and_defaults():
    # Every registered protocol builds a non-empty timeline from its defaults.
    for proto in protocols.PROTOCOLS:
        params = proto.params_cls()
        steps, total = protocols.build_steps(proto.key, params)
        assert steps and total > 0
        assert steps[-1].state == "done"
    assert set(protocols.default_params("blinks")) == {
        "lead_in_s", "blinks_per_set", "sets", "inter_set_s", "inter_blink_s"}


def test_oddball_sequence_respects_constraints():
    p = protocols.OddballParams(n_trials=100, target_prob=0.2, isi_s=1.0, jitter_s=0.2,
                                lead_in_s=5, min_standards_before_target=2, seed=42)
    steps, total = protocols.build_oddball(p)

    stimuli = [s for s in steps
               if s.marker_code in (protocols.MARK_STANDARD, protocols.MARK_TARGET)]
    assert len(stimuli) == 100
    codes = [s.marker_code for s in stimuli]
    tgt_idx = [i for i, c in enumerate(codes) if c == protocols.MARK_TARGET]
    assert 14 <= len(tgt_idx) <= 20               # ~20% under the spacing constraint
    assert min(tgt_idx) >= 2                       # no target before min standards
    assert all(b - a > 2 for a, b in zip(tgt_idx, tgt_idx[1:]))  # spacing between targets
    ats = [s.at for s in steps]
    assert ats == sorted(ats)
    assert total > p.lead_in_s


def test_oddball_seed_is_reproducible():
    p = protocols.OddballParams(n_trials=40, seed=123)
    a = [s.marker_code for s in protocols.build_oddball(p)[0]]
    b = [s.marker_code for s in protocols.build_oddball(p)[0]]
    assert a == b


def test_eye_movements_fixed_order():
    p = protocols.EyeMovementParams(left=True, right=True, up=False, down=False,
                                    hold_s=2, center_s=1, repeats=2, lead_in_s=3,
                                    randomize=False)
    steps, _ = protocols.build_eye_movements(p)
    looks = [s for s in steps if s.state == "look"]
    assert [s.marker_code for s in looks] == [
        protocols.MARK_LOOK_LEFT, protocols.MARK_LOOK_RIGHT] * 2
    assert looks[0].at == 3                        # first look after the lead-in
    centers = [s for s in steps if s.state == "center"]
    assert len(centers) == len(looks)             # a centre baseline after each look
    ats = [s.at for s in steps]
    assert ats == sorted(ats)


def test_cued_movement_uses_selected_action():
    p = protocols.CuedMovementParams(action="Swallow", lead_in_s=2, reps_per_set=3,
                                     sets=2, inter_set_s=5, inter_rep_s=1)
    steps, _ = protocols.build_cued_movement(p)
    actions = [s for s in steps if s.state == "action"]
    assert len(actions) == 3 * 2
    assert all(s.marker_code == protocols.MARK_SWALLOW and s.marker_label == "Swallow"
               for s in actions)


def test_cognitive_alternates_and_can_start_with_task():
    p = protocols.CognitiveParams(blocks=4, seconds_per_block=30, start_with="REST")
    steps, total = protocols.build_cognitive(p)
    blocks = [s for s in steps if s.marker_code is not None]
    assert [s.at for s in blocks] == [0, 30, 60, 90]
    assert [s.marker_code for s in blocks] == [
        protocols.MARK_REST, protocols.MARK_TASK, protocols.MARK_REST, protocols.MARK_TASK]
    assert total == 120

    q = protocols.CognitiveParams(blocks=2, seconds_per_block=10, start_with="TASK")
    first_two = [s.marker_code for s in protocols.build_cognitive(q)[0] if s.marker_code]
    assert first_two == [protocols.MARK_TASK, protocols.MARK_REST]


def test_all_emitted_marker_codes_are_registered():
    # Every code any builder emits must be a shared code (1..5) or in PROTOCOL_MARKERS,
    # so SessionView's merged code->label map can always name it in review/exports.
    known = set(range(1, 6)) | set(protocols.PROTOCOL_MARKERS)
    for proto in protocols.PROTOCOLS:
        steps, _ = protocols.build_steps(proto.key, proto.params_cls())
        emitted = {s.marker_code for s in steps if s.marker_code is not None}
        assert emitted <= known, (proto.key, emitted - known)
