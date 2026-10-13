"""core.hold: the proprioceptive hold detector on synthetic jaw-gap traces (MS7c design section 5)."""

import pytest

from maker_arm_lab.core import hold
from maker_arm_lab.robots.frames import JAW_FULL_GAP

W = 0.0158  # m, a 2x4's width: the jaws stall here
CLOSE = W - 0.006  # m, the teacher's close command, grasp.close_gap(W)
LANDING = W + 0.008  # m, the teacher's landing gap
RAMP = 2 * 0.1 / 30  # m per step, both jaws at the env's 0.1 m/s ramp


def _ramp(start: float, stop: float) -> list[float]:
    """The measured gap moving from ``start`` to ``stop`` at the env's jaw ramp."""
    gaps, g = [], start
    while abs(g - stop) > 1e-12:
        g = max(stop, g - RAMP) if stop < g else min(stop, g + RAMP)
        gaps.append(g)
    return gaps


def _feed(d, steps):
    """Feed (measured, commanded, pick_z) rows; return each step's (holding, lifted, events)."""
    out = []
    for measured, commanded, z in steps:
        d.update(measured, commanded, z)
        out.append((d.holding, d.lifted, d.started, d.lift_started, d.ended, d.lost))
    return out


def _grasp(z=0.02):
    """Open and commanded open, then the close: the jaws ramp down and stall on the part."""
    return [(LANDING, LANDING, z)] * 3 + [(g, CLOSE, z) for g in _ramp(LANDING, W)]


def test_a_close_on_nothing_converges_to_the_command_and_never_holds():
    d = hold.HoldDetector()
    for command in (CLOSE, 0.010, 0.0):
        settled = [(command, command, 0.02)] * 10
        steps = [(g, command, 0.02) for g in _ramp(LANDING, command)] + settled
        assert not any(row[0] for row in _feed(d, steps)), command


def test_the_closing_ramp_itself_never_reads_as_a_hold():
    # A long ramp with a large excess on every step: only the stability condition stops it.
    d = hold.HoldDetector()
    steps = [(g, 0.0, 0.02) for g in _ramp(JAW_FULL_GAP, 0.004)]
    assert len(steps) > 10
    assert not any(row[0] for row in _feed(d, steps))


def test_a_stall_on_a_part_holds_on_the_third_stable_step():
    d = hold.HoldDetector()
    steps = _grasp()
    stall = len(steps) - 1  # the first step at the stall gap
    rows = _feed(d, steps + [(W, CLOSE, 0.02)] * 6)
    first = next(t for t, row in enumerate(rows) if row[0])
    # The window of STABLE_STEPS readings is still at the stall STABLE_STEPS - 1 steps after
    # it, and CONFIRM_STEPS stable steps in a row start the hold.
    assert first == stall + (hold.STABLE_STEPS - 1) + (hold.CONFIRM_STEPS - 1)
    assert rows[first][2] and sum(row[2] for row in rows) == 1
    assert not any(row[1] for row in rows)  # no lift without a rise


def test_a_lift_latches_and_a_slip_in_the_carry_is_lost():
    d = hold.HoldDetector()
    _feed(d, _grasp() + [(W, CLOSE, 0.02)] * 6)
    assert d.holding and d.start_z == 0.02
    carry = [(W, CLOSE, 0.02 + 0.011 * k) for k in range(1, 8)]
    rows = _feed(d, carry)
    lifted_at = next(k for k, row in enumerate(rows) if row[1])
    assert carry[lifted_at][2] - 0.02 >= hold.LIFT_MIN_RISE > carry[lifted_at - 1][2] - 0.02
    assert rows[lifted_at][3] and sum(row[3] for row in rows) == 1
    assert all(row[1] for row in rows[lifted_at:])
    # The part slips out: the jaws close on to the command, the hold ends after EXIT_STEPS.
    rows = _feed(d, [(g, CLOSE, 0.10) for g in _ramp(W, CLOSE)] + [(CLOSE, CLOSE, 0.10)] * 3)
    ended = [t for t, row in enumerate(rows) if row[4]]
    assert len(ended) == 1 and rows[ended[0]][5]  # lost: it ended while lifted
    assert not d.holding and not d.lifted


def test_a_one_step_wobble_in_the_carry_keeps_the_hold_and_two_end_it():
    d = hold.HoldDetector()
    _feed(d, _grasp() + [(W, CLOSE, 0.02)] * 6)
    rows = _feed(d, [(W, CLOSE, 0.03), (CLOSE, CLOSE, 0.03), (W, CLOSE, 0.03), (W, CLOSE, 0.03)])
    assert all(row[0] for row in rows) and not any(row[4] for row in rows)
    rows = _feed(d, [(CLOSE, CLOSE, 0.03)] * 2)
    assert rows[0][0] and rows[1][4] and not rows[1][5]  # ended on the ground: not lost
    assert not d.holding


def test_the_binary_gripper_maps_to_the_closed_and_the_full_gap():
    assert hold.commanded_gap(-1.0, "binary") == 0.0
    assert hold.commanded_gap(0.0, "binary") == JAW_FULL_GAP
    assert hold.commanded_gap(1.0, "binary") == JAW_FULL_GAP
    assert hold.commanded_gap(0.02, "continuous") == 0.02
    assert hold.commanded_gap(-0.01, "continuous") == 0.0
    assert hold.commanded_gap(0.5, "continuous") == JAW_FULL_GAP
    # A binary close on a part: the jaws stall at its width, the command reads 0.
    d = hold.HoldDetector()
    closed = hold.commanded_gap(-1.0, "binary")
    steps = [(g, closed, 0.02) for g in _ramp(JAW_FULL_GAP, W)] + [(W, closed, 0.02)] * 6
    assert any(row[0] for row in _feed(d, steps))


def test_reset_forgets_the_hold_and_the_window():
    d = hold.HoldDetector()
    _feed(d, _grasp() + [(W, CLOSE, 0.02)] * 6)
    d.reset()
    assert not d.holding and len(d.gaps) == 0
    rows = _feed(d, [(W, CLOSE, 0.02)] * (hold.STABLE_STEPS + hold.CONFIRM_STEPS - 1))
    assert not rows[-2][0] and rows[-1][0]


def _row(held=0, rise=0.0, seen=False, lifted=False, at=None, truth_at=None):
    return {
        "held_steps": held,
        "max_rise": rise,
        "hold_detected": seen,
        "lift_detected": lifted,
        "first_hold_at": at,
        "first_held_at": truth_at,
    }


def test_agreement_scores_both_events_and_the_offset():
    n = hold.TRUTH_MIN_HELD_STEPS
    rows = [
        _row(n, 0.06, True, True, 30, 25),  # a stow: both events agree, offset 5
        _row(n, 0.01, True, False, 40, 36),  # held, not lifted, offset 4
        _row(n - 1, 0.0, True, False, 12, 10),  # a brush the detector reads as a hold
        _row(n, 0.06, True, False, 50, 44),  # a lift the detector missed, offset 6
        _row(0, 0.0, False, False),  # a close on nothing
    ]
    s = hold.agreement(rows)
    assert s["cycles"] == 5 and s["offset_cycles"] == 4
    assert s["hold"] == {"tp": 3, "fp": 1, "fn": 0, "tn": 1, "precision": 0.75, "recall": 1.0}
    assert s["lift"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 3, "precision": 1.0, "recall": 0.5}
    assert s["first_hold_offset_median"] == 4.5
    assert not hold.meets_bar(s)
    assert hold.meets_bar(hold.agreement([rows[0], rows[1], rows[4]]))
    # An offset over the bar fails on its own; so does a run with no lift at all.
    late = _row(n, 0.06, True, True, 37, 25)
    assert not hold.meets_bar(hold.agreement([late, rows[1], rows[4]]))
    assert not hold.meets_bar(hold.agreement([rows[1], rows[4]]))
    assert hold.agreement([])["first_hold_offset_median"] is None


def test_cycle_rows_rebuild_the_census_from_a_trace():
    # env 0: two steps between cycles, then cycle 1 with a held part that rises 7.6 cm.
    trace = [[0, s, -1, LANDING, LANDING, 0.02, -1, 0.0] for s in range(2)]
    step = 2
    for g in _ramp(LANDING, W):
        trace.append([0, step, 1, g, CLOSE, 0.02, -1, 0.0])
        step += 1
    first_held = step
    for k in range(20):
        z = 0.02 + 0.004 * k
        trace.append([0, step, 1, W, CLOSE, z, 3, z - 0.02])
        step += 1
    rows = hold.cycle_rows(trace)
    assert len(rows) == 1
    row = rows[0]
    assert row["held_steps"] == 20 and row["max_rise"] == round(0.004 * 19, 4)
    assert row["first_held_at"] == first_held - 2
    assert row["hold_detected"] and row["lift_detected"]
    # The jaws stall at step 3 (cycle step 1) and the contact holds from step 4; the hold
    # starts STABLE_STEPS - 1 + CONFIRM_STEPS - 1 steps after the stall.
    assert row["first_hold_at"] == 1 + hold.STABLE_STEPS - 1 + hold.CONFIRM_STEPS - 1
    # A threshold override replays a different detector.
    quick = hold.cycle_rows(trace, stable_steps=1, confirm_steps=1)[0]
    assert quick["first_hold_at"] < row["first_hold_at"]
    assert not hold.cycle_rows(trace, min_excess=0.01)[0]["hold_detected"]


def test_the_bar_constants_are_the_pre_registered_rule():
    assert (hold.BAR_PRECISION, hold.BAR_RECALL, hold.BAR_OFFSET_STEPS) == (0.95, 0.95, 5)
    assert hold.TRUTH_MIN_HELD_STEPS == 10 and hold.LIFT_MIN_RISE == pytest.approx(0.05)


def _subtasks(frames, width=W):
    """Run ``subtask_step`` over (gap, prev_cmd, pick_z) frames from a cycle's start."""
    ids, lifted, low = [], False, None
    for gap, cmd, z in frames:
        sid, lifted, low = hold.subtask_step(lifted, low, gap, cmd, z, width)
        ids.append(hold.SUBTASKS[sid])
    return ids


def test_subtask_reads_reach_while_open_and_grasp_once_the_command_closes():
    assert hold.SUBTASKS == ("reach", "grasp", "carry", "release")
    frames = [(LANDING, LANDING, 0.02)] * 3 + [(LANDING, CLOSE, 0.02)]
    assert _subtasks(frames) == ["reach"] * 3 + ["grasp"]
    # No previous command reads open: a corpus episode's first frame.
    assert _subtasks([(W, None, 0.02)]) == ["reach"]


def test_a_rise_with_empty_jaws_stays_grasp():
    # The jaws close on nothing (the gap converges to the command) and the arm goes up 6 cm.
    frames = [(g, CLOSE, 0.02) for g in _ramp(LANDING, CLOSE)]
    frames += [(CLOSE, CLOSE, 0.02 + 0.01 * k) for k in range(1, 7)]
    assert set(_subtasks(frames)) == {"grasp"}


def test_blocked_jaws_and_a_5_cm_rise_over_the_lowest_point_read_carry_then_release():
    # The close starts at 3 cm, the arm presses down to 1 cm, then lifts: the rise counts
    # from 1 cm, so 5.5 cm reads carry and 5.0 cm does not yet (pick z 0.059 < 0.01 + 0.05).
    frames = [(LANDING, CLOSE, 0.03), (W, CLOSE, 0.01), (W, CLOSE, 0.059), (W, CLOSE, 0.065)]
    assert _subtasks(frames) == ["grasp", "grasp", "grasp", "carry"]
    frames += [(W, LANDING, 0.065), (LANDING, CLOSE, 0.065)]  # an open, then a new close
    assert _subtasks(frames)[-2:] == ["release", "carry"]


def test_a_reopen_before_the_lift_resets_the_lowest_point():
    # Pressed to 1 cm, re-opened, then closed again at 3 cm: the rise counts from 3 cm.
    frames = [(W, CLOSE, 0.01), (LANDING, LANDING, 0.03), (W, CLOSE, 0.03), (W, CLOSE, 0.065)]
    assert _subtasks(frames) == ["grasp", "reach", "grasp", "grasp"]
    assert _subtasks(frames + [(W, CLOSE, 0.081)])[-1] == "carry"


def test_a_zero_width_reads_the_idle_threshold():
    # No served row: a command under OPEN_HALF_GAP reads closed, one at it reads open.
    assert hold.OPEN_HALF_GAP == pytest.approx(0.0524)
    below, at = hold.OPEN_HALF_GAP - 1e-4, hold.OPEN_HALF_GAP
    assert _subtasks([(W, below, 0.02), (W, at, 0.02)], width=0.0) == ["grasp", "reach"]
    # With a row, the threshold is its width.
    assert _subtasks([(W, W - 1e-4, 0.02), (W, W, 0.02)]) == ["grasp", "reach"]


def test_subtask_ids_resets_on_a_new_cycle_and_carries_over_the_frames_between():
    # Cycle 1 lifts, two frames between cycles keep its lift (release), cycle 2 starts over.
    rows = [
        (W, CLOSE, 0.01, W, 1),
        (W, CLOSE, 0.07, W, 1),
        (W, LANDING, 0.07, W, 0),
        (W, CLOSE, 0.07, W, 0),
        (W, CLOSE, 0.07, W, 2),
        (W, CLOSE, 0.07, W, 2),
    ]
    ids = hold.subtask_ids(*zip(*rows))
    assert [hold.SUBTASKS[i] for i in ids] == [
        "grasp",
        "carry",
        "release",
        "carry",
        "grasp",
        "grasp",
    ]
    # The same cycle value again after the 0 frames is no new cycle.
    again = [r[:4] + (1,) if r[4] == 2 else r for r in rows]
    assert [hold.SUBTASKS[i] for i in hold.subtask_ids(*zip(*again))][-2:] == ["carry", "carry"]


def test_gripping_is_the_detectors_test():
    assert hold.gripping(W, CLOSE) and not hold.gripping(CLOSE, CLOSE)
    assert not hold.gripping(0.002, 0.0)  # under MIN_GAP
    assert not hold.gripping(W, CLOSE, min_excess=0.01)
