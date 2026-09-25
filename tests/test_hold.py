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
