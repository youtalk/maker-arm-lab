"""MS7e flat study (2026-09-30), ``snap_closed``: ``pick_snap``'s ``snap_hold`` lets the jaws go
once the measured jaw gap is under the served row's width plus ``loop.SNAP_CLOSED_MARGIN``, so a
pinch the hold detector misses is not forced open when the policy lifts. Off, ``snap_hold`` gets
no gap and holds as before."""

import argparse
import itertools
import json
import pathlib

import pytest

from maker_arm_lab.core import batch_loop, grasp, loop
from test_perceived_loop import _rollout, _served, _snap_closed_scene, _thin_snap_scene
from test_thin_snap import BRICK_ROW, PLATE_ROW, _same

HOVER = 0.030
"""m: the policy's command over the row centre, the hover ``snap_shift`` has not lowered yet."""


def _landing(row) -> float:
    """The jaw gap ``snap_hold`` holds the jaws at on the way down (the rollout's ``landing``)."""
    return 2.0 * (grasp.landing_half_gap(row[5]) - grasp.JAW_HOLD_BAND)


def _hold(row, jaw_gap, thin=False) -> bool:
    cmd = (row[0], row[1], row[2] + HOVER)
    return loop.snap_hold(cmd, cmd, row, (0.0, 0.0, 0.0), True, thin=thin, jaw_gap=jaw_gap)


def test_jaws_closed_on_the_part_end_the_hold_and_open_landing_jaws_keep_it():
    for row, thin in itertools.product((PLATE_ROW, BRICK_ROW), (False, True)):
        width = row[5]
        assert _landing(row) > width + loop.SNAP_CLOSED_MARGIN
        assert _hold(row, None, thin) and _hold(row, _landing(row), thin)
        assert _hold(row, width + loop.SNAP_CLOSED_MARGIN, thin)  # at the margin: not closed
        assert not _hold(row, width + loop.SNAP_CLOSED_MARGIN - 1e-4, thin)
        assert not _hold(row, width - 0.002, thin)  # the tips sunk 1 mm into each side
        assert not _hold(row, 0.0, thin)


def test_with_no_gap_or_an_open_one_every_snap_hold_is_todays():
    """A grid over the cases of test_heap_loop and test_thin_snap: the lowering, at the floor,
    stopped short, the policy at grasp height, the stop flag, no row, off the centre."""
    rows = (PLATE_ROW, BRICK_ROW, PLATE_ROW[:5] + (0.0, PLATE_ROW[6]))
    xs = (0.2, 0.2 + loop.SNAP_Z_RADIUS + 0.001)
    heights = (0.0, 0.0021, 0.004, 0.0078, 0.012, 0.030, 0.060)
    shifts = (0.0, -0.004, -0.028)
    held = 0
    for row, x, cz, pz, dz, descending, stop, thin in itertools.product(
        rows, xs, heights, heights, shifts, (False, True), (False, True), (False, True)
    ):
        args = ((x, 0.0, cz), (x, 0.0, pz), row, (0.0, 0.0, dz), descending)
        today = loop.snap_hold(*args, held=stop, thin=thin)
        assert loop.snap_hold(*args, held=stop, thin=thin, jaw_gap=None) is today
        open_gap = max(_landing(row), row[5] + loop.SNAP_CLOSED_MARGIN)
        assert loop.snap_hold(*args, held=stop, thin=thin, jaw_gap=open_gap) is today
        closed = loop.snap_hold(*args, held=stop, thin=thin, jaw_gap=row[5])
        assert closed is False
        held += today
    assert held > 0  # the grid reaches the holds, not only the early returns


def _run(scene, monkeypatch, **switch):
    """The scene's perceived rollout; returns the payload without its wall clock, what env 0
    was served on every step, and the jaw gap of every ``snap_hold`` call with the rule's
    width."""
    env, rig, policy, budget, switches = scene(monkeypatch)
    switches.pop("snap_closed", None)
    served, calls = _served(env), []
    real = loop.snap_hold

    def spy(cmd, pick, row, shift, descending, jaw_gap=None, **kwargs):
        calls.append((jaw_gap, float(row[5])))
        return real(cmd, pick, row, shift, descending, jaw_gap=jaw_gap, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(loop, "snap_hold", spy)
        payload = _rollout(env, rig, policy, budget=budget, perceived=True, **switches, **switch)
    env.close()
    for key in ("wall_sec", "wall_sec_per_episode"):
        payload.pop(key)
    return json.dumps(payload, sort_keys=True), served, calls


def test_with_the_switch_off_every_step_and_payload_byte_is_mains(monkeypatch):
    main, main_served, main_calls = _run(_thin_snap_scene, monkeypatch)
    off, off_served, off_calls = _run(_thin_snap_scene, monkeypatch, snap_closed=False)
    assert main_calls and {g for g, _ in main_calls} == {g for g, _ in off_calls} == {None}
    assert off == main and _same(off_served, main_served)


class _Blind:
    """A hold detector that never reads a hold or a lift: the flat study's missed pinch."""

    def __init__(self):
        self.holding = self.lifted = self.started = self.lift_started = False
        self.ended = self.lost = self.blocked = False

    def update(self, measured_gap, commanded_gap, pick_z, entry=True):
        pass


def test_a_pinch_the_detector_misses_is_forced_open_only_with_the_switch_off(monkeypatch):
    """The flat scene with a detector that never holds: off, ``snap_hold`` reopens the jaws
    the teacher closed once its lift raises the command, and the scene stows 2 of its 3 parts
    with no cycle ending on a stow; under the switch the closed gap ends the hold, and the
    scene stows all 3 in 2 cycles."""
    monkeypatch.setattr(loop, "HoldDetector", _Blind)
    off, _, _ = _run(_thin_snap_scene, monkeypatch, snap_closed=False)
    on, _, calls = _run(_thin_snap_scene, monkeypatch, snap_closed=True)
    off, on = json.loads(off)["episodes"][0], json.loads(on)["episodes"][0]

    def held_open(ep):
        return sum(row["snap_held_steps"] for row in ep["cycle_census"])

    assert "stowed" not in off["cycle_reasons"] and "stowed" in on["cycle_reasons"], (off, on)
    assert on["bricks_stowed"] > off["bricks_stowed"] and held_open(on) < held_open(off)
    gaps = [g for g, _ in calls]
    assert gaps and all(0.0 <= g <= 2.0 * grasp.OPEN_HALF_GAP + 1e-3 for g in gaps)
    assert any(g >= w + loop.SNAP_CLOSED_MARGIN for g, w in calls)  # landing open
    assert any(g < w + loop.SNAP_CLOSED_MARGIN for g, w in calls)  # closed on the part


def _checked(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--desk_boxes", type=int, default=0)
    parser.add_argument("--pick_snap", action="store_true")
    batch_loop.add_ms7e_arguments(parser)
    args = parser.parse_args(argv)
    batch_loop.check_ms7e(parser, args)
    return args


def test_the_flag_needs_pick_snap_and_the_read_client_passes_it():
    with pytest.raises(SystemExit):
        _checked(["--snap_closed"])
    args = _checked(["--pick_snap", "--snap_closed"])
    assert batch_loop.ms7e_switches(args)["snap_closed"] is True
    assert batch_loop.ms7e_switches(argparse.Namespace())["snap_closed"] is False
    root = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    assert (root / "mj_eval_heap.py").read_text().count("snap_closed=args.snap_closed") == 1


def test_the_rollout_refuses_the_switch_without_pick_snap(monkeypatch):
    env, rig, policy, budget, switches = _snap_closed_scene(monkeypatch)
    switches["pick_snap"] = switches["thin_snap"] = False
    with pytest.raises(loop.EvalFailure) as err:
        _rollout(env, rig, policy, budget=budget, perceived=True, **switches)
    env.close()
    assert err.value.reason == "snap_closed"
