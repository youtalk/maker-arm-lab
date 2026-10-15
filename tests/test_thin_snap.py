"""MS7e serving lever S2 (ledger R13), ``thin_snap``: ``pick_snap`` lowers the served pick point
of a flat row (half height at most ``loop.THIN_SNAP_MAX_HALF_HEIGHT``) to desk contact instead
of row z + ``SNAP_Z_MARGIN``. Off, and on any thicker row, the snap is MS7d's."""

import argparse
import json
import pathlib

import pytest
import torch

from maker_arm_lab.core import batch_loop, loop
from test_perceived_loop import _rollout, _served, _teacher_scene, _thin_snap_scene

PLATE_ROW = (0.2, 0.0, 0.0016, 1.0, 0.0, 0.016, 0.0016)  # a 3.2 mm part
BRICK_ROW = (0.2, 0.0, 0.0048, 1.0, 0.0, 0.016, 0.0048)  # a 9.6 mm part


def _lowered(row, thin: bool) -> float:
    """The served pick z once ``snap_shift`` has lowered a hover 30 mm over the row centre."""
    cmd = (row[0], row[1], row[2] + 0.030)
    shift = (0.0, 0.0, 0.0)
    for _ in range(40):
        shift = loop.snap_shift(cmd, cmd, row, shift, 0.05, thin=thin)
    return cmd[2] + shift[2]


def test_a_flat_row_snaps_to_desk_contact_and_a_brick_row_keeps_the_margin():
    assert loop.DESK_CONTACT_Z == pytest.approx(0.0021)  # level tips on the desk
    assert _lowered(PLATE_ROW, False) == pytest.approx(0.0016 + loop.SNAP_Z_MARGIN)
    assert _lowered(PLATE_ROW, True) == pytest.approx(loop.DESK_CONTACT_Z)
    assert _lowered(BRICK_ROW, True) == _lowered(BRICK_ROW, False)
    assert _lowered(BRICK_ROW, True) == pytest.approx(0.0048 + loop.SNAP_Z_MARGIN)
    # The perceived half height decides, at the bar and just over it.
    at_bar = PLATE_ROW[:6] + (loop.THIN_SNAP_MAX_HALF_HEIGHT,)
    over = PLATE_ROW[:6] + (loop.THIN_SNAP_MAX_HALF_HEIGHT + 1e-4,)
    assert loop.snap_floor(at_bar, True) == loop.DESK_CONTACT_Z
    assert loop.snap_floor(over, True) == loop.snap_floor(over) == PLATE_ROW[2] + loop.SNAP_Z_MARGIN
    # A policy command already under the desk floor is left alone.
    deep = (0.2, 0.0, 0.001)
    assert loop.snap_shift(deep, deep, PLATE_ROW, (0.0, 0.0, 0.0), 0.05, thin=True)[2] == 0.0


def test_the_jaws_stay_open_past_the_ms7d_floor_only_on_a_flat_row_under_the_switch():
    cmd = (0.2, 0.0, 0.030)
    for row in (PLATE_ROW, BRICK_ROW):
        at_ms7d_floor = (0.0, 0.0, row[2] + loop.SNAP_Z_MARGIN - cmd[2])
        still = loop.snap_hold(cmd, cmd, row, at_ms7d_floor, False, thin=True)
        assert still is (row is PLATE_ROW)
        assert loop.snap_hold(cmd, cmd, row, at_ms7d_floor, False) is False
    at_desk = (0.0, 0.0, loop.DESK_CONTACT_Z - cmd[2])
    assert loop.snap_hold(cmd, cmd, PLATE_ROW, at_desk, False, thin=True) is False


def _checked(argv):
    """``argv`` through the MS7e switches and ``batch_loop.check_ms7e``, on a parser with the
    client flags the MS7e rules read."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--desk_boxes", type=int, default=0)
    parser.add_argument("--pick_snap", action="store_true")
    parser.add_argument("--gripper", default="binary")
    batch_loop.add_ms7e_arguments(parser)
    args = parser.parse_args(argv)
    batch_loop.check_ms7e(parser, args)
    return args


def test_the_switch_needs_pick_snap_and_the_read_client_passes_it():
    with pytest.raises(SystemExit):
        _checked(["--thin_snap"])
    assert batch_loop.ms7e_switches(_checked(["--pick_snap", "--thin_snap"]))["thin_snap"] is True
    assert batch_loop.ms7e_switches(argparse.Namespace())["thin_snap"] is False
    root = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    assert (root / "mj_eval_heap.py").read_text().count("thin_snap=args.thin_snap") == 1


def _run(scene, monkeypatch, thin: bool):
    """The scene's perceived rollout; returns the payload without its wall clock, what env 0
    was served on every step, and every floor ``snap_floor`` returned under the switch."""
    env, rig, policy, budget, switches = scene(monkeypatch)
    switches["thin_snap"] = thin
    served, floors = _served(env), []
    real = loop.snap_floor

    def spy(row, thin=False):
        floor = real(row, thin)
        if thin:
            floors.append((float(row[6]), floor))
        return floor

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(loop, "snap_floor", spy)
        payload = _rollout(env, rig, policy, budget=budget, perceived=True, **switches)
    env.close()
    for key in ("wall_sec", "wall_sec_per_episode"):
        payload.pop(key)
    return json.dumps(payload, sort_keys=True), served, floors


def _same(a, b) -> bool:
    return len(a) == len(b) and all(
        all(torch.equal(torch.as_tensor(x), torch.as_tensor(y)) for x, y in zip(s, t))
        for s, t in zip(a, b)
    )


def test_with_no_flat_row_the_switch_changes_no_step_and_no_payload_byte(monkeypatch):
    off, off_served, _ = _run(_teacher_scene, monkeypatch, False)
    on, on_served, floors = _run(_teacher_scene, monkeypatch, True)
    assert floors and all(h > loop.THIN_SNAP_MAX_HALF_HEIGHT for h, _ in floors)
    assert on == off and _same(on_served, off_served)


def test_on_a_flat_row_the_switch_lowers_the_floor_to_the_desk(monkeypatch):
    _, off_served, _ = _run(_thin_snap_scene, monkeypatch, False)
    _, on_served, floors = _run(_thin_snap_scene, monkeypatch, True)
    thin = [f for h, f in floors if h <= loop.THIN_SNAP_MAX_HALF_HEIGHT]
    assert thin and set(thin) == {loop.DESK_CONTACT_Z}
    assert not _same(on_served, off_served)
