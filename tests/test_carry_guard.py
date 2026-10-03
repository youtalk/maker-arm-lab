"""MS7g ``carry_guard`` (design 5.2): the guard on a lifted part's release, its margin, the rig's
perceived carry of a still guard, the refusals and the switch-off pin. The no-truth test of the
carry entry is ``test_perceived_loop.test_no_decision_reads_the_truth`` (``_carry_scene``)."""

import json
import math
from typing import ClassVar

import pytest

from maker_arm_lab.core import desk_boxes, grasp, loop
from maker_arm_lab.core.hold import MIN_EXCESS
from test_ms7f_stack import ROOT, _gate_argv, _load, ge
from test_perceived_loop import ZERO, _carry_scene, _rollout, _served, _teacher_scene
from test_run_gate_eval import _heap_payload
from test_thin_snap import _same

BOX = (0.20, 0.21, math.radians(30.0))
SIZE = (0.09, 0.07, 0.04)


def _at(u: float, v: float) -> tuple[float, float]:
    """The world xy of the point (u, v) in BOX's own frame."""
    c, s = math.cos(BOX[2]), math.sin(BOX[2])
    return BOX[0] + c * u - s * v, BOX[1] + s * u + c * v


def test_the_guard_margin_is_read_in_the_yawed_box_frame():
    margin = loop.GUARD_BOX_MARGIN_M
    assert margin == 0.010
    for u, v, inside, deep in (
        (0.0, 0.0, True, True),
        (0.045 - 0.011, 0.0, True, True),  # 11 mm off the long wall
        (0.045 - 0.009, 0.0, True, False),  # 9 mm off it: the rim band
        (0.0, 0.035 - 0.009, True, False),  # the short wall's band
        (0.046, 0.0, False, False),  # over the wall
        (0.0, 0.040, False, False),  # past the short wall, within the long one's reach
    ):
        xy = _at(u, v)
        assert loop.in_box_interior(xy, BOX, SIZE) is inside, (u, v)
        assert loop.in_box_interior(xy, BOX, SIZE, margin) is deep, (u, v)


def test_the_guard_serves_no_command_that_would_end_a_lifted_hold_away_from_the_box():
    measured, last = 0.0158, 0.0120  # the last command looser than the squeeze
    squeeze = grasp.close_gap(measured)
    assert squeeze == pytest.approx(measured - 0.006)
    closing = measured - MIN_EXCESS  # the widest command that keeps the hold
    assert loop.guarded_gap(closing, measured, last, None) == (closing, None)
    assert loop.guarded_gap(0.001, measured, last, None) == (0.001, None)  # a tighter squeeze
    for opening in (closing + 1e-4, measured, 0.1048):
        # The run's gap is the tighter of the last command and the teacher's squeeze.
        assert loop.guarded_gap(opening, measured, last, None) == (squeeze, squeeze)
        assert loop.guarded_gap(opening, measured, 0.004, None) == (0.004, 0.004)
    assert loop.guarded_gap(0.1048, 0.004, 0.003, None) == (0.0, 0.0)  # a narrow part: shut
    # The run's gap stays fixed while the jaws close on a part that slipped out, so the measured
    # gap reaches it and the detector's hold ends; a command that keeps the hold ends the run.
    held = squeeze
    for slipping in (0.0150, 0.0120, 0.0100, held):
        assert loop.guarded_gap(0.1048, slipping, held, held) == (held, held)
    assert loop.guarded_gap(0.002, 0.0158, held, held) == (0.002, None)


def test_the_carry_release_height_clears_the_rim_from_the_holds_start():
    deep, shallow = (0.09, 0.09, 0.05), (0.09, 0.09, 0.03)
    assert desk_boxes.rim_z(deep) == pytest.approx(0.055)
    assert loop.carry_release_z(0.006, deep) == desk_boxes.DESK_BOX_RELEASE_Z == 0.10
    assert loop.carry_release_z(0.035, deep) == pytest.approx(0.10)
    assert loop.carry_release_z(0.05, deep) == pytest.approx(0.115)
    assert loop.carry_release_z(0.05, shallow) == pytest.approx(0.10)
    assert loop.carry_release_z(0.30, deep) == loop.CARRY_RELEASE_MAX_Z == 0.12


def _run(scene, monkeypatch, **switch):
    """The scene's perceived rollout: the payload without its wall clock, and what env 0 was
    served on every step."""
    env, rig, policy, budget, switches = scene(monkeypatch)
    switches |= switch
    served = _served(env)
    payload = _rollout(env, rig, policy, budget=budget, perceived=True, **switches)
    env.close()
    for key in ("wall_sec", "wall_sec_per_episode"):
        payload.pop(key)
    return json.loads(json.dumps(payload, sort_keys=True)), served


def test_with_the_switch_off_every_step_and_payload_byte_is_the_same(monkeypatch):
    main, main_served = _run(_teacher_scene, monkeypatch)
    off, off_served = _run(_teacher_scene, monkeypatch, carry_guard=False)
    assert off == main and _same(off_served, main_served)
    assert "carry_guard" not in off["inference"]["loop"]
    rows = off["episodes"][0]["cycle_census"]
    assert rows and not set(loop.CARRY_GUARD_CENSUS) & set(rows[0])


def test_a_teacher_that_releases_inside_its_box_is_never_guarded(monkeypatch):
    """At zero perception noise the served box is the teacher's own, and its RELEASE opens over
    the box centre: the switch changes no step and no payload byte but its flag and its fields."""
    off, off_served = _run(_teacher_scene, monkeypatch, noise_scales=ZERO)
    on, on_served = _run(_teacher_scene, monkeypatch, noise_scales=ZERO, carry_guard=True)
    assert on["inference"]["loop"].pop("carry_guard") is True
    rows = on["episodes"][0]["cycle_census"]
    assert rows and all(row.pop("guard_steps") == 0 for row in rows)
    assert all(row.pop("guard_rim_steps") == 0 and row.pop("handoff") is None for row in rows)
    assert all(row.pop("stall_open") is False for row in rows)
    assert on == off and _same(on_served, off_served)


def test_a_drop_away_from_the_box_is_held_and_the_rigs_carry_places_it(monkeypatch):
    """The carry scene's teacher opens over the parts it picks from. Off, every lifted cycle
    ends ``dropped``; on, the guard holds each open for GUARD_STALL_STEPS, the rig carries the
    part to the served box and the cycle ends ``placed`` with handoff "placed"."""
    off, _ = _run(_carry_scene, monkeypatch, carry_guard=False)
    on, _ = _run(_carry_scene, monkeypatch)
    off_ep, on_ep = off["episodes"][0], on["episodes"][0]
    assert "dropped" in off_ep["cycle_reasons"] and "placed" not in off_ep["cycle_reasons"]
    rows = on_ep["cycle_census"]
    handed = [row for row in rows if row["handoff"] is not None]
    assert handed and {row["handoff"] for row in handed} == {"placed"}, rows
    assert all(row["guard_steps"] >= loop.GUARD_STALL_STEPS for row in handed)
    reasons = [on_ep["cycle_reasons"][j] for j, row in enumerate(rows) if row["handoff"]]
    assert set(reasons) == {"placed"}
    assert on_ep["bricks_stowed"] > off_ep["bricks_stowed"]
    assert on["inference"]["loop"]["carry_guard"] is True


class _Stalled(loop.AirStall):
    """The census's air stall, stalled on every step (the scenes' arm is never over a tray)."""

    def update(self, pick, over_tray):
        self.stalled = True
        return True


def test_an_air_stall_with_nothing_held_opens_the_jaws_and_fails_the_cycle(monkeypatch):
    """A policy that holds still with open jaws: off, every cycle runs to its step budget; on,
    each ends ``stall_open`` once the stall has lasted AIR_STALL_STEPS steps of the cycle, the
    jaws are served fully open until the next selection, and the target is booked."""
    monkeypatch.setattr(loop, "AirStall", _Stalled)
    off, _ = _run(_still_scene, monkeypatch, carry_guard=False)
    on, _ = _run(_still_scene, monkeypatch)
    off_ep, on_ep = off["episodes"][0], on["episodes"][0]
    assert set(off_ep["cycle_reasons"]) == {"budget"}
    assert on_ep["cycle_reasons"] and set(on_ep["cycle_reasons"]) == {"stall_open"}
    assert set(on_ep["cycle_steps"]) == {loop.AIR_STALL_STEPS}
    assert all(row["stall_open"] and row["handoff"] is None for row in on_ep["cycle_census"])
    assert on_ep["cycle_classes"] and set(on_ep["cycle_classes"]) == {"no_close"}


def _still_scene(monkeypatch):
    env, rig, _, budget, switches = _teacher_scene(monkeypatch)
    from test_perceived_loop import _Still

    return env, rig, _Still(), budget, switches | {"carry_guard": True}


class _StalledWhileLifted(loop.AirStall):
    """Stalled while the loop's hold detector holds a part its hold lifted."""

    def update(self, pick, over_tray):
        d = _Detector.made[0]
        self.stalled = d.holding and d.lifted
        return self.stalled


class _Detector(loop.HoldDetector):
    made: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _Detector.made.append(self)


def test_an_air_stall_with_a_held_part_hands_it_to_the_rig(monkeypatch):
    """A stall read while the detector holds a lifted part hands the part to the rig with no
    guard step. At zero noise the first carry is the teacher's own and the rig places it; a
    later one starts beside box 2 bound for box 1 across the base, where only the leg by way of
    ``CARRY_VIA_POINT`` plans (it was declined without it), and the jaws it takes are already
    empty there, so it ends lifted_lost."""
    _Detector.made = []
    monkeypatch.setattr(loop, "HoldDetector", _Detector)
    monkeypatch.setattr(loop, "AirStall", _StalledWhileLifted)
    on, _ = _run(_teacher_scene, monkeypatch, noise_scales=ZERO, carry_guard=True)
    rows = on["episodes"][0]["cycle_census"]
    handed = [row for row in rows if row["handoff"] is not None]
    assert [row["handoff"] for row in handed] == ["placed", "lifted_lost"], rows
    assert all(row["guard_steps"] == 0 for row in handed)


def test_the_rollout_refuses_the_switch_without_perceived_or_with_the_passenger_drop(
    monkeypatch,
):
    for kwargs in ({}, {"perceived": True, "passenger_drop": True}):
        env, rig, policy, budget, _ = _teacher_scene(monkeypatch)
        with pytest.raises(loop.EvalFailure) as err:
            _rollout(env, rig, policy, budget=budget, carry_guard=True, **kwargs)
        env.close()
        assert err.value.reason == "carry_guard", kwargs


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--oracle_driver"], "--carry_guard needs --perceived"),
        (["--perceived", "--passenger_drop"], "--carry_guard refuses --passenger_drop"),
    ],
)
def test_the_client_refuses_the_switch_with_its_token(argv, message, capsys):
    module = _load(ROOT / "scripts" / "mj_eval_heap.py")
    with pytest.raises(SystemExit) as exit_info:
        module.main(argv + ["--heap_k", "3", "--desk_boxes", "4", "--carry_guard"])
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("EVAL_FAIL ") and message in out[0], out


def test_the_supervisor_forwards_the_switch_and_its_cache_key_records_it(tmp_path):
    draw = ge.make_eval_draws(1, 1)[0]
    argv = _gate_argv(tmp_path) + ["--gripper", "continuous", "--perceived"]
    plain = ge._build_cfg(ge._parse_args(argv))
    cfg = ge._build_cfg(ge._parse_args(argv + ["--carry_guard"]))
    assert ge._eval_command(cfg, draw).count("--carry_guard") == 1
    assert "--carry_guard" not in ge._eval_command(plain, draw)
    base = {name: False for name in ge.LOOP_SWITCHES} | {"perceived": True}
    base["noise_scales"] = ge.noise_scales_key("")
    assert ge.loop_switches(cfg.client_flags) == base | {"carry_guard": True, "rake_cap": 0}
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None}
    path = tmp_path / "eval_000.json"

    def hit(gate, loop_block):
        inference = {**served, "loop": loop_block}
        payload = _heap_payload(draw, desk_boxes=4, gripper="continuous", inference=inference)
        path.write_text(json.dumps(payload))
        return ge._cached_payload(gate, draw) is not None

    recorded = {k: v for k, v in base.items() if k != "carry_guard"}  # off: not recorded
    assert hit(plain, recorded) and not hit(cfg, recorded)
    assert hit(cfg, recorded | {"carry_guard": True}) and not hit(
        plain, recorded | {"carry_guard": True}
    )
    for bad, message in (
        (_gate_argv(tmp_path) + ["--carry_guard"], "--carry_guard needs --perceived"),
        (argv + ["--carry_guard", "--passenger_drop"], "--carry_guard refuses --passenger_drop"),
    ):
        with pytest.raises(ValueError, match=message):
            ge._build_cfg(ge._parse_args(bad))
