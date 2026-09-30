"""MS7f track S (design section 8): the teacher's pace and the serving waits. With the default
values the walk and the waits are the MS7e ones."""

import inspect
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from maker_arm_lab.core import batch_loop, layout, loop, oracle, trajectory
from maker_arm_lab.tasks.manager_based.lego_sort import oracle_runtime
from test_oracle import _snapshot
from test_perceived_loop import _rollout, _served, _thin_snap_scene
from test_snap_closed import _Blind
from test_thin_snap import _same

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mj_eval_heap

SCRIPT = Path(mj_eval_heap.__file__).read_text()


def test_the_default_pace_is_the_ms7e_walk():
    assert mj_eval_heap.parse_args([]).transport_speed == trajectory.TRANSPORT_SPEED == 4.0
    default = inspect.signature(oracle_runtime.OracleRig).parameters["transport_speed"].default
    assert default == trajectory.TRANSPORT_SPEED
    assert SCRIPT.count("transport_speed=args.transport_speed") == 1
    # The MS7e walk at its pace: (waypoints, control steps) of three legs, read at 29a36ef.
    legs = ({"distance": 0.12}, {"distance": 0.30, "tilt_change": 1.4}, {"joint_change": 1.2})
    steps = [oracle_runtime.leg_resolution(300, 1.0, speed_scale=4.0, **leg) for leg in legs]
    assert steps == [(40, 22), (100, 50), (60, 30)]


def test_a_faster_pace_shortens_the_walk_and_a_pace_under_zero_is_refused(capsys):
    assert mj_eval_heap.parse_args(["--transport_speed", "6"]).transport_speed == 6.0
    fast = oracle_runtime.leg_resolution(300, 1.0, distance=0.30, tilt_change=1.4, speed_scale=6)
    assert fast == (100, 34)
    for bad in ("0", "-1"):
        with pytest.raises(SystemExit):
            mj_eval_heap.parse_args(["--transport_speed", bad])
        assert "EVAL_FAIL" in capsys.readouterr().out


def _cycle_goals(orc, scene):
    """Every goal of one happy pick cycle, SETTLE to RETREAT, as (phase, max, hold)."""
    goals = []
    for lifted in (None, None, None, None, True, None, None, None, None):
        goal = orc.next_goal(scene)
        goals.append((goal.phase.name, goal.max_steps, goal.hold_steps))
        orc.report(oracle.PhaseResult(outcome="converged", pinch_ok=True, lifted=lifted), scene)
    return goals


def test_the_default_waits_are_the_ms7e_waits_and_a_trim_changes_only_its_own():
    def fresh():
        draw = oracle.sample_episode_draw(random.Random(0), oracle.OracleNoise())
        return oracle.EnvOracle(layout.DEFAULT_BIN_LAYOUT, num_bricks=1, draw=draw)

    today = _cycle_goals(fresh(), _snapshot())
    assert ("CLOSE", 45, 45) in today and ("RELEASE", 20, 20) in today
    assert next(g for g in today if g[0] == "LIFT")[2] == 30
    args = mj_eval_heap.parse_args([])
    assert _cycle_goals(batch_loop.teacher_waits(fresh(), args), _snapshot()) == today
    trims = {"close_steps": 12, "lift_hold_steps": 8, "release_steps": 9}
    for name, value in trims.items():
        args = mj_eval_heap.parse_args([f"--{name}", str(value)])
        goals = _cycle_goals(batch_loop.teacher_waits(fresh(), args), _snapshot())
        changed = [(a, b) for a, b in zip(today, goals) if a != b]
        phase = {"close_steps": "CLOSE", "lift_hold_steps": "LIFT", "release_steps": "RELEASE"}
        assert [a[0] for a, _ in changed] == [phase[name]]
        assert changed[0][1][2] == value


def test_the_descend_pace_scales_descend_alone_and_bad_values_are_refused(capsys):
    assert mj_eval_heap.parse_args([]).descend_speed == 1.0
    default = inspect.signature(oracle_runtime.OracleRig).parameters["descend_speed"].default
    assert default == 1.0
    rig = SimpleNamespace(transport_speed=4.0, descend_speed=2.0, oracles=[None])
    rig.oracles[0] = SimpleNamespace(draw=SimpleNamespace(gain_scale=1.0))
    for phase, pace in ((oracle.Phase.DESCEND, 2.0), (oracle.Phase.HOVER, 4.0)):
        goal = SimpleNamespace(phase=phase, max_steps=300)
        got = oracle_runtime.OracleRig._leg_resolution(rig, 0, goal, distance=0.1)
        assert got == oracle_runtime.leg_resolution(300, 1.0, distance=0.1, speed_scale=pace)
    for flag, bad in (("--descend_speed", "0"), ("--close_steps", "0"), ("--release_steps", "-2")):
        with pytest.raises(SystemExit):
            mj_eval_heap.parse_args([flag, bad])
        assert "EVAL_FAIL" in capsys.readouterr().out
    for name in ("mj_eval_heap.py", "mj_generate_batch.py"):
        text = (Path(mj_eval_heap.__file__).parent / name).read_text()
        assert text.count("teacher_waits(orc, args)") == 1
        assert text.count("descend_speed=args.descend_speed") == 1


def test_the_lift_budget_rule():
    assert not loop.lift_budget_ends(10_000, False, False, 0)  # off
    assert loop.lift_budget_ends(300, False, False, 300)
    assert not loop.lift_budget_ends(299, False, False, 300)
    assert not loop.lift_budget_ends(500, True, False, 300)  # the cycle lifted a part
    assert not loop.lift_budget_ends(500, False, True, 300)  # it holds one now


def _run(monkeypatch, **switch):
    env, rig, policy, budget, switches = _thin_snap_scene(monkeypatch)
    served = _served(env)
    payload = _rollout(env, rig, policy, budget=budget, perceived=True, **switches, **switch)
    env.close()
    for key in ("wall_sec", "wall_sec_per_episode"):
        payload.pop(key)
    return json.loads(json.dumps(payload, sort_keys=True)), served


def test_with_the_lift_budget_off_every_step_and_payload_byte_is_mains(monkeypatch):
    main, main_served = _run(monkeypatch)
    off, off_served = _run(monkeypatch, lift_budget=0)
    assert off == main and _same(off_served, main_served)
    assert "lift_budget" not in main["episodes"][0]["cycle_reasons"]


def test_a_cycle_that_lifts_nothing_ends_at_the_lift_budget(monkeypatch):
    """A detector that never reads a hold: every cycle of the flat scene lifts nothing in its
    eyes, so under the switch each one ends at the budget, and it needs perceived."""
    monkeypatch.setattr(loop, "HoldDetector", _Blind)
    on, _ = _run(monkeypatch, lift_budget=40)
    ep = on["episodes"][0]
    assert ep["cycle_reasons"] and set(ep["cycle_reasons"]) == {"lift_budget"}
    assert set(ep["cycle_steps"]) == {40}
    env, rig, policy, budget, _ = _thin_snap_scene(monkeypatch)
    with pytest.raises(loop.EvalFailure) as err:
        _rollout(env, rig, policy, budget=budget, lift_budget=40)
    env.close()
    assert err.value.reason == "lift_budget"
    with pytest.raises(SystemExit):
        mj_eval_heap.parse_args(["--lift_budget", "300"])  # needs --perceived


def _close_holds(guard: bool) -> list[int]:
    """The CLOSE holds of a pick whose first LIFT fails, at --close_steps 12."""
    draw = oracle.sample_episode_draw(random.Random(0), oracle.OracleNoise())
    orc = oracle.EnvOracle(layout.DEFAULT_BIN_LAYOUT, num_bricks=1, draw=draw)
    orc.close_steps, orc.close_guard = 12, guard
    holds, scene = [], _snapshot()
    for lifted in (None, None, None, None, False, None, None, None, None):
        goal = orc.next_goal(scene)
        if goal.phase == oracle.Phase.CLOSE:
            holds.append(goal.hold_steps)
        orc.report(oracle.PhaseResult(outcome="converged", pinch_ok=True, lifted=lifted), scene)
    return holds


def test_the_close_guard_gives_a_retry_the_full_close():
    assert _close_holds(False) == [12, 12]
    assert _close_holds(True) == [12, oracle.CLOSE_STEPS]
    assert mj_eval_heap.parse_args([]).close_guard is False
    orc = batch_loop.teacher_waits(
        oracle.EnvOracle(layout.DEFAULT_BIN_LAYOUT, 1, oracle.nominal_draw()),
        mj_eval_heap.parse_args(["--close_steps", "25", "--close_guard"]),
    )
    assert (orc.close_steps, orc.close_guard) == (25, True)


def test_the_recorder_gain_floor_moves_only_the_gain_draw():
    import argparse

    parser = argparse.ArgumentParser()
    batch_loop.add_batch_arguments(parser)
    today = batch_loop.build_run(parser.parse_args(["--out_dir", "x"]))[2]
    assert today.gain_scale == (0.7, 1.0)
    floor = batch_loop.build_run(parser.parse_args(["--out_dir", "x", "--gain_min", "0.9"]))[2]
    assert floor.gain_scale == (0.9, 1.0)
    assert {**vars(floor), "gain_scale": today.gain_scale} == vars(today)


def test_a_quiet_landing_drops_the_noise_only_after_descends_walk():
    orc = SimpleNamespace(noise_factor=lambda: 1.0, draw=SimpleNamespace(action_noise_std=0.05))

    def noise(phase, walking, quiet):
        goal = SimpleNamespace(phase=phase, target_pos=(0.2, 0.0, 0.01))
        rig = SimpleNamespace(oracles=[orc], goals=[goal], plans=[[1] if walking else []])
        rig.quiet_landing = quiet
        return oracle_runtime.OracleRig._command_noise(rig, 0)

    descend, hover = oracle.Phase.DESCEND, oracle.Phase.HOVER
    assert isinstance(noise(descend, False, True), float)  # landed: no noise, no RNG draw
    for phase, walking, quiet in (
        (descend, True, True),
        (descend, False, False),
        (hover, False, True),
    ):
        assert noise(phase, walking, quiet).shape == (6,)
    assert mj_eval_heap.parse_args([]).quiet_landing is False
    for name in ("mj_eval_heap.py", "mj_generate_batch.py"):
        text = (Path(mj_eval_heap.__file__).parent / name).read_text()
        assert text.count("quiet_landing=args.quiet_landing") == 1
