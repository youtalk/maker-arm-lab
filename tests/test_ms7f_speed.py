"""MS7f track S (design section 8): the teacher's pace and the serving waits. With the default
values the walk and the waits are the MS7e ones."""

import inspect
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from maker_arm_lab.core import batch_loop, layout, oracle, trajectory
from maker_arm_lab.tasks.manager_based.lego_sort import oracle_runtime
from test_oracle import _snapshot

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
