"""scripts/mj_stretch_run.py (MS7f design 10.4, ledger R13): the chain of episodes on one env,
the person's acts after an episode (rules 1 and 3), the report and the token contract."""

import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import mujoco
import pytest

from maker_arm_lab.core import desk_boxes, layout, loop, oracle, parts
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig
from test_mj_desk_boxes import FIXED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mj_stretch_run

PARTS = [parts.make_part("brick", 2, 4), parts.make_part("brick", 2, 2)] * 3


def _env(monkeypatch, **kwargs):
    """Six bricks, three in the tray at the start and three waiting (the refill is shrunk)."""
    monkeypatch.setattr(layout, "TRAY_REFILL_START", 3)
    monkeypatch.setattr(layout, "TRAY_REFILL_AT", 1)
    draw = layout.sample_run_draw(random.Random(4))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1,
        draw,
        parts=PARTS,
        cameras=False,
        tray=True,
        tray_lean_deg=30.0,
        desk_boxes=4,
        gripper_mode="continuous",
        context="camera_box_target",
        tray_refill=True,
        episode_length_s=mj_stretch_run.STRETCH_EPISODE_S,
        **kwargs,
    )
    env.reset(seed=11)
    env.place_desk_boxes(0, FIXED)
    return env


def _pos(env, b):
    return env.scene[f"brick_{b}"].data.root_pos_w[0].tolist()


def test_rule_1_the_parts_left_in_the_tray_are_removed_as_one_intervention(monkeypatch):
    env = _env(monkeypatch)
    left = env.tray_parts(0)
    assert len(left) == 3
    counts = Counter(dict.fromkeys(mj_stretch_run.INTERVENTIONS, 0))
    ep = {"done_reason": "no_candidate", "interventions": {"strays": 2, "miss_streak": 0}}
    mj_stretch_run.after_episode(env, ep, counts)
    assert counts == {"tray_left": 1, "strays": 2, "miss_streak": 0}
    assert env.tray_parts(0) == [] and all(bool(env.parked[0, b]) for b in left)
    mj_stretch_run.after_episode(env, ep, counts)  # nothing left: no second intervention
    assert counts["tray_left"] == 1
    env.close()


def test_rule_3_the_jaws_and_the_tray_top_are_cleared_back_into_the_tray(monkeypatch):
    """After a miss streak the parts over the rim go back into the tray, one intervention, and
    nothing is removed."""
    env = _env(monkeypatch)
    m, d, h = env.models[0], env.datas[0], env.handles
    b = env.tray_parts(0)[0]
    x, y = layout.TRAY_CENTER
    adr = h.brick_qpos[b]
    d.qpos[adr : adr + 7] = (x, y, 0.06, 1, 0, 0, 0)  # held up over the tray, above its rim
    d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
    mujoco.mj_forward(m, d)
    env._refresh([0], images=False)
    assert b in env.tray_top_parts(0)
    counts = Counter(dict.fromkeys(mj_stretch_run.INTERVENTIONS, 0))
    ep = {"done_reason": "miss_streak", "interventions": {"strays": 0, "miss_streak": 1}}
    mj_stretch_run.after_episode(env, ep, counts)
    assert counts == {"tray_left": 0, "strays": 0, "miss_streak": 1}
    assert _pos(env, b)[2] < layout.TRAY_WALL_HEIGHT and len(env.tray_parts(0)) == 3
    env.close()


def test_a_teacher_stretch_run_serves_the_whole_pool_and_reports(monkeypatch):
    """The teacher on six bricks, three at a time, one brick filling a box: the chain serves
    every part, refills the tray, empties the boxes, and reports every part correct."""
    env = _env(monkeypatch)
    monkeypatch.setattr(desk_boxes, "BOX_VOLUME_PER_PART_VOLUME", 1e4)
    rng = random.Random(1)
    budget = loop.cycle_budget(len(PARTS))

    def factory(i, bl, draw):
        return HeapOracle(
            bl,
            env.parts,
            env.destination_bins[i],
            draw,
            rng=rng,
            cycle_budget=budget,
            boxes=env.desk_box_poses[i],
            box_sizes=env.desk_box_sizes[i],
            tray=True,
            tray_lean_deg=30.0,
        )

    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory)
    report = mj_stretch_run.stretch(
        env,
        rig,
        None,
        run_seed=11,
        layout_seed=4,
        permutation=(0, 1, 2, 3),
        cycle_budget=budget,
        max_wall_minutes=10.0,
        box_empty=True,
    )
    env.close()
    assert report["complete"] and report["parts_served"] == report["consecutive_parts"] == 6
    assert report["correct"] == 6 and report["correct_share"] == 1.0, report
    assert report["events"]["tray_refill"] >= 1 and report["events"]["box_empty"] >= 1
    assert report["interventions_total"] == 0 and report["interventions_per_1000"] == 0.0
    steps = sum(e["steps"] for e in report["episodes"])
    assert math.isclose(report["stows_per_hour"], 3600.0 * 6 / (steps / 30.0))


def test_a_stretch_run_needs_one_env_the_refill_and_the_box_emptying(monkeypatch):
    env = _env(monkeypatch)
    with pytest.raises(loop.EvalFailure, match="tray_refill"):
        mj_stretch_run.stretch(env, None, None, run_seed=1, box_empty=False)
    env.close()


def _run(*argv: str) -> list[str]:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts/mj_stretch_run.py"), *argv],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    return [line for line in out.stdout.splitlines() if line.startswith("STRETCH_")]


def test_every_exit_path_prints_one_stretch_token():
    assert [t.split()[0] for t in _run("--no_such_flag")] == ["STRETCH_FAIL"]
    tokens = _run("--oracle_driver", "--num_envs", "1", "--heap_k", "3", "--tray")
    assert len(tokens) == 1 and tokens[0].startswith("STRETCH_FAIL") and "stretch" in tokens[0]


def test_the_stretch_pool_may_hold_1000_parts_and_no_more():
    import mj_eval_heap

    args = mj_eval_heap.parse_args(["--heap_k", "1000"], mj_stretch_run._Parser)
    assert args.heap_k == 1000
    with pytest.raises(SystemExit):
        mj_eval_heap.parse_args(["--heap_k", "1001"], mj_stretch_run._Parser)
    with pytest.raises(SystemExit):  # the heap client keeps the MS7f heap's cap
        mj_eval_heap.parse_args(["--heap_k", "151"])
