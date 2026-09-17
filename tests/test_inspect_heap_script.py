"""The heap inspection tool: its token contract, its logging oracle, and its read script."""

import ast
import importlib.util
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import mujoco
import pytest
import torch

from maker_arm_lab.core import layout, oracle, parts
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mj_inspect_heap.py"
READ = ROOT / "scripts" / "heap_inspect_read.py"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_prints_one_token_and_exits_through_os_exit():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    assert '"INSPECT_OK "' in source and '"INSPECT_FAIL "' in source
    exits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_exit"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
    ]
    assert exits, "the script must leave through os._exit"
    assert "except SystemExit" in source  # an argparse error is INSPECT_FAIL, not a hang


def test_an_argparse_error_is_a_fail_token():
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--out_dir", "/tmp/x", "--heap_k", "0"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 2
    assert done.stdout.strip().splitlines()[-1].startswith("INSPECT_FAIL ")


PARTS = [
    parts.make_part("brick", 2, 4),
    parts.make_part("brick", 2, 2),
    parts.make_part("brick", 1, 4),
]


def _place(env, poses) -> None:
    m, d, h = env.models[0], env.datas[0], env.handles
    for b, (x, y, z, yaw) in enumerate(poses):
        adr = h.brick_qpos[b]
        d.qpos[adr : adr + 3] = (x, y, z)
        d.qpos[adr + 3 : adr + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=120)
    env._refresh([0], images=False)


def test_the_viewer_flag_exists_and_is_off_by_default():
    module = _load(SCRIPT)
    args = module.parse_args(["--out_dir", "/tmp/x", "--num_envs", "1", "--heap_k", "3"])
    assert args.viewer is False


def test_the_logging_oracle_writes_a_scan_per_selection_and_a_phase_per_report(tmp_path):
    module = _load(SCRIPT)
    draw = layout.sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, parts=PARTS, cameras=False)
    env.reset(seed=11)
    rng = random.Random(1)
    holder = []

    def factory(i, bl, episode_draw):
        orc = module.InspectHeapOracle(
            bl, env.parts, int(env.destination_bins[i]), episode_draw, rng=rng
        )
        holder.append(orc)
        return orc

    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory)
    module._install_lift_wrapper(rig)
    (x0, _x1), _ = layout.tray_bounds()
    # Two bricks in the tray, one on the bare desk inside reach but outside the tray.
    _place(
        env, [(0.195, -0.03, 0.006, 0.3), (0.195, 0.03, 0.006, -0.8), (x0 - 0.03, 0.0, 0.006, 0.0)]
    )
    orc = holder[0]
    steps = 0
    while steps < 6000:
        orc.step = steps
        out = rig.step()
        steps += 1
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    env.close()
    scans = [e for e in orc.events if e["kind"] == "scan"]
    phases = [e for e in orc.events if e["kind"] == "phase"]
    assert len(scans) >= 2 and orc.cycles >= 2
    assert len(scans) == orc.cycles + orc.pushes + (1 if orc.phase is oracle.Phase.DONE else 0)
    assert phases and all("outcome" in e and "next_phase" in e for e in phases)
    lifts = [e for e in phases if e["phase"] == "LIFT"]
    assert lifts and all(e["pinch_ok"] is not None for e in lifts)
    factors = {
        "stall",
        "left",
        "right",
        "asymmetry",
        "min_stall",
        "min_force",
        "rise",
        "xy_error",
        "grasp_width",
    }
    assert all(factors <= set(e) for e in lifts)
    assert all(e["stall"] >= 0.0 and e["min_force"] > 0.0 for e in lifts)
    first = scans[0]
    assert all({"inside", "best_clearance", "best_wall", "axes"} <= set(p) for p in first["parts"])
    outside = [p for p in first["parts"] if not p["inside"]]
    assert [p["id"] for p in outside] == [2] and not outside[0]["candidate"]
    assert first["outside_tray"] == 1 and first["chosen"]["part"] in (0, 1)


def test_the_read_script_sums_a_fixture_log(tmp_path):
    log = tmp_path / "ep" / "events.jsonl"
    log.parent.mkdir()
    part = {
        "id": 0,
        "family": "brick",
        "name": "brick_2x4",
        "x": 0.2,
        "y": 0.0,
        "z": 0.005,
        "inside": True,
        "reach": True,
        "cell_open": True,
        "candidate": True,
        "axes": 2,
        "best_clearance": 0.01,
        "best_wall": 0.01,
        "min_swept": 0,
    }
    wall_side = dict(part, id=1, x=0.2, y=0.09, best_wall=-0.01, best_clearance=-0.01)
    events = [
        {
            "step": 0,
            "phase": "SETTLE",
            "kind": "exit",
            "part": 2,
            "family": "tile",
            "from": None,
            "to": [0.3, 0.0, 0.002],
            "target": None,
            "in_push": False,
        },
        {
            "step": 10,
            "phase": "HOVER",
            "kind": "scan",
            "candidates": 2,
            "parts": [part, wall_side],
            "chosen": {"part": 0},
            "push": None,
            "pushes_in_a_row": 0,
            "cycles": 1,
            "budget_left": 49,
            "outside_tray": 1,
        },
        {
            "step": 40,
            "phase": "LIFT",
            "kind": "phase",
            "outcome": "converged",
            "pinch_ok": True,
            "lifted": True,
            "target": 0,
            "family": "brick",
            "retries_used": 0,
            "next_phase": "RAISE",
            "target_pose": [0.2, 0.0, 0.1],
        },
        {
            "step": 50,
            "phase": "EXTEND",
            "kind": "exit",
            "part": 0,
            "family": "brick",
            "from": [0.25, 0.0, 0.2],
            "to": [0.27, 0.0, 0.2],
            "target": 0,
            "in_push": False,
        },
        {
            "step": 90,
            "phase": "PUSH_SWEEP",
            "kind": "exit",
            "part": 1,
            "family": "brick",
            "from": [0.2, 0.098, 0.005],
            "to": [0.2, 0.108, 0.005],
            "target": None,
            "in_push": True,
        },
        {
            "step": 100,
            "phase": "DONE",
            "kind": "end",
            "reason": "done",
            "done_step": 95,
            "idle_steps": 5,
            "stowed": 1,
            "parts": 3,
            "extra_stats": {
                "cycles": 1,
                "pushes": 1,
                "forced_picks": 0,
                "cycle_parts": [0],
                "cycle_outcomes": [True],
                "cells_excluded": 0,
                "parts_outside_tray": 2,
            },
        },
    ]
    log.write_text("".join(json.dumps(e) + "\n" for e in events))
    done = subprocess.run(
        [sys.executable, str(READ), str(log)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    token = [line for line in done.stdout.splitlines() if line.startswith("INSPECT_READ ")]
    assert len(token) == 1
    summary = json.loads(token[0][len("INSPECT_READ ") :])
    assert summary["stowed"] == 1 and summary["parts"] == 3
    assert summary["stowed_fraction"] == pytest.approx(1 / 3)
    assert summary["exits"] == {"settle": 1, "carried": 1, "push": 1}
    assert summary["outside_at_start_mean"] == 0  # the settle exit is not a scan row
    assert summary["wall_blocked_at_start_mean"] == 1
    assert summary["lift_outcomes"] == {"brick_ok": 1}
    assert summary["stowed_by_family"] == {"brick": 1}
    assert torch.tensor(summary["idle_steps_median"]).item() == 5
