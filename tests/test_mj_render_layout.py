"""``MjLegoSortEnv.set_part_poses`` and ``scripts/mj_render_layout.py`` (MS5 sensing)."""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from maker_arm_lab.core import parts
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj.env import MjLegoSortEnv

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mj_render_layout.py"
NAMES = ["brick_2x4", "plate_1x2"]


def small_heap_env(cameras: bool):
    names = [p.name for p in parts.by_name(NAMES)]
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1,
        draw,
        num_bricks=len(names),
        parts=parts.by_name(NAMES),
        cameras=cameras,
        tray=True,
        tray_lean_deg=30.0,
        desk_boxes=4,
        class_names=parts.PART_CLASSES,
    )
    env.reset(seed=1)
    return env


def test_set_part_poses_puts_each_part_where_the_truth_says():
    env = small_heap_env(cameras=False)
    env.set_part_poses(0, [(0.19, 0.03, 0.0098, 0.4), (0.20, -0.04, 0.0098, 1.2)])
    for b, (x, y, yaw) in enumerate([(0.19, 0.03, 0.4), (0.20, -0.04, 1.2)]):
        pos = env.scene[f"brick_{b}"].data.root_pos_w[0]
        assert pos.tolist() == pytest.approx([x, y, 0.0098], abs=1e-6)
        q = env.scene[f"brick_{b}"].data.root_quat_w[0].tolist()
        assert q == pytest.approx([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], abs=1e-6)
        assert not env.parked[0, b]


def _run(*args):
    env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "disable")}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env, check=False
    )


def test_the_render_script_usage_error_prints_one_fail_line():
    r = _run("--layout")
    tokens = [ln for ln in r.stdout.splitlines() if ln.startswith(("RENDER_OK", "RENDER_FAIL"))]
    assert r.returncode != 0 and len(tokens) == 1 and tokens[0].startswith("RENDER_FAIL")


def test_the_render_script_fails_with_a_token_on_a_missing_layout(tmp_path):
    r = _run("--layout", str(tmp_path / "x.json"), "--rig", "r", "--real", "v", "--out", "o")
    tokens = [ln for ln in r.stdout.splitlines() if ln.startswith(("RENDER_OK", "RENDER_FAIL"))]
    assert r.returncode != 0 and len(tokens) == 1 and tokens[0].startswith("RENDER_FAIL")
    json.loads(tokens[0].split(" ", 1)[1])


def test_ctrl_c_ends_the_render_script_with_its_fail_token(tmp_path, monkeypatch, capsys):
    from helpers import load_script
    from maker_arm_lab.real.rig import RigConfig

    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(RigConfig, "from_json", interrupt)
    code = load_script(SCRIPT).main(
        ["--layout", "l", "--rig", "r", "--real", "v", "--out", str(tmp_path / "o.png")]
    )
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("RENDER_")]
    assert code == 1 and lines == ["RENDER_FAIL " + json.dumps({"reason": "interrupted"})]
