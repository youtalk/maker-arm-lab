"""scripts/pick_points.py, the FK bridge of lerobot_stack/reopen_mask.py: one token per exit."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from maker_arm_lab.core.global_ik import GlobalIK
from maker_arm_lab.core.loop import commanded_pick

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pick_points.py"


def _run(*argv):
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, argv)], capture_output=True, text=True, check=False
    )
    return out.returncode, out.stdout.strip().splitlines()


def test_the_pick_points_are_commanded_pick_row_by_row(tmp_path):
    ik = GlobalIK()
    rng = np.random.default_rng(0)
    q = rng.uniform(ik.lower.numpy(), ik.upper.numpy(), size=(5, 6))
    np.save(tmp_path / "q.npy", q.astype(np.float32))  # the corpus rows are float32
    code, lines = _run(tmp_path / "q.npy", tmp_path / "pick.npy")
    assert code == 0 and lines[-1] == "PICK_OK rows=5"
    pick = np.load(tmp_path / "pick.npy")
    for row, p in zip(q.astype(np.float32), pick):
        assert np.allclose(p, commanded_pick(ik, torch.as_tensor(row)).numpy(), atol=1e-12)


def test_no_rows_is_no_pick_points(tmp_path):
    np.save(tmp_path / "q.npy", np.zeros((0, 6)))
    code, lines = _run(tmp_path / "q.npy", tmp_path / "pick.npy")
    assert code == 0 and lines[-1] == "PICK_OK rows=0"
    assert np.load(tmp_path / "pick.npy").shape == (0, 3)


def test_a_usage_error_prints_the_fail_token():
    code, lines = _run("only_one.npy")
    assert code == 1 and lines[-1].startswith("PICK_FAIL ValueError: usage")
