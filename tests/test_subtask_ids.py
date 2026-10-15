"""scripts/subtask_ids.py, the subtask bridge of lerobot_stack/prompt_codes.py: a hand-made
teacher trace and one token per exit."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from maker_arm_lab.core.global_ik import GlobalIK
from maker_arm_lab.core.loop import commanded_pick

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "subtask_ids.py"


def _run(*argv):
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, argv)], capture_output=True, text=True, check=False
    )
    return out.returncode, out.stdout.strip().splitlines()


def _low_and_high():
    """Two joint rows whose pick points stand at least 6 cm apart in z, low first."""
    ik = GlobalIK()
    q = np.random.default_rng(0).uniform(ik.lower.numpy(), ik.upper.numpy(), size=(50, 6))
    z = [float(commanded_pick(ik, torch.as_tensor(row))[2]) for row in q]
    low, high = q[int(np.argmin(z))], q[int(np.argmax(z))]
    assert max(z) - min(z) > 0.06
    return low, high


def test_a_teacher_cycle_reads_reach_grasp_carry_release_and_each_episode_starts_over(tmp_path):
    low, high = _low_and_high()
    open_m, shut_m = 0.1048, 0.01  # the jaw commands; the target is 2 cm wide
    # episode 0: open, close (served next frame), a grip at the bottom, the rise, the open
    # (served next frame), the release; episode 1 starts open again, high up
    q = np.stack([low, low, low, high, high, high, high]).astype(np.float32)
    opening = np.array([1.0, 1.0, 0.2, 0.2, 0.2, 1.0, 1.0])  # 0.2: a 21 mm gap on the part
    cmd = np.array([open_m, shut_m, shut_m, shut_m, open_m, open_m, open_m])
    episode = np.array([0, 0, 0, 0, 0, 0, 1])
    frames = {"q": q, "opening": opening, "cmd": cmd, "width": np.full(7, 0.02), "episode": episode}
    np.savez(tmp_path / "in.npz", **frames)
    code, lines = _run(tmp_path / "in.npz", tmp_path / "ids.npy")
    assert code == 0 and lines[-1] == "SUBTASK_OK rows=7 counts=3,1,2,1", lines
    ids = np.load(tmp_path / "ids.npy")
    assert ids.dtype == np.int8 and ids.tolist() == [0, 0, 1, 2, 2, 3, 0]


def test_a_split_episode_and_a_usage_error_print_the_fail_token(tmp_path):
    frames = {k: np.zeros(3) for k in ("opening", "cmd", "width")}
    np.savez(tmp_path / "in.npz", q=np.zeros((3, 6)), episode=np.array([0, 1, 0]), **frames)
    code, lines = _run(tmp_path / "in.npz", tmp_path / "ids.npy")
    assert (
        code == 1 and lines[-1] == "SUBTASK_FAIL ValueError: an episode's frames are not contiguous"
    )
    code, lines = _run("only_one.npz")
    assert code == 1 and lines[-1].startswith("SUBTASK_FAIL ValueError: usage")
