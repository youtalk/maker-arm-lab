"""prompt_codes.py: a V root's frames reach the subtask bridge by index, and the codes are
3 * subtask + POSITIVE, one int8 a frame."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import prompt_codes
from train_pi05 import CODES, prompt

STATE_NAMES = [f"j{i}" for i in range(6)] + ["gripper_open"] + [f"c{i}" for i in range(14)]
STATE_NAMES += ["target_width", "target_half_height"]


def _root(tmp_path, index=None):
    """A V root of two episodes (3 + 2 frames): state[:, 0] = the frame, the width 0.02 + frame."""
    root = tmp_path / "root"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    n = 5
    state = np.zeros((n, 23), np.float32)
    state[:, 0], state[:, 6], state[:, 21] = np.arange(n), 0.5, 0.02 + np.arange(n)
    action = np.zeros((n, 7), np.float32)
    action[:, 6] = 0.1 + np.arange(n)
    table = pa.table(
        {
            "index": np.arange(n) if index is None else np.asarray(index),
            "episode_index": [0, 0, 0, 1, 1],
            "observation.state": list(state),
            "action": list(action),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    names = {"observation.state": {"names": STATE_NAMES}}
    names["action"] = {"names": [f"j{i}" for i in range(6)] + ["gripper_gap_m"]}
    (root / "meta" / "info.json").write_text(json.dumps({"features": names}))
    return root


def test_the_codes_are_positive_subtask_codes_by_frame_index(tmp_path):
    root, seen = _root(tmp_path), []

    def fake_ids(frames):
        seen.append(frames)
        return np.array([0, 1, 2, 3, 0])

    summary = prompt_codes.write_codes(root, root / CODES, fake_ids)
    (frames,) = seen
    assert frames["q"].shape == (5, 6) and frames["q"][:, 0].tolist() == [0, 1, 2, 3, 4]
    assert frames["opening"].tolist() == [0.5] * 5
    np.testing.assert_allclose(frames["width"], 0.02 + np.arange(5), rtol=1e-6)
    np.testing.assert_allclose(frames["cmd"], 0.1 + np.arange(5), rtol=1e-6)
    assert frames["episode"].tolist() == [0, 0, 0, 1, 1]
    codes = np.load(root / CODES)
    assert codes.dtype == np.int8 and codes.tolist() == [1, 4, 7, 10, 1]
    assert prompt(codes[2]).endswith("carry it to its box. Advantage: positive")
    assert (summary["frames"], summary["episodes"]) == (5, 2)
    assert summary["shares"]["reach the marked part"] == 0.4


def test_a_root_out_of_index_order_fails_and_the_real_bridge_runs(tmp_path):
    with pytest.raises(ValueError, match="not 0 to N - 1"):
        prompt_codes.read_frames(_root(tmp_path, index=[0, 1, 3, 2, 4]))
    root = _root(tmp_path / "ok")
    out = subprocess.run(
        [sys.executable, "prompt_codes.py", "--root", str(root), "--out", str(tmp_path / "c.npy")],
        cwd=Path(prompt_codes.__file__).parent,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith("CODES_OK "), out.stdout + out.stderr
    # each episode starts at reach; a command reads against the width its frame answered, the
    # last frame's (hold.subtask_ids): 0.1048 m at most over 0.02 m reads open (reach), over the
    # fixture's 1.02 m and up closed with no lift (grasp)
    assert np.load(tmp_path / "c.npy").tolist() == [1, 1, 4, 1, 4]


def test_a_usage_error_prints_the_fail_token():
    out = subprocess.run(
        [sys.executable, "prompt_codes.py"],
        cwd=Path(prompt_codes.__file__).parent,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 2 and out.stdout.strip().startswith("CODES_FAIL usage:")
