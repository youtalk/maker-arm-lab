import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from convert_to_lerobot import (  # noqa: F401 -- kept for parity
    BASE_FEATURES,
    TASK,
    convert,
    cycle_segments,
)


def _write_shard(path, demos=2, steps=5):
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["total"] = demos * steps
        for d in range(demos):
            g = data.create_group(f"demo_{d}")
            g.attrs["num_samples"] = steps
            g.attrs["success"] = True
            g.create_dataset("actions", data=np.full((steps, 7), 0.5, dtype=np.float32))
            g.create_dataset(
                "joint_targets", data=np.arange(steps * 8, dtype=np.float32).reshape(steps, 8)
            )
            state = np.ones((steps, 8), dtype=np.float32) * d
            # The jaw fractions are deliberately 0.25 / 0.75 rather than 0.5: `actions` is
            # filled with a constant 0.5, so a gripper_open of 0.5 would coincide with the
            # RAW action gripper column and a converter that skipped the divide by
            # GRIPPER_TRAVEL would still pass. The two jaws differ from each other for the
            # same reason -- reading the right jaw must not produce the left jaw's answer.
            state[:, 6] = 0.0524125 * 0.25  # left jaw (the reference) a quarter open
            state[:, 7] = 0.0524125 * 0.75  # right jaw, dropped by the converter
            g.create_dataset("obs/joint_pos_abs", data=state)
            img = np.linspace(0, 255, steps * 224 * 224 * 3).astype(np.uint8)
            img = img.reshape(steps, 224, 224, 3)
            g.create_dataset("obs/images/wrist_rgb", data=img)
            g.create_dataset("obs/images/overhead_rgb", data=img)


def _add_cycle_channels(path, phase_by_demo, brick_by_demo):
    with h5py.File(path, "a") as f:
        for d, (phase, brick) in enumerate(zip(phase_by_demo, brick_by_demo)):
            g = f["data"][f"demo_{d}"]
            g.create_dataset("obs/cycle_phase", data=np.asarray(phase, dtype=np.int8))
            g.create_dataset("obs/cycle_brick", data=np.asarray(brick, dtype=np.int8))


def test_round_trip(tmp_path):
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard)
    root = tmp_path / "dataset"
    converted, total, *_ = convert([shard], root, "youtalk/maker-arm-legosort-test", fps=30)
    assert (converted, total) == (2, 2)

    ds = LeRobotDataset("youtalk/maker-arm-legosort-test", root=root)
    assert ds.meta.fps == 30
    assert len(ds) == 10  # 2 episodes x 5 frames
    frame = ds[0]
    assert frame["action"].shape == (7,)
    # action = joint_targets[:6] + raw gripper (0.5)
    assert frame["action"][:6].tolist() == pytest.approx([0, 1, 2, 3, 4, 5], abs=1e-5)
    assert float(frame["action"][6]) == pytest.approx(0.5)
    # The action's seventh column is the RAW gripper command, and its name says so -- the
    # state's counterpart is `gripper_open`, a fraction, and the two must not be confused.
    assert ds.meta.info["features"]["action"]["names"][-1] == "gripper"
    assert frame["observation.images.wrist"].shape[-3:] != ()  # decodes to a tensor


def test_state_is_six_joints_plus_gripper_fraction(tmp_path):
    shard = tmp_path / "s.hdf5"
    _write_shard(shard, demos=2, steps=3)
    root = tmp_path / "ds"
    convert([shard], root, "youtalk/maker-arm-legosort-test", fps=30)
    ds = LeRobotDataset("youtalk/maker-arm-legosort-test", root=root)
    # Frame 3 is demo_1's first frame, whose six arm columns the fixture fills with 1.0. A
    # non-zero fill separates "the six arm angles came through" from an all-zero array, and
    # from a converter that sliced the wrong six of the eight recorded columns.
    frame = ds[3]
    assert frame["observation.state"].shape == (7,)
    assert frame["observation.state"][:6].tolist() == pytest.approx([1.0] * 6)
    assert float(frame["observation.state"][6]) == pytest.approx(0.25)
    assert ds.meta.info["features"]["observation.state"]["names"][-1] == "gripper_open"
    assert ds.meta.info["robot_type"] == "maker_arm"


def test_resume_appends(tmp_path):
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard)
    root = tmp_path / "dataset"
    convert([shard], root, "youtalk/maker-arm-legosort-test", fps=30)
    shard2 = tmp_path / "shard2.hdf5"
    _write_shard(shard2, demos=1)
    converted, total, *_ = convert([shard2], root, "youtalk/maker-arm-legosort-test", fps=30)
    assert (converted, total) == (1, 3)


def test_create_into_pre_existing_empty_dir(tmp_path):
    # Mirrors the campaign supervisor pre-creating dataset_root as an empty directory before
    # the first conversion ever runs -- must take the create path, not resume (which would fall
    # back to querying the HuggingFace Hub and 404).
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard)
    root = tmp_path / "dataset"
    root.mkdir()
    converted, total, *_ = convert([shard], root, "youtalk/maker-arm-legosort-test", fps=30)
    assert (converted, total) == (2, 2)

    ds = LeRobotDataset("youtalk/maker-arm-legosort-test", root=root)
    assert len(ds) == 10


def test_partial_root_errors(tmp_path):
    # A root that exists, is non-empty, but has no meta/info.json is neither a fresh root nor a
    # resumable one -- it's ambiguous (partial/corrupt), and must error rather than silently
    # guess create or resume.
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard)
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "junk.txt").write_text("not a lerobot dataset")
    with pytest.raises(RuntimeError, match="partial or corrupt"):
        convert([shard], root, "youtalk/maker-arm-legosort-test", fps=30)


def test_cycle_segments_cuts_on_new_brick_and_drops_settle():
    phase = np.array([0, 0, 1, 2, 3, 8, 1, 2, 3, 8, 10], dtype=np.int8)
    brick = np.array([-1, -1, 0, 0, 0, 0, 1, 1, 1, 1, -1], dtype=np.int8)
    segs = cycle_segments(phase, brick)
    assert [s.tolist() for s in segs] == [[2, 3, 4, 5], [6, 7, 8, 9]]


def test_cycle_segments_keeps_retries_and_minus_one_gaps_in_cycle():
    # Retries never change cycle_brick; a RETREAT frame with brick -1 stays
    # with the currently open cycle.
    phase = np.array([1, 2, 3, 5, 2, 3, 4, 9, 1, 2], dtype=np.int8)
    brick = np.array([0, 0, 0, 0, 0, 0, 0, -1, 1, 1], dtype=np.int8)
    segs = cycle_segments(phase, brick)
    assert [s.tolist() for s in segs] == [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9]]


def test_cycle_segments_lead_keep_attaches_the_last_settle_frames_to_the_first_cycle():
    """The recipe's 60-frame lead (openyam-lab's --lead_keep 60, MS3 recipe design 5.3): the
    first cycle keeps the LAST lead_keep SETTLE frames, later cycles are untouched, and a
    lead longer than the SETTLE run keeps the whole run."""
    phase = np.array([0, 0, 0, 1, 2, 8, 1, 2, 8], dtype=np.int8)
    brick = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    two = cycle_segments(phase, brick, lead_keep=2)
    assert [s.tolist() for s in two] == [[1, 2, 3, 4, 5], [6, 7, 8]]
    whole = cycle_segments(phase, brick, lead_keep=10)
    assert [s.tolist() for s in whole] == [[0, 1, 2, 3, 4, 5], [6, 7, 8]]
    assert [s.tolist() for s in cycle_segments(phase, brick)] == [[3, 4, 5], [6, 7, 8]]


def test_cycle_segments_with_no_brick_yields_no_segments():
    phase = np.array([0, 1], dtype=np.int8)
    brick = np.array([-1, -1], dtype=np.int8)
    assert cycle_segments(phase, brick) == []


def test_split_cycles_lead_keep_extends_the_first_cycle(tmp_path):
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard, demos=1, steps=10)
    _add_cycle_channels(shard, [[0, 0, 1, 2, 3, 8, 1, 2, 3, 8]], [[-1, -1, 0, 0, 0, 0, 1, 1, 1, 1]])
    root = tmp_path / "dataset"
    converted, total, *_ = convert(
        [shard], root, "youtalk/maker-arm-legosort-test", fps=30, split_cycles=True, lead_keep=1
    )
    assert (converted, total) == (2, 2)
    lengths = sorted(
        int(n)
        for f in root.glob("meta/episodes/**/*.parquet")
        for n in __import__("pandas").read_parquet(f)["length"]
    )
    assert lengths == [4, 5]  # the first cycle carries one SETTLE frame, the second none
    with pytest.raises(ValueError, match="split_cycles"):
        convert([shard], tmp_path / "other", "youtalk/maker-arm-legosort-test", fps=30, lead_keep=1)


def test_split_cycles_roundtrip(tmp_path):
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard, demos=1, steps=10)
    phase = [0, 0, 1, 2, 3, 8, 1, 2, 3, 8]
    brick = [-1, -1, 0, 0, 0, 0, 1, 1, 1, 1]
    _add_cycle_channels(shard, [phase], [brick])
    root = tmp_path / "dataset"
    converted, total, *_ = convert(
        [shard],
        root,
        "youtalk/maker-arm-legosort-test",
        fps=30,
        split_cycles=True,
    )
    assert (converted, total) == (2, 2)
    ds = LeRobotDataset("youtalk/maker-arm-legosort-test", root=root)
    assert len(ds) == 8  # 10 frames minus the 2 SETTLE frames


def test_split_cycles_without_channels_errors(tmp_path):
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard)
    with pytest.raises(RuntimeError, match="cycle channels"):
        convert(
            [shard], tmp_path / "ds", "youtalk/maker-arm-legosort-test", fps=30, split_cycles=True
        )


SCRIPT = Path(__file__).resolve().parent / "convert_to_lerobot.py"


def _run_script(*argv):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, check=False
    )


def test_a_conversion_failure_still_prints_a_token(tmp_path):
    # scripts/generate_dataset.py reads stdout for CONVERT_OK; a traceback on stderr with an
    # empty stdout is indistinguishable there from a wedged process.
    result = _run_script("--shards", str(tmp_path / "absent.hdf5"), "--root", str(tmp_path / "ds"))
    assert result.returncode != 0
    assert result.stdout.startswith("CONVERT_FAIL "), result.stdout


def test_a_usage_error_still_prints_a_token(tmp_path):
    # argparse's own error() exits 2 having written only to stderr.
    result = _run_script(
        "--shards",
        str(tmp_path / "s.hdf5"),
        "--root",
        str(tmp_path / "ds"),
        "--host",
        "a",
    )
    assert result.returncode != 0
    assert "CONVERT_FAIL usage: --host names the host of --own_frames rows" in result.stdout


def test_a_successful_run_prints_the_ok_token(tmp_path):
    shard = tmp_path / "s.hdf5"
    _write_shard(shard, demos=1, steps=3)
    result = _run_script("--shards", str(shard), "--root", str(tmp_path / "ds"))
    assert result.returncode == 0, result.stderr
    assert "CONVERT_OK episodes=1 total=1" in result.stdout
