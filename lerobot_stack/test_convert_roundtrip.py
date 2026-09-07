import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from convert_to_lerobot import (  # noqa: F401 -- kept for parity
    FEATURES,
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
            state[:, 6] = 0.0524125 * 0.5  # left jaw half open
            state[:, 7] = 0.0524125 * 0.5
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


def _add_initial_state(path, num_bricks=2):
    # Same layout test_extract_replay_bundle.py's fixture pins against IsaacLab's
    # InitialStateRecorder (leading record axis of length 1 on every leaf).
    with h5py.File(path, "a") as f:
        for demo in f["data"]:
            g = f["data"][demo]
            init = g.create_group("initial_state")
            robot = init.create_group("articulation").create_group("robot")
            robot.create_dataset("joint_position", data=np.zeros((1, 8), dtype=np.float32))
            robot.create_dataset("joint_velocity", data=np.zeros((1, 8), dtype=np.float32))
            robot.create_dataset("root_pose", data=np.zeros((1, 7), dtype=np.float32))
            robot.create_dataset("root_velocity", data=np.zeros((1, 6), dtype=np.float32))
            rigid = init.create_group("rigid_object")
            for b in range(num_bricks):
                brick = rigid.create_group(f"brick_{b}")
                brick.create_dataset("root_pose", data=np.full((1, 7), b, dtype=np.float32))
                brick.create_dataset("root_velocity", data=np.zeros((1, 6), dtype=np.float32))


def test_round_trip(tmp_path):
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard)
    root = tmp_path / "dataset"
    converted, total = convert([shard], root, "youtalk/maker-arm-legosort-test", fps=30)
    assert (converted, total) == (2, 2)

    ds = LeRobotDataset("youtalk/maker-arm-legosort-test", root=root)
    assert ds.meta.fps == 30
    assert len(ds) == 10  # 2 episodes x 5 frames
    frame = ds[0]
    assert frame["action"].shape == (7,)
    # action = joint_targets[:6] + raw gripper (0.5)
    assert frame["action"][:6].tolist() == pytest.approx([0, 1, 2, 3, 4, 5], abs=1e-5)
    assert float(frame["action"][6]) == pytest.approx(0.5)
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
    assert float(frame["observation.state"][6]) == pytest.approx(0.5)
    assert ds.meta.info["features"]["observation.state"]["names"][-1] == "gripper_open"
    assert ds.meta.info["robot_type"] == "maker_arm"


def test_resume_appends(tmp_path):
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard)
    root = tmp_path / "dataset"
    convert([shard], root, "youtalk/maker-arm-legosort-test", fps=30)
    shard2 = tmp_path / "shard2.hdf5"
    _write_shard(shard2, demos=1)
    converted, total = convert([shard2], root, "youtalk/maker-arm-legosort-test", fps=30)
    assert (converted, total) == (1, 3)


def test_create_into_pre_existing_empty_dir(tmp_path):
    # Mirrors the campaign supervisor pre-creating dataset_root as an empty directory before
    # the first conversion ever runs -- must take the create path, not resume (which would fall
    # back to querying the HuggingFace Hub and 404).
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard)
    root = tmp_path / "dataset"
    root.mkdir()
    converted, total = convert([shard], root, "youtalk/maker-arm-legosort-test", fps=30)
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


def test_cycle_segments_with_no_brick_yields_no_segments():
    phase = np.array([0, 1], dtype=np.int8)
    brick = np.array([-1, -1], dtype=np.int8)
    assert cycle_segments(phase, brick) == []


def test_split_cycles_roundtrip_with_sidecars(tmp_path):
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard, demos=1, steps=10)
    phase = [0, 0, 1, 2, 3, 8, 1, 2, 3, 8]
    brick = [-1, -1, 0, 0, 0, 0, 1, 1, 1, 1]
    _add_cycle_channels(shard, [phase], [brick])
    _add_initial_state(shard)
    root, sidecars = tmp_path / "dataset", tmp_path / "sidecars"
    converted, total = convert(
        [shard],
        root,
        "youtalk/maker-arm-legosort-test",
        fps=30,
        split_cycles=True,
        sidecar_dir=sidecars,
    )
    assert (converted, total) == (2, 2)
    ds = LeRobotDataset("youtalk/maker-arm-legosort-test", root=root)
    assert len(ds) == 8  # 10 frames minus the 2 SETTLE frames
    s0 = np.load(sidecars / "0.npz")
    assert s0["cycle_brick"].tolist() == [0, 0, 0, 0]
    assert s0["brick_root_pose"].shape == (2, 7)  # source episode's bricks, intact
    s1 = np.load(sidecars / "1.npz")
    assert s1["cycle_brick"].tolist() == [1, 1, 1, 1]
    assert s1["cycle_phase"].tolist() == [1, 2, 3, 8]


def test_split_cycles_roundtrip_without_sidecars(tmp_path):
    # generate_dataset._convert_command emits exactly this (split_cycles, no sidecar_dir)
    # whenever cfg.sidecar_dir is unset -- the campaign's most common mode.
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard, demos=1, steps=10)
    phase = [0, 0, 1, 2, 3, 8, 1, 2, 3, 8]
    brick = [-1, -1, 0, 0, 0, 0, 1, 1, 1, 1]
    _add_cycle_channels(shard, [phase], [brick])
    _add_initial_state(shard)
    root = tmp_path / "dataset"
    converted, total = convert(
        [shard],
        root,
        "youtalk/maker-arm-legosort-test",
        fps=30,
        split_cycles=True,
        sidecar_dir=None,
    )
    assert (converted, total) == (2, 2)
    ds = LeRobotDataset("youtalk/maker-arm-legosort-test", root=root)
    assert len(ds) == 8  # 10 frames minus the 2 SETTLE frames
    assert not (tmp_path / "sidecars").exists()


def test_sidecar_dir_non_empty_on_create_errors(tmp_path):
    # Campaign A's sidecars must not silently mix with a campaign B that starts a brand
    # new --root but reuses A's --sidecar_dir; episode indices in both start at 0.
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard, demos=1, steps=10)
    phase = [0, 0, 1, 2, 3, 8, 1, 2, 3, 8]
    brick = [-1, -1, 0, 0, 0, 0, 1, 1, 1, 1]
    _add_cycle_channels(shard, [phase], [brick])
    _add_initial_state(shard)
    sidecars = tmp_path / "sidecars"
    sidecars.mkdir()
    (sidecars / "0.npz").write_bytes(b"not a real npz, just occupying the slot")
    root = tmp_path / "dataset"  # fresh -- never created
    with pytest.raises(RuntimeError, match="being created fresh"):
        convert(
            [shard],
            root,
            "youtalk/maker-arm-legosort-test",
            fps=30,
            split_cycles=True,
            sidecar_dir=sidecars,
        )


def test_sidecar_dir_non_empty_on_resume_appends(tmp_path):
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard, demos=1, steps=10)
    phase = [0, 0, 1, 2, 3, 8, 1, 2, 3, 8]
    brick = [-1, -1, 0, 0, 0, 0, 1, 1, 1, 1]
    _add_cycle_channels(shard, [phase], [brick])
    _add_initial_state(shard)
    root, sidecars = tmp_path / "dataset", tmp_path / "sidecars"
    convert(
        [shard],
        root,
        "youtalk/maker-arm-legosort-test",
        fps=30,
        split_cycles=True,
        sidecar_dir=sidecars,
    )
    assert sorted(p.name for p in sidecars.iterdir()) == ["0.npz", "1.npz"]

    shard2 = tmp_path / "shard2.hdf5"
    _write_shard(shard2, demos=1, steps=10)
    _add_cycle_channels(shard2, [phase], [brick])
    _add_initial_state(shard2)
    converted, total = convert(
        [shard2],
        root,
        "youtalk/maker-arm-legosort-test",
        fps=30,
        split_cycles=True,
        sidecar_dir=sidecars,
    )
    assert (converted, total) == (2, 4)
    assert sorted(p.name for p in sidecars.iterdir()) == ["0.npz", "1.npz", "2.npz", "3.npz"]


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
        "--sidecar_dir",
        str(tmp_path / "sidecars"),
    )
    assert result.returncode != 0
    assert "CONVERT_FAIL usage: --sidecar_dir requires --split_cycles" in result.stdout


def test_a_successful_run_prints_the_ok_token(tmp_path):
    shard = tmp_path / "s.hdf5"
    _write_shard(shard, demos=1, steps=3)
    result = _run_script("--shards", str(shard), "--root", str(tmp_path / "ds"))
    assert result.returncode == 0, result.stderr
    assert "CONVERT_OK episodes=1 total=1" in result.stdout
