"""convert_to_lerobot.py on a tiny synthetic shard, in the LeRobot venv."""

import json

import h5py
import numpy as np
import pytest

import convert_to_lerobot as c2l


def _shard(path, frames=8, context=True):
    with h5py.File(path, "w") as f:
        g = f.create_group("data/demo_0")
        g.attrs["success"] = True
        g.attrs["num_samples"] = frames
        g.create_dataset("actions", data=np.zeros((frames, 7), np.float32))
        targets = np.zeros((frames, 8), np.float32)
        targets[:, 0] = np.arange(frames)
        g.create_dataset("joint_targets", data=targets)
        state = np.zeros((frames, 8), np.float32)
        state[:, 0] = np.arange(frames) - 0.5
        state[:, 6] = c2l.GRIPPER_TRAVEL
        g.create_dataset("obs/joint_pos_abs", data=state)
        g.create_dataset("obs/images/wrist_rgb", data=np.zeros((frames, 224, 224, 3), np.uint8))
        g.create_dataset("obs/images/overhead_rgb", data=np.zeros((frames, 224, 224, 3), np.uint8))
        g.create_dataset("obs/cycle_phase", data=np.ones(frames, np.int8))
        g.create_dataset("obs/cycle_brick", data=np.zeros(frames, np.int8))
        g.create_dataset("obs/clamped", data=np.zeros(frames, bool))
        if context:
            cam = g.create_dataset(
                "obs/context_camera", data=np.full((frames, 6), 0.25, np.float32)
            )
            cam.attrs["names"] = json.dumps([f"overhead_cam_{k}" for k in "xyz"] + ["r", "p", "y"])
            priv = g.create_dataset(
                "obs/context_privileged", data=np.zeros((frames, 42), np.float32)
            )
            priv.attrs["names"] = json.dumps([f"v{i}" for i in range(42)])
        init = g.create_group("initial_state")
        robot = init.create_group("articulation/robot")
        for key, width in (
            ("joint_position", 8),
            ("joint_velocity", 8),
            ("root_pose", 7),
            ("root_velocity", 6),
        ):
            robot.create_dataset(key, data=np.zeros((1, width), np.float32))
        brick = init.create_group("rigid_object/brick_0")
        brick.create_dataset("root_pose", data=np.zeros((1, 7), np.float32))
        brick.create_dataset("root_velocity", data=np.zeros((1, 6), np.float32))
    return path


def test_camera_context_becomes_the_environment_state(tmp_path):
    shard = _shard(tmp_path / "s.hdf5")
    converted, total = c2l.convert([shard], tmp_path / "ds", "t/x", 30, context="camera")
    assert (converted, total) == (1, 1)
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    feat = info["features"]["observation.environment_state"]
    assert feat["shape"] == [6] and feat["names"][0] == "overhead_cam_x"
    assert "observation.images.wrist" in info["features"]


def test_privileged_context_without_images(tmp_path):
    shard = _shard(tmp_path / "s.hdf5")
    c2l.convert([shard], tmp_path / "ds", "t/x", 30, context="privileged", images=False)
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["features"]["observation.environment_state"]["shape"] == [42]
    assert "observation.images.wrist" not in info["features"]
    assert "observation.images.overhead" not in info["features"]


def test_a_shard_without_the_channel_is_refused(tmp_path):
    shard = _shard(tmp_path / "s.hdf5", context=False)
    with pytest.raises(RuntimeError, match="context_camera"):
        c2l.convert([shard], tmp_path / "ds", "t/x", 30, context="camera")


def test_no_images_needs_a_context():
    with pytest.raises(ValueError, match="images"):
        c2l.convert([], None, "t/x", 30, images=False)


def test_delta_prev_actions_anchor_on_the_previous_target(tmp_path):
    shard = _shard(tmp_path / "s.hdf5")
    c2l.convert([shard], tmp_path / "ds", "t/x", 30, action_space="delta_prev")
    marker = json.loads((tmp_path / "ds" / "meta" / "action_space.json").read_text())
    assert marker == {"action_space": "delta_prev"}
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("t/x", root=tmp_path / "ds")
    # Frame 0 anchors on the measured joints: target 0 minus measured -0.5.
    assert float(ds[0]["action"][0]) == pytest.approx(0.5)
    # Later frames anchor on the previous frame's target: target 3 minus target 2.
    assert float(ds[3]["action"][0]) == pytest.approx(1.0)
    assert float(ds[3]["action"][6]) == 0.0
    with pytest.raises(RuntimeError, match="action_space"):
        c2l.convert([shard], tmp_path / "ds", "t/x", 30, action_space="absolute")


def test_the_c1_delta_definition_is_refused():
    with pytest.raises(ValueError, match="action_space"):
        c2l.convert([], None, "t/x", 30, action_space="delta")


def test_absolute_is_the_default_and_leaves_a_marker(tmp_path):
    shard = _shard(tmp_path / "s.hdf5")
    c2l.convert([shard], tmp_path / "ds", "t/x", 30)
    assert json.loads((tmp_path / "ds" / "meta" / "action_space.json").read_text()) == {
        "action_space": "absolute"
    }
