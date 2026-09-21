"""convert_to_lerobot.py on a tiny synthetic shard, in the LeRobot venv."""

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import convert_to_lerobot as c2l

SCRIPT = Path(__file__).resolve().parent / "convert_to_lerobot.py"


def _shard(
    path,
    frames=8,
    context=True,
    bricks=None,
    actor=None,
    success=True,
    demo=0,
    box=False,
    target=False,
):
    if bricks is None:
        bricks = np.zeros(frames, np.int8)
    mode = "a" if Path(path).exists() else "w"
    with h5py.File(path, mode) as f:
        g = f.create_group(f"data/demo_{demo}")
        g.attrs["success"] = success
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
        g.create_dataset("obs/cycle_brick", data=bricks)
        g.create_dataset("obs/clamped", data=np.zeros(frames, bool))
        if actor is not None:
            g.create_dataset("obs/actor", data=np.asarray(actor, np.int16))
        if context:
            cam = g.create_dataset(
                "obs/context_camera", data=np.full((frames, 6), 0.25, np.float32)
            )
            cam.attrs["names"] = json.dumps([f"overhead_cam_{k}" for k in "xyz"] + ["r", "p", "y"])
            priv = g.create_dataset(
                "obs/context_privileged", data=np.zeros((frames, 42), np.float32)
            )
            priv.attrs["names"] = json.dumps([f"v{i}" for i in range(42)])
        if box:
            bx = g.create_dataset("obs/context_box", data=np.full((frames, 3), 0.75, np.float32))
            bx.attrs["names"] = json.dumps(["box_x", "box_y", "box_yaw"])
        if target:
            tg = g.create_dataset("obs/context_target", data=np.full((frames, 7), 0.5, np.float32))
            tg.attrs["names"] = json.dumps([f"target_{k}" for k in range(7)])
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
    converted, total, *_ = c2l.convert([shard], tmp_path / "ds", "t/x", 30, context="camera")
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
    assert marker == {"action_space": "delta_prev", "gripper": "binary"}
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


def test_an_unknown_gripper_contract_is_refused():
    # argparse's choices only guard the command line; convert() is also called directly.
    with pytest.raises(ValueError, match="gripper"):
        c2l.convert([], None, "t/x", 30, gripper="analog")


def test_delta_prev_resets_anchor_on_segment_boundary(tmp_path):
    # Regression test: each segment's first frame must anchor on measured joints,
    # not on the previous segment's last target. Create a 2-segment shard by
    # varying the brick column so cycle_segments splits on the brick change.
    bricks = np.array([0, 0, 0, 0, 1, 1, 1, 1], np.int8)
    shard = _shard(tmp_path / "s.hdf5", frames=8, bricks=bricks)
    c2l.convert(
        [shard],
        tmp_path / "ds",
        "t/x",
        30,
        split_cycles=True,
        action_space="delta_prev",
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("t/x", root=tmp_path / "ds")
    # Episode 0 (segment 0, frames 0-3): first frame anchors on measured.
    # target[0] - measured[0] = 0 - (-0.5) = 0.5
    assert float(ds[0]["action"][0]) == pytest.approx(0.5)
    # Episode 1 (segment 1, frames 4-7, starting at shard frame 4):
    # First frame of segment must also anchor on measured, not on previous
    # segment's last target. target[4] - measured[4] = 4 - 3.5 = 0.5
    # (If it incorrectly anchored on previous segment's last target[3]=3,
    # it would give 4 - 3 = 1.0, which would fail this test.)
    assert float(ds[4]["action"][0]) == pytest.approx(0.5)


def test_absolute_is_the_default_and_leaves_a_marker(tmp_path):
    shard = _shard(tmp_path / "s.hdf5")
    c2l.convert([shard], tmp_path / "ds", "t/x", 30)
    assert json.loads((tmp_path / "ds" / "meta" / "action_space.json").read_text()) == {
        "action_space": "absolute",
        "gripper": "binary",
    }


def test_cycle_segments_cut_on_actor_changes_and_drop_nonpositive_actors():
    phase = np.ones(8, np.int8)
    brick = np.zeros(8, np.int8)
    actor = np.array([1, 1, 1, 0, 0, 2, 2, 2], np.int16)
    segs = c2l.cycle_segments(phase, brick, actor=actor)
    assert [s.tolist() for s in segs] == [[0, 1, 2], [5, 6, 7]]
    actor = np.array([1, 1, -2, -2, 3, 3, 3, 3], np.int16)
    segs = c2l.cycle_segments(phase, brick, actor=actor)
    assert [s.tolist() for s in segs] == [[0, 1], [4, 5, 6, 7]]
    # Without an actor column nothing changes.
    assert [s.tolist() for s in c2l.cycle_segments(phase, brick)] == [list(range(8))]
    # The actor value returning to its earlier number across a dropped run must NOT cut: the
    # cut test is against the last KEPT frame's actor, not against whatever the dropped frames
    # in between happened to carry. (Catches a last_actor that updates on every visited frame,
    # dropped ones included, instead of only on frames that survive the actor>0 filter.)
    actor = np.array([1, 1, 0, 0, 1, 1], np.int16)
    segs = c2l.cycle_segments(np.ones(6, np.int8), np.zeros(6, np.int8), actor=actor)
    assert [s.tolist() for s in segs] == [[0, 1, 4, 5]]


def test_dagger_conversion_keeps_takeover_cycles_only(tmp_path):
    shard = _shard(tmp_path / "s.hdf5", actor=[1, 1, 1, 0, 0, -2, -2, -2])
    converted, total, *_ = c2l.convert(
        [shard], tmp_path / "ds", "t/x", 30, split_cycles=True, dagger=True, context="camera"
    )
    assert (converted, total) == (1, 1)
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["total_frames"] == 3
    with pytest.raises(ValueError, match="split_cycles"):
        c2l.convert([shard], tmp_path / "ds2", "t/x", 30, dagger=True)
    with pytest.raises(ValueError, match="lead_keep"):
        c2l.convert(
            [shard], tmp_path / "ds3", "t/x", 30, split_cycles=True, lead_keep=5, dagger=True
        )
    plain = _shard(tmp_path / "p.hdf5")
    with pytest.raises(RuntimeError, match="obs/actor"):
        c2l.convert([plain], tmp_path / "ds4", "t/x", 30, split_cycles=True, dagger=True)


def test_dagger_keeps_a_failed_episode_but_not_its_negated_cycles(tmp_path):
    # MS6: a heap episode's success attr means "every part stowed", so it is almost always
    # False. Under --dagger the positive-actor cut is the filter, and the success attr must not
    # get a veto over it -- see G1 of the pipeline verification run (8 episodes vs 15).
    shard = _shard(
        tmp_path / "s.hdf5",
        actor=[1, 1, 1, -2, -2, 3, 3, 3],
        success=False,
        bricks=np.array([0, 0, 0, 1, 1, 2, 2, 2], np.int8),
    )
    counts = c2l.convert(
        [shard],
        tmp_path / "ds",
        "t/x",
        30,
        split_cycles=True,
        dagger=True,
        images=False,
        context="camera",
    )
    # Cycles 1 and 3 survive, cycle 2 is negated and dropped; nothing is skipped up front.
    assert counts == (2, 2, 1, 0)
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["total_frames"] == 6


def test_without_dagger_a_failed_episode_is_still_skipped(tmp_path):
    shard = _shard(tmp_path / "s.hdf5", success=False)
    counts = c2l.convert([shard], tmp_path / "ds", "t/x", 30, images=False, context="camera")
    assert counts == (0, 0, 1, 1)


def test_camera_box_context_is_the_camera_pose_then_the_box_pose(tmp_path):
    shard = _shard(tmp_path / "s.hdf5", box=True)
    c2l.convert([shard], tmp_path / "ds", "t/x", 30, context="camera_box", images=False)
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    feat = info["features"]["observation.environment_state"]
    assert feat["shape"] == [9]
    assert feat["names"][0] == "overhead_cam_x" and feat["names"][6] == "box_x"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    row = LeRobotDataset("t/x", root=tmp_path / "ds")[0]["observation.environment_state"]
    # Camera first (0.25 from _shard), then the box (0.75) -- not the other way round.
    assert row.tolist() == pytest.approx([0.25] * 6 + [0.75] * 3)


def test_camera_box_target_appends_the_target_row_after_the_box(tmp_path):
    shard = _shard(tmp_path / "s.hdf5", box=True, target=True)
    c2l.convert([shard], tmp_path / "ds", "t/x", 30, context="camera_box_target", images=False)
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    feat = info["features"]["observation.environment_state"]
    assert feat["shape"] == [16]
    assert feat["names"][6] == "box_x" and feat["names"][9] == "target_0"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    row = LeRobotDataset("t/x", root=tmp_path / "ds")[0]["observation.environment_state"]
    assert row.tolist() == pytest.approx([0.25] * 6 + [0.75] * 3 + [0.5] * 7)
    with pytest.raises(RuntimeError, match="context_target"):
        c2l.convert(
            [_shard(tmp_path / "t.hdf5", box=True)],
            tmp_path / "ds2",
            "t/y",
            30,
            context="camera_box_target",
            images=False,
        )


def test_camera_box_needs_the_box_channel(tmp_path):
    shard = _shard(tmp_path / "s.hdf5")  # camera and privileged only
    with pytest.raises(RuntimeError, match="context_box"):
        c2l.convert([shard], tmp_path / "ds", "t/x", 30, context="camera_box", images=False)


def test_convert_ok_reports_what_was_read_and_skipped(tmp_path):
    # A supervisor has to be able to tell "the corpus was empty" from "the success filter ate
    # it", so read= and skipped= are part of the stdout contract, not just the return value.
    shard = tmp_path / "s.hdf5"
    _shard(shard, demo=0, success=True)
    _shard(shard, demo=1, success=False)
    argv = [
        "--shards",
        str(shard),
        "--root",
        str(tmp_path / "ds"),
        "--repo_id",
        "t/x",
        "--no_images",
        "--context",
        "camera",
    ]
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "CONVERT_OK episodes=1 total=1 read=2 skipped=1" in result.stdout


def test_the_converter_refuses_a_shard_of_the_other_gripper_contract(tmp_path):
    shard = _shard(tmp_path / "s.hdf5")
    with h5py.File(shard, "a") as f:
        f.attrs["gripper_mode"] = "continuous"
    with pytest.raises(RuntimeError, match="gripper"):
        c2l.convert([shard], tmp_path / "ds", "t/x", 30, gripper="binary")


def test_the_gripper_name_and_the_marker_follow_the_flag(tmp_path):
    shard = _shard(tmp_path / "s.hdf5")
    with h5py.File(shard, "a") as f:
        f.attrs["gripper_mode"] = "continuous"
    c2l.convert([shard], tmp_path / "ds", "t/x", 30, gripper="continuous")
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["features"]["action"]["names"][6] == "gripper_gap_m"
    marker = json.loads((tmp_path / "ds" / "meta" / "action_space.json").read_text())
    assert marker == {"action_space": "absolute", "gripper": "continuous"}


def test_a_shard_without_the_attribute_is_binary(tmp_path):
    shard = _shard(tmp_path / "s.hdf5")
    c2l.convert([shard], tmp_path / "ds", "t/x", 30)
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["features"]["action"]["names"][6] == "gripper"
