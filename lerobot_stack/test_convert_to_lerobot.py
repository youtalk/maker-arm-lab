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


def test_cycle_segments_drop_the_tray_rake_between_two_picks():
    """The shape of a recorded MS7a rake (K = 20 tray heap, 2026-09-23): the retreating cycle
    of part 9 under actor 17, the rake (PUSH_HOVER, PUSH_DESCEND, PUSH_SWEEP: 12, 13, 14)
    under the NEXT pick's actor 18 with no target (brick -1), then that pick's HOVER on part
    17. The rake used to become a training episode of its own; it is dropped."""
    phase = np.array([9, 9, 12, 12, 13, 14, 14, 1, 2, 3], np.int8)
    brick = np.array([9, 9, -1, -1, -1, -1, -1, 17, 17, 17], np.int8)
    actor = np.array([17, 17, 18, 18, 18, 18, 18, 18, 18, 18], np.int16)
    segs = c2l.cycle_segments(phase, brick, actor=actor)
    assert [s.tolist() for s in segs] == [[0, 1], [7, 8, 9]]
    # The tip-up's phases (15, 16, 17) are correctives too.
    tip = np.array([1, 15, 16, 17, 2], np.int8)
    assert [s.tolist() for s in c2l.cycle_segments(tip, np.zeros(5, np.int8))] == [[0, 4]]


def test_cycle_segments_drop_the_reopen_left_positive_by_a_failed_teacher_cycle():
    """The shape of a failed tray pick in MS7a's rung 1 corpus (proc_0, 2026-09-23): cycle 1
    on part 49 gives up at LIFT, the batch loop negates its frames there, and the give-up
    REOPEN (phase 5) is recorded after the negation under the positive ordinal 1. Then cycle 2
    picks part 44. The REOPEN used to become a 20-frame training episode of its own."""
    phase = np.array([1, 2, 3, 4, 5, 5, 5, 1, 2, 3], np.int8)
    brick = np.array([49, 49, 49, 49, 49, 49, 49, 44, 44, 44], np.int8)
    actor = np.array([-1, -1, -1, -1, 1, 1, 1, 2, 2, 2], np.int16)
    segs = c2l.cycle_segments(phase, brick, actor=actor)
    assert [s.tolist() for s in segs] == [[7, 8, 9]]
    # A stowed cycle with the same shape keeps every frame.
    segs = c2l.cycle_segments(phase, brick, actor=np.abs(actor))
    assert [s.tolist() for s in segs] == [list(range(7)), [7, 8, 9]]


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


def test_reopen_lead_is_the_teacher_opening_jaws_the_policy_left_closed():
    cmd = np.full(6, 0.016)
    opening = np.array([0.0, 0.004, 0.010, 0.0145, 0.016, 0.016])
    assert c2l.reopen_lead(opening, cmd, 0.002) == 3  # within 2 mm of the command at frame 3
    assert c2l.reopen_lead(np.full(6, 0.016), cmd, 0.002) == 0  # starts open
    assert c2l.reopen_lead(np.full(6, 0.006), np.full(6, 0.0036), 0.002) == 0  # holds a part
    blocked = np.array([0.016, 0.016, 0.0, 0.0])  # the jaws never open: the first close ends it
    assert c2l.reopen_lead(np.full(4, 0.002), blocked, 0.002) == 2
    assert c2l.reopen_lead(np.zeros(3), cmd[:3], 0.002) == 3


def _continuous(shard, per_jaw, command):
    """Give a _shard both jaws at ``per_jaw`` and a continuous gap command."""
    with h5py.File(shard, "a") as f:
        f.attrs["gripper_mode"] = "continuous"
        g = f["data/demo_0"]
        state = g["obs/joint_pos_abs"][:]
        state[:, 6] = state[:, 7] = per_jaw
        g["obs/joint_pos_abs"][...] = state
        actions = g["actions"][:]
        actions[:, 6] = command
        g["actions"][...] = actions
    return shard


def test_drop_reopen_lead_cuts_each_takeover_and_counts_it(tmp_path):
    # takeover 1 opens closed jaws over its first 3 frames, takeover 2 starts open
    shard = _shard(tmp_path / "s.hdf5", actor=[1] * 5 + [2] * 3, bricks=[0] * 5 + [1] * 3)
    per_jaw = np.array([0.0, 0.002, 0.005, 0.00725, 0.008, 0.008, 0.008, 0.008])
    _continuous(shard, per_jaw, 0.016)
    kw = {"split_cycles": True, "dagger": True, "context": "camera", "gripper": "continuous"}
    reopen = c2l.ReopenLead(0.002)
    assert c2l.convert([shard], tmp_path / "ds", "t/x", 30, reopen=reopen, **kw) == (2, 2, 1, 0)
    assert reopen == c2l.ReopenLead(0.002, frames=3, takeovers=1, emptied=0)
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["total_frames"] == 5
    assert c2l.convert([shard], tmp_path / "off", "t/x", 30, **kw) == (2, 2, 1, 0)
    with pytest.raises(ValueError, match="dagger and gripper continuous"):
        c2l.convert([shard], tmp_path / "b", "t/x", 30, split_cycles=True, reopen=reopen)


def test_a_takeover_that_never_opens_is_not_written(tmp_path):
    shard = _continuous(_shard(tmp_path / "s.hdf5", actor=[1] * 8), 0.0, 0.016)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--shards", str(shard), "--root", str(tmp_path / "ds")]
        + ["--split_cycles", "--dagger", "--gripper", "continuous", "--drop_reopen_lead", "0.002"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.splitlines()[-1] == (
        "CONVERT_OK episodes=0 total=0 read=1 skipped=0 reopen_tol=0.002 reopen_frames=8"
        " reopen_takeovers=1 reopen_emptied=1"
    ), result.stdout + result.stderr


# -- MS7i --own_frames -----------------------------------------------------------------------

NEG, POS, UNT = c2l.TAG_NEGATIVE, c2l.TAG_POSITIVE, c2l.TAG_UNTAGGED
REACH, GRASP, CARRY, RELEASE = range(4)


def _row(frames, **fields):
    """A census row of ``frames`` steps: nothing held, nothing lifted, no event."""
    row = {
        "open_steps": frames,
        "closed_steps": 0,
        "first_close_at": None,
        "off_x": None,
        "off_y": None,
        "off_z": None,
        "max_rise": 0.0,
        "held_steps": 0,
        "first_held": -1,
        "last_held": -1,
        "last_held_at": None,
        "first_held_at": None,
        "release_place": None,
        "release_at": None,
        "box": -1,
    }
    return row | fields


def _episode(n=60, s=5, last=54):
    """One cycle on frames s..last of an n-frame episode, policy frames everywhere: the
    per-frame cycle, actor, phase and subtask arrays."""
    cycle = np.zeros(n, np.int16)
    cycle[s : last + 1] = 1
    return cycle, np.zeros(n, np.int16), np.full(n, -1, np.int8), np.zeros(n, np.int8)


def _labels(row, cycle, actor, phase, subtask, cls="close_miss", reason="budget", final=(-1,) * 4):
    return c2l.own_labels([row], [cls], [reason], list(final), cycle, actor, phase, subtask)


def _window(label, lo, hi):
    """The frames lo..hi are negative and every other frame untagged."""
    want = np.full(len(label), UNT, np.int8)
    want[lo : hi + 1] = NEG
    return np.array_equal(label, want)


def test_an_off_target_first_close_ends_a_30_frame_window_clipped_to_the_cycle_start():
    cycle, actor, phase, subtask = _episode()
    row = _row(50, first_close_at=40, off_x=0.0, off_y=0.0, off_z=0.03, max_rise=0.06)
    label, (info,) = _labels(row, cycle, actor, phase, subtask, cls="lifted_lost")
    assert info["events"]["off_target"] == [45] and _window(label, 16, 45)
    assert (info["first"], info["last"], info["carry_start"]) == (5, 54, None)
    row |= {"first_close_at": 10, "off_z": 0.0, "off_x": 0.015, "off_y": 0.015}  # 21 mm in xy
    label, _ = _labels(row, cycle, actor, phase, subtask, cls="lifted_lost")
    assert _window(label, 5, 15)
    row |= {"off_x": 0.01, "off_y": 0.01}  # 14 mm: on the target
    assert not _labels(row, cycle, actor, phase, subtask, cls="lifted_lost")[0].any()
    # The window covers policy frames only.
    phase[20:25] = 3
    row |= {"first_close_at": 40, "off_z": 0.03}
    label, _ = _labels(row, cycle, actor, phase, subtask, cls="lifted_lost")
    assert (
        (label[20:25] == UNT).all() and (label[16:20] == NEG).all() and (label[25:46] == NEG).all()
    )


def test_every_quick_reopen_and_the_last_close_of_a_cycle_with_no_lift_are_events():
    cycle, actor, phase, subtask = _episode()
    subtask[20:25] = GRASP  # closed 20..24, open at 25: a re-open 5 steps after its close
    subtask[26:40] = GRASP  # closed 26..39, open at 40: 14 steps, no event
    subtask[45:55] = GRASP  # the last close, held to the cycle end
    label, (info,) = _labels(_row(50, held_steps=3), cycle, actor, phase, subtask)
    assert info["events"]["reopen"] == [25] and info["events"]["no_lift"] == [45]
    assert _window(label, 5, 45)  # 5..25 and 16..45
    # A stowed cycle has no no-lift event, and neither has one that never closed.
    _, (info,) = _labels(_row(50), cycle, actor, phase, subtask, cls="stowed")
    assert info["events"]["no_lift"] == []
    _, (info,) = _labels(_row(50), *_episode(), cls="no_close")
    assert info["events"]["no_lift"] == [] and info["events"]["reopen"] == []
    # A close of fewer than REOPEN_MIN_CLOSED steps is a command flicker: no event.
    cycle, actor, phase, subtask = _episode()
    subtask[20:22] = GRASP  # closed 20..21, open at 22
    _, (info,) = _labels(_row(50, held_steps=3), cycle, actor, phase, subtask, cls="stowed")
    assert info["events"]["reopen"] == []


def test_a_takeover_ends_the_window_on_the_last_policy_frame_and_a_decline_on_the_cycle_end():
    cycle, actor, phase, subtask = _episode()
    actor[30:55], phase[30:55] = 1, 2
    label, (info,) = _labels(_row(50), cycle, actor, phase, subtask, cls="stowed")
    assert info["events"]["takeover"] == [29] and _window(label, 5, 29)
    cycle, actor, phase, subtask = _episode()
    label, (info,) = _labels(_row(50), cycle, actor, phase, subtask, reason="takeover_declined")
    assert info["events"]["takeover"] == [54] and _window(label, 25, 54)


def test_a_slip_of_a_lifted_part_and_a_release_off_the_boxes_are_events():
    cycle, actor, phase, subtask = _episode()
    subtask[20:41] = CARRY  # the frame after the last held step (s + 30 + 1 = 36) reads closed
    row = _row(50, max_rise=0.06, held_steps=5, last_held=2, last_held_at=30)  # no positive
    label, (info,) = _labels(row, cycle, actor, phase, subtask, cls="lifted_lost")
    assert info["events"]["lost"] == [36] and _window(label, 7, 36)
    # A part that the opening jaws let go of over a box is no slip.
    boxed = row | {"release_place": "right_box", "release_at": 40}
    _, (info,) = _labels(boxed, cycle, actor, phase, subtask, cls="stowed")
    assert info["events"]["lost"] == []
    subtask[36:41] = RELEASE  # an open there is a release, not a slip
    _, (info,) = _labels(row, cycle, actor, phase, subtask, cls="lifted_lost")
    assert info["events"]["lost"] == []
    for place, hit in (("desk", [45]), ("rim", [45]), ("tray", [45]), ("wrong_box", [])):
        row |= {"release_place": place, "release_at": 40}
        _, (info,) = _labels(row, cycle, actor, phase, subtask, cls="lifted_lost")
        assert info["events"]["release"] == hit, place


def _clean(cycle, actor, phase, subtask):
    """A clean pick of part 2: reach 5..14, grasp 15..24, carry 25..39, release 40..54, its
    release over box 1 at cycle step 36."""
    subtask[15:25], subtask[25:40], subtask[40:55] = GRASP, CARRY, RELEASE
    row = _row(50, max_rise=0.08, held_steps=20, first_held=2, last_held=2, last_held_at=34)
    return row | {"first_held_at": 12, "release_place": "right_box", "release_at": 36, "box": 1}


def test_a_clean_grasp_and_a_carry_the_policy_ends_in_its_served_box_are_positive():
    cycle, actor, phase, subtask = _episode()
    row = _clean(cycle, actor, phase, subtask)
    label, (info,) = _labels(row, cycle, actor, phase, subtask, "stowed", "stowed", (-1, -1, 1, -1))
    assert info["carry_start"] == 25 and (label[5:55] == POS).all()
    assert (label[:5] == UNT).all() and (label[55:] == UNT).all()
    # A classifier's wrong box: the part rests in the served box, so the carry stays positive.
    row_w = row | {"release_place": "wrong_box", "box": 3}
    label, _ = _labels(row_w, cycle, actor, phase, subtask, final=(-1, -1, 3, -1))
    assert (label[25:55] == POS).all()
    # The policy's wrong box: the box served was right, the part rests in another.
    label, _ = _labels(row, cycle, actor, phase, subtask, final=(-1, -1, 3, -1))
    assert (label[5:25] == POS).all() and (label[25:55] == UNT).all()
    # A handoff finished the carry: untagged.
    label, _ = _labels(
        row | {"handoff": "placed"}, cycle, actor, phase, subtask, final=(-1, -1, 1, -1)
    )
    assert (label[5:25] == POS).all() and (label[25:55] == UNT).all()
    # A re-open before the carry spoils the grasp segment; the window wins where it lands.
    sub = subtask.copy()
    sub[18] = REACH
    label, (info,) = _labels(row, cycle, actor, phase, sub, final=(-1, -1, 1, -1))
    assert info["events"]["reopen"] == [18] and (label[5:19] == NEG).all()
    assert (label[19:25] == UNT).all() and (label[25:55] == POS).all()
    # An open mid-carry, more than RELEASE_LEAD steps before the release: no positive carry.
    sub = subtask.copy()
    sub[28] = RELEASE
    label, (info,) = _labels(row, cycle, actor, phase, sub, final=(-1, -1, 1, -1))
    assert info["events"]["reopen"] == [] and (label[5:25] == POS).all()
    assert (label[25:55] == UNT).all()
    sub[28], sub[31:40] = CARRY, RELEASE  # opens RELEASE_LEAD steps before the release (41)
    label, _ = _labels(row, cycle, actor, phase, sub, final=(-1, -1, 1, -1))
    assert (label[5:55] == POS).all()
    # A census hold that begins after the carry start (frame 25): the carry read is false.
    label, _ = _labels(
        row | {"first_held_at": 21}, cycle, actor, phase, subtask, final=(-1, -1, 1, -1)
    )
    assert not label.any()
    # A close and a quick open after the release is no failing re-open.
    sub = subtask.copy()
    sub[44:48] = CARRY
    label, (info,) = _labels(row, cycle, actor, phase, sub, final=(-1, -1, 1, -1))
    assert info["events"]["reopen"] == [] and (label[5:55] == POS).all()


def test_a_takeover_carry_stays_untagged_and_a_part_picked_again_is_no_positive_carry():
    cycle, actor, phase, subtask = _episode()
    row = _clean(cycle, actor, phase, subtask)
    actor[45:55], phase[45:55] = 1, 7
    label, (info,) = _labels(row, cycle, actor, phase, subtask, final=(-1, -1, 1, -1))
    assert info["events"]["takeover"] == [44]
    assert (label[15:45] == NEG).all() and (label[5:15] == POS).all() and (label[45:] == UNT).all()
    cycle, actor, phase, subtask = _episode(n=80)
    row = _clean(cycle, actor, phase, subtask)
    cycle[60:70] = 2
    again = _row(10, first_held=2, last_held=2, held_steps=10)
    label, _ = c2l.own_labels(
        [row, again],
        ["stowed", "close_miss"],
        ["stowed", "budget"],
        [-1, -1, 1, -1],
        cycle,
        actor,
        phase,
        subtask,
    )
    assert (label[5:25] == POS).all() and (label[25:55] == UNT).all()


def test_the_join_raises_on_any_mismatch():
    cycle, actor, phase, subtask = _episode()
    with pytest.raises(RuntimeError, match="census steps"):
        _labels(_row(49), cycle, actor, phase, subtask)
    split = cycle.copy()
    split[30] = 0
    split[55] = 1
    with pytest.raises(RuntimeError, match="census steps"):
        _labels(_row(50), split, actor, phase, subtask)
    extra = cycle.copy()
    extra[56:58] = 2
    with pytest.raises(RuntimeError, match="reaches 2"):
        _labels(_row(50), extra, actor, phase, subtask)
    # A row whose cycle has no frame (it opened and closed with no census step) joins empty.
    label, infos = c2l.own_labels(
        [_row(50), _row(0)],
        ["close_miss"] * 2,
        ["budget"] * 2,
        [-1] * 4,
        cycle,
        actor,
        phase,
        subtask,
    )
    assert infos[1]["first"] is None and not label.any()


def _own_shard(path, cycle, actor, phase, subtask, rows, final_box, classes=None, reasons=None):
    """A record_all shard of one demo and its payload ``<name>.meta.json``."""
    n = len(cycle)
    _shard(path, frames=n, actor=actor, box=True, target=True, success=False)
    with h5py.File(path, "a") as f:
        f.attrs["gripper_mode"] = "continuous"
        g = f["data/demo_0"]
        g.attrs["env_index"] = 0
        g["obs/cycle_phase"][...] = phase
        g.create_dataset("obs/loop_cycle", data=cycle)
        g.create_dataset("obs/subtask", data=subtask)
        g.create_dataset("obs/policy_action", data=np.zeros((n, 7), np.float32))
    payload = {
        "shard": f"/elsewhere/{path.name}",
        "record_all": True,
        "episodes": [
            {
                "env_index": 0,
                "steps": n,
                "cycle_census": rows,
                "cycle_classes": classes or ["stowed"] * len(rows),
                "cycle_reasons": reasons or ["stowed"] * len(rows),
                "final_box": final_box,
            }
        ],
    }
    path.with_name(path.stem + ".meta.json").write_text(json.dumps(payload))
    return path


def _two_cycles(tmp_path, name="proc_0.hdf5"):
    """Cycle 1 clean (positive), a handoff's rig frames 55..57, frames 58..59 between cycles,
    cycle 2 a close miss whose last close is an event (frames 60..79)."""
    cycle, actor, phase, subtask = _episode(n=80)
    row = _clean(cycle, actor, phase, subtask)
    phase[55:58] = 4
    cycle[60:80] = 2
    subtask[70:80] = GRASP
    miss = _row(20, held_steps=2)
    path = tmp_path / "host_a" / name
    path.parent.mkdir(exist_ok=True)
    return _own_shard(
        path,
        cycle,
        actor,
        phase,
        subtask,
        [row, miss],
        [-1, -1, 1, -1],
        ["stowed", "close_miss"],
        ["stowed", "budget"],
    )


def test_labeled_writes_every_policy_frame_with_the_label_and_subtask_columns(tmp_path):
    shard = _two_cycles(tmp_path)
    own = c2l.OwnFrames("labeled")
    kw = {"context": "camera_box_target", "gripper": "continuous", "images": False}
    counts = c2l.convert([shard], tmp_path / "ds", "t/x", 30, own=own, **kw)
    # Runs: 0..4 (no cycle), 5..54 (cycle 1), 58..59 (no cycle), 60..79 (cycle 2).
    assert counts == (4, 4, 1, 0)
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["total_frames"] == 77
    assert info["features"]["label"]["dtype"] == "int8"
    assert info["features"]["subtask"]["shape"] == [1]
    assert own.counts["own_pos"] == 50 and own.counts["own_neg"] == 11
    assert own.counts["own_untagged"] == 16 and own.counts["rig_frames"] == 3
    assert own.counts["ev_no_lift"] == 1 and own.counts["joined"] == own.counts["cycles"] == 2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("t/x", root=tmp_path / "ds")
    labels = [int(ds[i]["label"]) for i in range(len(ds))]
    assert labels[5:55] == [POS] * 50 and labels[-20:] == [NEG] * 11 + [UNT] * 9
    assert int(ds[30]["subtask"]) == CARRY
    rows = json.loads((tmp_path / "ds" / "meta" / "own_frames.json").read_text())["cycles"]
    assert [r["ordinal"] for r in rows] == [1, 2] and rows[0]["host"] == "host_a"
    assert rows[0]["written"] == [[1, 5, 50]] and rows[1]["events"]["no_lift"] == [70]
    assert rows[0]["labels"] == {"positive": 50, "negative": 0, "untagged": 0}
    assert rows[0]["carry_start"] == 25 and rows[0]["demo"] == "demo_0"


def test_positive_writes_only_positive_frames_in_the_stock_schema(tmp_path):
    shard = _two_cycles(tmp_path)
    kw = {"context": "camera_box_target", "gripper": "continuous", "images": False}
    own = c2l.OwnFrames("positive", host="c")
    assert c2l.convert([shard], tmp_path / "ds", "t/x", 30, own=own, **kw)[:2] == (1, 1)
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["total_frames"] == 50 and "label" not in info["features"]
    stock = _shard(tmp_path / "stock.hdf5", box=True, target=True)
    with h5py.File(stock, "a") as f:
        f.attrs["gripper_mode"] = "continuous"
    c2l.convert([stock], tmp_path / "corpus", "t/y", 30, **kw)
    corpus = json.loads((tmp_path / "corpus" / "meta" / "info.json").read_text())
    assert info["features"] == corpus["features"]
    rows = json.loads((tmp_path / "ds" / "meta" / "own_frames.json").read_text())["cycles"]
    assert {r["host"] for r in rows} == {"c"}


def test_own_frames_refuses_a_masked_shard_a_foreign_payload_and_the_stock_switches(tmp_path):
    kw = {"context": "camera_box_target", "gripper": "continuous", "images": False}
    shard = _two_cycles(tmp_path)
    with h5py.File(shard, "a") as f:
        f.create_dataset("data/demo_0/obs/actor_raw", data=np.zeros(80, np.int16))
    with pytest.raises(RuntimeError, match="actor_raw"):
        c2l.convert([shard], tmp_path / "a", "t/x", 30, own=c2l.OwnFrames("labeled"), **kw)
    other = _two_cycles(tmp_path, "proc_1.hdf5")
    meta = other.with_name("proc_1.meta.json")
    meta.write_text(meta.read_text().replace("proc_1.hdf5", "proc_9.hdf5"))
    with pytest.raises(RuntimeError, match="record_all payload"):
        c2l.convert([other], tmp_path / "b", "t/x", 30, own=c2l.OwnFrames("labeled"), **kw)
    for bad in ({"dagger": True, "split_cycles": True}, {"split_cycles": True}):
        with pytest.raises(ValueError, match="own_frames refuses"):
            c2l.convert(
                [shard], tmp_path / "c", "t/x", 30, own=c2l.OwnFrames("labeled"), **kw, **bad
            )
    with pytest.raises(ValueError, match="camera_box_target"):
        c2l.convert(
            [shard],
            tmp_path / "d",
            "t/x",
            30,
            own=c2l.OwnFrames("labeled"),
            gripper="continuous",
            context="camera_box",
            images=False,
        )


def test_convert_ok_reports_the_label_and_event_counts(tmp_path):
    shard = _two_cycles(tmp_path)
    argv = ["--shards", str(shard), "--root", str(tmp_path / "ds"), "--no_images"]
    argv += ["--context", "camera_box_target", "--gripper", "continuous", "--own_frames", "labeled"]
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, check=False
    )
    line = result.stdout.splitlines()[-1]
    assert line.startswith("CONVERT_OK episodes=4 total=4 read=1 skipped=0 own_frames=labeled")
    for part in ("own_pos=50", "own_neg=11", "own_untagged=16", "cycles=2", "joined=2"):
        assert f" {part}" in line, line
    assert " ev_no_lift=1 " in line and line.endswith(" ev_release=0")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *argv[:4], "--host", "x"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.startswith("CONVERT_FAIL usage: --host")


def test_the_stock_dagger_pass_keeps_exactly_the_takeover_frames_of_an_all_frames_shard(tmp_path):
    """Policy, handoff and takeover frames interleave in a record_all shard; the stock pass
    keeps the same frames and cuts as on the shard that records the takeovers only."""
    actor = np.array([0, 0, 1, 1, 1, 0, 0, 0, 2, 2, -3, -3, 0, 4, 4, 0], np.int16)
    phase = np.array([-1, -1, 1, 2, 3, -1, 4, 5, 1, 2, 1, 2, -1, 1, 9, -1], np.int8)
    bricks = np.array([-1, -1, 0, 0, 0, -1, 1, 1, 1, 1, 2, 2, -1, 3, 3, -1], np.int8)
    full = _shard(tmp_path / "full.hdf5", frames=16, actor=actor, bricks=bricks)
    kept = actor != 0
    part = _shard(
        tmp_path / "part.hdf5", frames=int(kept.sum()), actor=actor[kept], bricks=bricks[kept]
    )
    for path, rows in ((full, slice(None)), (part, kept)):
        with h5py.File(path, "a") as f:
            f.attrs["gripper_mode"] = "continuous"
            g = f["data/demo_0"]
            g["obs/cycle_phase"][...] = phase[rows]
            state = np.zeros((16, 8), np.float32)
            state[:, 0] = np.arange(16)
            state[:, 6:8] = 0.008
            g["obs/joint_pos_abs"][...] = state[rows]
            actions = np.zeros((16, 7), np.float32)
            actions[:, 6] = 0.016
            g["actions"][...] = actions[rows]
    kw = {"split_cycles": True, "dagger": True, "context": "camera", "gripper": "continuous"}
    out = {}
    for name, path in (("full", full), ("part", part)):
        counts = c2l.convert(
            [path], tmp_path / name, "t/x", 30, images=False, reopen=c2l.ReopenLead(0.002), **kw
        )
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset("t/x", root=tmp_path / name)
        out[name] = (counts[:2], [float(ds[i]["observation.state"][0]) for i in range(len(ds))])
    # Three takeovers (the negated one dropped), the same frames in both: state[0] is the frame.
    assert out["full"] == out["part"] == ((3, 3), [2.0, 3.0, 4.0, 8.0, 9.0, 13.0, 14.0])


VIDEO = {
    "dtype": "video",
    "shape": [224, 224, 3],
    "names": ["height", "width", "channels"],
    "info": {
        "video.height": 224,
        "video.width": 224,
        "video.codec": "av1",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": 30,
        "video.channels": 3,
        "has_audio": False,
    },
}
CAMERA = ["overhead_cam_" + k for k in ("x", "y", "z", "roll", "pitch", "yaw")]
TARGET = ["target_" + k for k in ("x", "y", "z", "cos", "sin", "width", "half_height")]
INDEX = {"dtype": "int64", "shape": [1], "names": None}
CORPUS_FEATURES = {
    "action": {"dtype": "float32", "shape": [7], "names": c2l.ARM_JOINTS + ["gripper_gap_m"]},
    "observation.state": {
        "dtype": "float32",
        "shape": [7],
        "names": c2l.ARM_JOINTS + ["gripper_open"],
    },
    "observation.images.wrist": VIDEO,
    "observation.images.overhead": VIDEO,
    "observation.environment_state": {
        "dtype": "float32",
        "shape": [16],
        "names": CAMERA + ["box_x", "box_y", "box_yaw"] + TARGET,
    },
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": INDEX,
    "episode_index": INDEX,
    "index": INDEX,
    "task_index": INDEX,
}
"""The F9n corpus's features (~/data/maker-arm-ms7/g/w1234567/dataset/meta/info.json): the
ACT-C mix refuses an extra root whose features differ from it in any byte."""
CORPUS_INFO = Path.home() / "data/maker-arm-ms7/g/w1234567/dataset/meta/info.json"


def test_the_positive_and_takeover_roots_have_the_corpus_schema_and_marker(tmp_path):
    shard = _two_cycles(tmp_path)
    with h5py.File(shard, "a") as f:
        obs = f["data/demo_0/obs"]
        obs["actor"][60:80] = 1  # cycle 2 is a takeover: the stock pass converts it
        for key, names in (("camera", CAMERA), ("box", ["box_x", "box_y", "box_yaw"])):
            obs[f"context_{key}"].attrs["names"] = json.dumps(names)
        obs["context_target"].attrs["names"] = json.dumps(TARGET)
    kw = {"context": "camera_box_target", "gripper": "continuous"}
    own = c2l.OwnFrames("positive")
    assert c2l.convert([shard], tmp_path / "pos", "t/x", 30, own=own, **kw)[:2] == (1, 1)
    stock = {"split_cycles": True, "dagger": True, "reopen": c2l.ReopenLead(0.002)}
    assert c2l.convert([shard], tmp_path / "dag", "t/y", 30, **kw, **stock)[:2] == (1, 1)
    if CORPUS_INFO.exists():  # Machine A's corpus, the reference the constant copies
        assert json.loads(CORPUS_INFO.read_text())["features"] == CORPUS_FEATURES
    for root in ("pos", "dag"):
        info = json.loads((tmp_path / root / "meta" / "info.json").read_text())
        assert info["features"] == CORPUS_FEATURES and info["fps"] == 30, root
        marker = json.loads((tmp_path / root / "meta" / "action_space.json").read_text())
        assert marker == {"action_space": "absolute", "gripper": "continuous"}, root
    # The recorded subtasks ride along by frame index (the prompt codes' sidecar).
    with h5py.File(shard, "r") as f:
        recorded = f["data/demo_0/obs/subtask"][:]
    assert np.array_equal(np.load(tmp_path / "pos/meta/subtask.npy"), recorded[5:55])
    assert np.array_equal(np.load(tmp_path / "dag/meta/subtask.npy"), recorded[60:80])
    stock = _shard(tmp_path / "stock.hdf5", box=True, target=True)
    with h5py.File(stock, "a") as f:
        f.attrs["gripper_mode"] = "continuous"
    with pytest.raises(RuntimeError, match="without obs/subtask would join"):
        c2l.convert([stock], tmp_path / "dag", "t/y", 30, **kw)
    c2l.convert([stock], tmp_path / "plain", "t/z", 30, **kw)
    assert not (tmp_path / "plain/meta/subtask.npy").exists()


def test_an_event_on_a_teacher_frame_is_dropped():
    """The still scene's cycle 2 (2026-10-13 audit): the census's first close was the takeover's
    first frame, so off_target landed on a teacher frame. Events come from policy frames only;
    the takeover's own event still ends on the last policy frame."""
    cycle, actor, phase, subtask = _episode()
    actor[30:55], phase[30:55] = 1, 2
    row = _row(50, first_close_at=25, off_x=0.0, off_y=0.0, off_z=0.03)
    label, (info,) = _labels(row, cycle, actor, phase, subtask, cls="stowed")
    assert info["events"]["off_target"] == [] and info["events"]["takeover"] == [29]
    assert _window(label, 5, 29)
    phase[30:55], actor[30:55] = 4, 0  # a handoff's rig frames: the teacher's too
    _, (info,) = _labels(row, cycle, actor, phase, subtask, cls="stowed")
    assert info["events"]["off_target"] == []
