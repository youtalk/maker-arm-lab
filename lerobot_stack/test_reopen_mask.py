"""reopen_mask.py (the MS7g corpus mask) on tiny synthetic shards, in the LeRobot venv."""

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import convert_to_lerobot as c2l
import reopen_mask as rm

SCRIPT = Path(__file__).resolve().parent / "reopen_mask.py"
OPEN, HELD, PRE = rm.FULL_OPEN_M, 0.005, 0.02
HOVER, DESCEND, CLOSE, LIFT, REOPEN, RELEASE, RETREAT = 1, 2, 3, 4, 5, 8, 9


def fake_pick(q):
    """The test shards write the pick point straight into joints 0-2."""
    return np.asarray(q)[:, :3]


def attempt(rise, reopen=True):
    """(phase, gripper, z, x) of one grasp: HOVER, DESCEND, CLOSE, a LIFT that rises ``rise``,
    then a still REOPEN (or a moving RELEASE + RETREAT over a box when ``reopen`` is False)."""
    phase = [HOVER] * 3 + [DESCEND] * 3 + [CLOSE] * 3 + [LIFT] * 5
    gripper = [PRE] * 6 + [HELD] * 8
    z = [0.1] * 3 + [0.05, 0.02, 0.0] + [0.0] * 3 + list(np.linspace(0.0, rise, 5))
    x = [0.0] * 14
    if reopen:
        phase += [REOPEN] * rm.REOPEN_FRAMES
        gripper += [OPEN] * rm.REOPEN_FRAMES
        z += [rise] * rm.REOPEN_FRAMES
        x += [0.0] * rm.REOPEN_FRAMES
    else:  # carried away, released over a box at height, then the retreat moves the arm
        phase += [RELEASE] * 4 + [RETREAT] * rm.REOPEN_FRAMES
        gripper += [PRE] * 4 + [OPEN] * rm.REOPEN_FRAMES
        z += [rise] * (4 + rm.REOPEN_FRAMES)
        x += [0.3] * 4 + list(0.3 + 0.01 * np.arange(rm.REOPEN_FRAMES))
    return phase, gripper, z, x


def write_demo(path, demo, attempts, bricks, actors):
    """One demo: the attempts back to back, attempt k with brick bricks[k] and actor actors[k]."""
    cols = [[], [], [], [], [], []]
    for (phase, gripper, z, x), brick, actor in zip(attempts, bricks, actors):
        for col, values in zip(
            cols, (phase, gripper, z, x, [brick] * len(phase), [actor] * len(phase))
        ):
            col.extend(values)
    phase, gripper, z, x, brick, actor = (np.asarray(c) for c in cols)
    n = len(phase)
    mode = "a" if Path(path).exists() else "w"
    with h5py.File(path, mode) as f:
        f.attrs["gripper_mode"] = "continuous"
        g = f.create_group(f"data/demo_{demo}")
        g.attrs["success"] = False
        actions = np.zeros((n, 7), np.float32)
        actions[:, 6] = gripper
        g.create_dataset("actions", data=actions)
        g.create_dataset("joint_targets", data=np.zeros((n, 8), np.float32))
        state = np.zeros((n, 8), np.float32)
        state[:, 0], state[:, 2] = x, z
        g.create_dataset("obs/joint_pos_abs", data=state)
        for cam in ("wrist_rgb", "overhead_rgb"):
            g.create_dataset(f"obs/images/{cam}", data=np.zeros((n, 224, 224, 3), np.uint8))
        g.create_dataset("obs/cycle_phase", data=phase.astype(np.int8))
        g.create_dataset("obs/cycle_brick", data=brick.astype(np.int16))
        g.create_dataset("obs/actor", data=actor.astype(np.int16))
        for name, width in (("camera", 6), ("box", 3), ("target", 7)):
            ds = g.create_dataset(f"obs/context_{name}", data=np.zeros((n, width), np.float32))
            ds.attrs["names"] = json.dumps([f"{name}_{k}" for k in range(width)])
    return path


def corpus(tmp_path):
    """Two shards. Shard 0: a risen REOPEN (dropped), a grasp-height REOPEN (kept), a policy
    stretch (actor -1, not converted), a risen REOPEN on a new brick, then a delivery whose
    RETREAT opens high but moves. Shard 1: a risen REOPEN and a delivery."""
    s0 = write_demo(
        tmp_path / "proc_0.hdf5",
        0,
        [attempt(0.10), attempt(0.01), attempt(0.10), attempt(0.10), attempt(0.10, reopen=False)],
        bricks=[3, 3, 3, 4, 4],
        actors=[2, 2, -1, 3, 3],  # actor 1 would go too: -1 negates it (cycle_segments)
    )
    s1 = write_demo(
        tmp_path / "proc_1.hdf5",
        0,
        [attempt(0.08), attempt(0.09, reopen=False)],
        bricks=[0, 0],
        actors=[1, 1],
    )
    return [s0, s1]


def converted(tmp_path, shards):
    root = tmp_path / "dataset"
    c2l.convert(
        shards,
        root,
        rm.REPO_ID,
        30,
        split_cycles=True,
        dagger=True,
        context="camera_box_target",
        gripper="continuous",
    )
    return root


def test_truth_drops_the_reopen_after_a_risen_lift_only():
    phase, _, z, _ = (np.asarray(c) for c in attempt(0.10))
    regrasp_phase, _, regrasp_z, _ = (np.asarray(c) for c in attempt(0.01))
    phase, z = np.r_[phase, regrasp_phase], np.r_[z, regrasp_z]
    labels = rm.truth_labels(phase, z[phase == LIFT])
    reopen = np.flatnonzero(phase == REOPEN)
    assert np.array_equal(np.flatnonzero(labels["drop"]), reopen[: rm.REOPEN_FRAMES])
    assert labels["reopen"].sum() == 2 * rm.REOPEN_FRAMES
    assert labels["risen"].sum() == labels["risen_reopen"].sum() == 1


def test_a_risen_lift_that_carries_drops_nothing():
    phase, _, z, _ = (np.asarray(c) for c in attempt(0.10, reopen=False))
    labels = rm.truth_labels(phase, z[phase == LIFT])
    assert not labels["drop"].any() and labels["risen"].sum() == 1
    assert labels["risen_reopen"].sum() == 0


def test_lane_order_is_convert_sh_lane_order():
    assert rm.lane_order(list(range(5)), 2) == [0, 2, 4, 1, 3]
    assert rm.lane_order(list(range(3)), 6) == [0, 1, 2]


def test_truth_reproduces_the_conversion_and_drops_exactly_the_risen_reopens(tmp_path):
    shards = corpus(tmp_path)
    root = converted(tmp_path, shards)
    lengths, drop, counts = rm.truth_mask(shards, pick_points=fake_pick)
    rm.check_lengths(lengths, root)  # raises on any difference from the converted root
    _, state, action = rm.read_frames(root)
    # Three risen REOPENs kept by the converter (the policy stretch is not converted).
    assert len(drop) == 3 * rm.REOPEN_FRAMES
    assert np.all(np.abs(action[drop, 6] - OPEN) < rm.OPEN_TOL_M)
    assert np.all(state[drop, 2] >= rm.RISE_MIN_M)
    assert counts["reopen"] == 4 * rm.REOPEN_FRAMES  # plus the grasp-height one
    assert counts["risen"] == 5 and counts["risen_reopen"] == 3


def test_the_proxy_agrees_with_the_truth(tmp_path):
    shards = corpus(tmp_path)
    root = converted(tmp_path, shards)
    _, truth, _ = rm.truth_mask(shards, pick_points=fake_pick)
    lengths, proxy = rm.proxy_mask(*rm.read_frames(root), pick_points=fake_pick)
    rm.check_lengths(lengths, root)
    agreement = rm.compare(truth, proxy)
    assert agreement["recall"] == agreement["precision"] == 1.0


def test_a_wrong_lane_order_is_refused(tmp_path):
    shards = corpus(tmp_path)
    root = converted(tmp_path, shards)
    lengths, _, _ = rm.truth_mask(shards[::-1], pick_points=fake_pick)
    with pytest.raises(RuntimeError, match="first differing episode"):
        rm.check_lengths(lengths, root)


def test_the_proxy_keeps_a_still_open_at_grasp_height_and_a_moving_open_up_high():
    regrasp = attempt(0.01)
    retreat = attempt(0.10, reopen=False)
    for phase, gripper, z, x in (regrasp, retreat):
        pick = np.stack([x, np.zeros(len(z)), z], axis=1)
        assert len(rm.proxy_drop(np.asarray(gripper), pick)) == 0


def test_compose_offsets_each_part_by_the_frames_before_it():
    a = (np.array([3, 2]), np.array([1, 4]))
    b = (np.array([4]), np.array([0, 3]))
    lengths, drop = rm.compose([a, b])
    assert lengths.tolist() == [3, 2, 4] and drop.tolist() == [1, 4, 5, 8]


def test_save_and_load_round_trip(tmp_path):
    summary = rm.save(tmp_path / "m.npz", np.array([5, 5]), np.array([2, 7]), rule="truth")
    assert summary["dropped"] == 2 and summary["share"] == 0.2
    lengths, drop = rm.load(tmp_path / "m.npz")
    assert lengths.tolist() == [5, 5] and drop.tolist() == [2, 7]


def _run(*argv):
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, check=False
    )
    return out.returncode, out.stdout.strip().splitlines()


def test_a_usage_error_prints_the_fail_token():
    code, lines = _run("truth", "--lanes", "2")
    assert code == 2 and lines[-1].startswith("MASK_FAIL usage:")


def test_compose_against_the_wrong_root_prints_the_fail_token(tmp_path):
    shards = corpus(tmp_path)
    root = converted(tmp_path, shards)
    rm.save(tmp_path / "m.npz", np.array([1, 2]), np.zeros(0, np.int64))
    mask, out = str(tmp_path / "m.npz"), str(tmp_path / "out.npz")
    code, lines = _run("compose", "--dataset", str(root), "--masks", mask, "--out", out)
    assert code == 1 and lines[-1].startswith("MASK_FAIL RuntimeError")


def test_compose_of_one_root_prints_the_ok_token(tmp_path):
    shards = corpus(tmp_path)
    root = converted(tmp_path, shards)
    lengths, drop, _ = rm.truth_mask(shards, pick_points=fake_pick)
    rm.save(tmp_path / "m.npz", lengths, drop)
    mask, out = str(tmp_path / "m.npz"), str(tmp_path / "out.npz")
    code, lines = _run("compose", "--dataset", str(root), "--masks", mask, "--out", out)
    summary = json.loads(lines[-1].removeprefix("MASK_OK "))
    assert code == 0 and summary["dropped"] == len(drop) == 3 * rm.REOPEN_FRAMES
