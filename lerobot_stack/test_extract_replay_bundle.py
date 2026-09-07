import h5py
import numpy as np
import pytest

from extract_replay_bundle import extract, initial_state_arrays


def _write_shard(path, demos=2, steps=5, num_bricks=2, record_dim=1):
    """A raw datagen shard with the schema `test_convert_roundtrip._write_shard` uses, plus
    the `initial_state` group that shard actually carries.

    The `initial_state` group's paths and entity names were verified by tracing
    IsaacLab's `InitialStateRecorder` / `InteractiveScene.get_state` /
    `HDF5DatasetFileHandler.write_episode` in the IsaacLab source (the existing
    roundtrip fixture predates initial-state recording and does not include it). The
    per-field *shapes* below -- a leading length-`record_dim` axis on every
    `initial_state` leaf -- were carried over from openyam-lab, where they were
    confirmed against a real shard on disk, and RE-CONFIRMED here on a Maker Arm
    datagen shard (Task D6's `/tmp/batch0/shard.hdf5`, all 7 demos), which likewise
    shows `(1, 8)` / `(1, 7)` / `(1, 6)` rather than the bare `(8,)` / `(7,)` /
    `(6,)` a source trace alone suggested: `EpisodeData.add` appends each
    `record_post_reset` call to a list, and `EpisodeData.pre_export` then
    `torch.stack`s that list before `write_episode`, which re-adds the leading dim that
    `RecorderManagerBase.add_to_episodes`'s per-env indexing had stripped. `initial_state`
    is recorded exactly once per episode, so `record_dim` is always 1 on real data; the
    default here matches that, and `record_dim != 1` is used below only to exercise the
    loud-failure path a wrongly-multi-recorded shard would hit.

    Per-demo values are offset by `d` and per-brick values by `b` so that a swapped
    demo/brick index, or a swapped actions/joint_targets source, produces a detectably
    wrong number rather than an accidental match.
    """
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["total"] = demos * steps
        for d in range(demos):
            g = data.create_group(f"demo_{d}")
            g.attrs["num_samples"] = steps
            g.attrs["success"] = True
            # raw pre-clamp actions: gripper column (6) is constant per demo and does not
            # overlap with any joint_targets value, so a column-source swap is detectable.
            g.create_dataset("actions", data=np.full((steps, 7), 0.5 + d, dtype=np.float32))
            g.create_dataset(
                "joint_targets",
                data=(np.arange(steps * 8, dtype=np.float32).reshape(steps, 8) + d * 100),
            )
            init = g.create_group("initial_state")
            robot = init.create_group("articulation").create_group("robot")
            robot.create_dataset(
                "joint_position",
                data=np.tile(np.arange(8, dtype=np.float32) + d, (record_dim, 1)),
            )
            robot.create_dataset(
                "joint_velocity",
                data=np.tile(np.full(8, 0.1 + d, dtype=np.float32), (record_dim, 1)),
            )
            robot.create_dataset(
                "root_pose",
                data=np.tile(np.array([d, 0, 0, 1, 0, 0, 0], dtype=np.float32), (record_dim, 1)),
            )
            robot.create_dataset(
                "root_velocity",
                data=np.tile(np.full(6, 0.2 + d, dtype=np.float32), (record_dim, 1)),
            )
            rigid = init.create_group("rigid_object")
            for b in range(num_bricks):
                brick = rigid.create_group(f"brick_{b}")
                brick.create_dataset(
                    "root_pose",
                    data=np.tile(
                        np.array([b, d, 0, 1, 0, 0, 0], dtype=np.float32), (record_dim, 1)
                    ),
                )
                brick.create_dataset(
                    "root_velocity",
                    data=np.tile(np.full(6, b + 0.5 + d, dtype=np.float32), (record_dim, 1)),
                )


def test_extract_writes_one_bundle_per_demo_with_expected_shapes(tmp_path):
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard, demos=2, steps=5, num_bricks=3)
    out_dir = tmp_path / "bundles"

    written = extract(shard, out_dir)

    assert written == 2
    assert sorted(p.name for p in out_dir.glob("*.npz")) == ["demo_0.npz", "demo_1.npz"]
    bundle = np.load(out_dir / "demo_0.npz")
    assert bundle["actions"].shape == (5, 7)
    assert bundle["actions"].dtype == np.float32
    assert bundle["robot_joint_position"].shape == (8,)
    assert bundle["robot_joint_velocity"].shape == (8,)
    assert bundle["robot_root_pose"].shape == (7,)
    assert bundle["robot_root_velocity"].shape == (6,)
    assert bundle["brick_root_pose"].shape == (3, 7)
    assert bundle["brick_root_velocity"].shape == (3, 6)
    # the fixture's initial_state leaves carry a leading length-1 "record" axis (matching
    # real shards, see _write_shard's docstring); an implementation that doesn't strip it
    # would produce (1, 8), (1, 7), and (3, 1, 7)/(3, 1, 6) for the brick fields instead.
    np.testing.assert_array_equal(bundle["robot_joint_position"], np.arange(8, dtype=np.float32))
    np.testing.assert_array_equal(bundle["robot_root_velocity"], np.full(6, 0.2, dtype=np.float32))


def test_actions_columns_are_not_swapped(tmp_path):
    # Pins the dataset-action recipe from convert_to_lerobot.convert(): arm columns 0-5 are
    # the POST-IK joint_targets, the gripper column 6 is the RAW actions value. A wrong
    # implementation that reads the arm columns from `actions` or the gripper from
    # `joint_targets` would fail this, because in the fixture the two sources never agree
    # (joint_targets varies per-step and per-demo; actions is a per-demo constant).
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard, demos=2, steps=5, num_bricks=1)
    out_dir = tmp_path / "bundles"

    extract(shard, out_dir)

    for d in range(2):
        with h5py.File(shard, "r") as f:
            targets = f[f"data/demo_{d}/joint_targets"][:]
            raw_actions = f[f"data/demo_{d}/actions"][:]
        bundle = np.load(out_dir / f"demo_{d}.npz")
        np.testing.assert_array_equal(bundle["actions"][:, :6], targets[:, :6])
        np.testing.assert_array_equal(bundle["actions"][:, 6], raw_actions[:, 6])
        # sanity: the two sources genuinely disagree in this fixture, or the assertions
        # above would pass under a swapped implementation too.
        assert not np.array_equal(bundle["actions"][:, :6], raw_actions[:, :6])
        assert not np.array_equal(bundle["actions"][:, 6], targets[:, 6])


def test_num_samples_attr_truncates_longer_arrays(tmp_path):
    # attrs["num_samples"] is authoritative; the underlying arrays can be longer (e.g. a
    # trailing padded/garbage tail). An implementation that uses len(actions) instead of
    # attrs["num_samples"] would emit 8 rows here instead of the correct 5.
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard, demos=1, steps=5, num_bricks=1)
    with h5py.File(shard, "a") as f:
        grp = f["data/demo_0"]
        original_targets = grp["joint_targets"][:]
        original_actions = grp["actions"][:]
        # pad marker (99.0) can't collide with any legitimate value in this fixture
        # (joint_targets and actions here only ever hold values < 40).
        extra_actions = np.full((8, 7), 99.0, dtype=np.float32)
        extra_actions[:5] = original_actions
        extra_targets = np.full((8, 8), 99.0, dtype=np.float32)
        extra_targets[:5] = original_targets
        del grp["actions"]
        del grp["joint_targets"]
        grp.create_dataset("actions", data=extra_actions)
        grp.create_dataset("joint_targets", data=extra_targets)
    out_dir = tmp_path / "bundles"

    extract(shard, out_dir)

    bundle = np.load(out_dir / "demo_0.npz")
    assert bundle["actions"].shape == (5, 7)
    np.testing.assert_array_equal(bundle["actions"][:, :6], original_targets[:5, :6])
    np.testing.assert_array_equal(bundle["actions"][:, 6], original_actions[:5, 6])
    assert not np.any(bundle["actions"] == 99.0)


@pytest.mark.parametrize(
    ("short_actions", "short_targets"),
    [(True, True), (True, False), (False, True)],
    ids=["both-short", "actions-short", "joint_targets-short"],
)
def test_num_samples_longer_than_arrays_errors_instead_of_short_reading(
    tmp_path, short_actions, short_targets
):
    """`grp[...][:n]` with n past the end returns fewer rows with NO error from h5py.

    The (True, True) case is the one that is otherwise completely silent: both
    arrays short-read to the same length, `np.concatenate` is happy, and the
    bundle is written with 3 of the 5 steps the shard claims. Replaying that
    bundle stops the demo two steps early and the eval reports a replay failure,
    i.e. "the harness is broken" -- the inverted verdict replay mode exists to
    prevent. The mixed cases would blow up inside `np.concatenate` even without
    the guard, so `match=` pins the message: a test that only asserted
    `pytest.raises(ValueError)` would pass on those two for the wrong reason.

    What this does NOT catch: a shard whose `num_samples` is wrong in the other
    direction (smaller than the arrays). That is legal and tested above by
    `test_num_samples_attr_truncates_longer_arrays` -- `num_samples` is
    authoritative and a padded tail is expected, so only a short read is refused.

    Built synthetically because the condition does not occur on real data: all 7 demos
    of Task D6's Maker Arm shard `/tmp/batch0/shard.hdf5` have
    num_samples == len(actions) == len(joint_targets) (3846-4589 rows each).
    """
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard, demos=1, steps=5, num_bricks=1)
    with h5py.File(shard, "a") as f:
        grp = f["data/demo_0"]
        assert grp.attrs["num_samples"] == 5
        for name, truncate in (("actions", short_actions), ("joint_targets", short_targets)):
            if truncate:
                rows = grp[name][:3]
                del grp[name]
                grp.create_dataset(name, data=rows)
    out_dir = tmp_path / "bundles"

    with pytest.raises(ValueError, match=r"num_samples=5 but arrays are"):
        extract(shard, out_dir)

    # Nothing written -- but only because demo_0, the sole demo here, is the one
    # that raises. extract() is not atomic in general; that is pinned separately
    # by test_a_truncated_demo_leaves_earlier_demos_on_disk below.
    assert list(out_dir.glob("*.npz")) == []


def test_a_truncated_demo_leaves_earlier_demos_on_disk(tmp_path):
    """extract() is NOT atomic: demos are written one at a time, in shard order.

    A characterization test, not a bug hunt -- it pins the documented
    partial-write behaviour so it cannot change silently. `out_dir` is left
    holding bundles for every demo BEFORE the bad one and none after it, which
    is why a consumer must check the exit status rather than infer success from
    the files being present.

    Catches a future change that alters when the raise fires -- hoisting
    validation into an up-front pass (demo_0.npz would be absent) or, worse,
    downgrading it to a skip (demo_2.npz would appear and no error would be
    raised) -- without updating extract()'s docstring to match.

    It does NOT assert that partial writes are the RIGHT design; the docstring
    argues that case. If atomicity is added deliberately, this test is expected
    to be rewritten alongside the docstring, which is the point of having it.
    """
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard, demos=3, steps=5, num_bricks=1)
    with h5py.File(shard, "a") as f:
        grp = f["data/demo_1"]  # the MIDDLE demo, so there is a before and an after
        for name in ("actions", "joint_targets"):
            rows = grp[name][:3]
            del grp[name]
            grp.create_dataset(name, data=rows)
    out_dir = tmp_path / "bundles"

    with pytest.raises(ValueError, match=r"demo_1: num_samples=5 but arrays are"):
        extract(shard, out_dir)

    assert sorted(p.name for p in out_dir.glob("*.npz")) == ["demo_0.npz"]


def test_episodes_filter_selects_only_named_demos(tmp_path):
    # An implementation that ignores --episodes, or off-by-ones the index parsing (e.g.
    # treats "1" as a count rather than a demo id), would either write all demos or write
    # the wrong one; this checks both the file set and the content of the selected file.
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard, demos=3, steps=4, num_bricks=1)
    out_dir = tmp_path / "bundles"

    written = extract(shard, out_dir, episodes=[1])

    assert written == 1
    assert sorted(p.name for p in out_dir.glob("*.npz")) == ["demo_1.npz"]
    bundle = np.load(out_dir / "demo_1.npz")
    # demo 1's joint_position initial state is arange(8) + 1
    np.testing.assert_array_equal(
        bundle["robot_joint_position"], np.arange(8, dtype=np.float32) + 1
    )


def test_multiple_recorded_initial_states_errors_instead_of_taking_the_first(tmp_path):
    # initial_state is recorded exactly once per episode on real shards (record_dim=1); a
    # shard where it was somehow recorded more than once must fail loudly rather than
    # silently take index 0, since that would replay a demo from the wrong starting state
    # with no error. Catches an implementation that does a blind `arr[0]` / `.squeeze()`
    # instead of validating the record-axis length first.
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard, demos=1, steps=4, num_bricks=1, record_dim=2)
    out_dir = tmp_path / "bundles"

    with pytest.raises(ValueError, match="expected one recorded initial state"):
        extract(shard, out_dir)


def test_episodes_filter_rejects_unknown_demo(tmp_path):
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard, demos=2, steps=4, num_bricks=1)
    out_dir = tmp_path / "bundles"

    with pytest.raises(ValueError, match="demo_5"):
        extract(shard, out_dir, episodes=[5])


def test_initial_state_arrays_returns_the_six_bundle_keys(tmp_path):
    shard = tmp_path / "shard.hdf5"
    _write_shard(shard)
    with h5py.File(shard, "r") as f:
        arrays = initial_state_arrays(f["data"]["demo_0"])
    assert sorted(arrays) == [
        "brick_root_pose",
        "brick_root_velocity",
        "robot_joint_position",
        "robot_joint_velocity",
        "robot_root_pose",
        "robot_root_velocity",
    ]
    assert arrays["robot_joint_position"].shape == (8,)
    assert arrays["brick_root_pose"].shape == (2, 7)
