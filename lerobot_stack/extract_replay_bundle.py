"""Raw datagen shard -> per-episode replay bundle npz.

Extracts exactly what's needed to replay a recorded oracle demonstration back through
the eval environment: the action sequence plus the exact initial state it started
from. Replaying these and confirming they still succeed tells the eval team "the
policy is bad" apart from "the eval harness is broken" -- an eval harness's `--replay_dir`
mode consumes the files this script writes.

Action recipe mirrors convert_to_lerobot.convert() exactly: 6 post-IK arm joint targets
from `joint_targets[:, :6]` (what a joint-position policy outputs and the real robot's
CAN interface consumes) plus the raw, never-binarized gripper command from column 6 of
`actions` (g >= 0 opens, g < 0 closes). That asymmetry is deliberate, not a typo: the
arm columns must be the post-IK absolute targets so replay reproduces what the
controller actually applied, while the gripper column must stay the raw oracle command
so replay reproduces the un-binarized signal the policy was trained on.

`initial_state` is IsaacLab's `InitialStateRecorder` output: `scene.get_state(is_relative=True)`
(`isaaclab/envs/mdp/recorders/recorders.py`). Every leaf array on disk carries an extra
leading length-1 axis (`EpisodeData.pre_export` stacks the single `record_post_reset`
call into a list before `write_episode`), which this module strips -- loudly, not with a
blind squeeze -- to produce the plain shapes below. Because the state is *relative*, a
replay must feed it back with `scene.reset_to(..., is_relative=True)`; the default `False`
would place every brick at the world origin with no error, which is a silent, not a loud,
failure.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def _first(ds: h5py.Dataset) -> np.ndarray:
    """Strip the recorder's leading record axis, refusing to guess if it isn't length 1.

    `initial_state` is recorded exactly once per episode on every real shard; a shard
    where a leaf was recorded more than once is not something to silently truncate --
    replaying from `arr[0]` while ignoring `arr[1:]` would start the replay from an
    arbitrary one of several recorded states with no error.
    """
    arr = ds[:]
    if arr.shape[0] != 1:
        raise ValueError(f"expected one recorded initial state, got {arr.shape[0]}: {ds.name}")
    return arr[0].astype(np.float32)


def initial_state_arrays(grp) -> dict[str, "np.ndarray"]:
    """The six initial-state arrays of one demo group, keyed as the bundle npz keys."""
    init = grp["initial_state"]
    robot = init["articulation"]["robot"]
    rigid = init["rigid_object"]
    brick_names = sorted(rigid.keys(), key=lambda k: int(k.split("_")[1]))
    return {
        "robot_joint_position": _first(robot["joint_position"]),
        "robot_joint_velocity": _first(robot["joint_velocity"]),
        "robot_root_pose": _first(robot["root_pose"]),
        "robot_root_velocity": _first(robot["root_velocity"]),
        "brick_root_pose": np.stack([_first(rigid[b]["root_pose"]) for b in brick_names]),
        "brick_root_velocity": np.stack([_first(rigid[b]["root_velocity"]) for b in brick_names]),
    }


def extract(shard: Path, out_dir: Path, episodes: list[int] | None = None) -> int:
    """Write one `out_dir/demo_{n}.npz` replay bundle per selected demo in `shard`.

    Args:
        shard: path to a raw datagen HDF5 shard (`data/demo_{n}` groups).
        out_dir: directory to write `demo_{n}.npz` bundles into; created if missing.
        episodes: demo indices to extract. `None` extracts every demo in the shard.
            Naming a demo that does not exist is an error, not a silent skip.

    Returns:
        The number of bundles written.

    Raises:
        ValueError: on a malformed demo -- a multi-recorded `initial_state`
            (`_first`) or a `num_samples` longer than the arrays it indexes.

    NOT ATOMIC. Demos are validated and written one at a time in shard order, so
    a demo that raises leaves the bundles for every demo before it already on
    disk, and none for the demos after it. `out_dir` therefore holds a PARTIAL
    set after a failure, and the returned count is only meaningful on success.
    Anything consuming a bundle directory must not treat "the files are there"
    as "the extraction succeeded" -- check the exit status. An eval harness's
    `--replay_dir` is safe on this point for a different reason: it demands one
    bundle per env lane and fails loudly on the first missing one.

    Not made atomic on purpose. An up-front validation pass could hoist the
    `num_samples` check, but not `_first`'s -- that one needs the leaf datasets
    it is already reading -- so the result would be a guarantee that holds for
    one failure mode and not the other, which is worse than a documented "no
    guarantee". Write to a fresh `out_dir` per extraction if atomicity matters.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with h5py.File(shard, "r") as f:
        data = f["data"]
        demo_names = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
        if episodes is not None:
            wanted = [f"demo_{n}" for n in episodes]
            missing = [name for name in wanted if name not in data]
            if missing:
                raise ValueError(f"--episodes names demo(s) not present in {shard}: {missing}")
            demo_names = wanted

        for demo in demo_names:
            grp = data[demo]
            n = int(grp.attrs["num_samples"])  # authoritative length; arrays may be longer

            actions_raw = grp["actions"][:n]  # (n, 7) raw pre-clamp oracle command
            targets = grp["joint_targets"][:n]  # (n, 8) post-IK absolute joint targets
            # h5py slices past the end without complaining, so an `n` LARGER than
            # the stored array yields a short bundle, not an error. The eval
            # replay loader accepts any T >= 1, so a truncated demo would replay
            # to a premature stop and be read as "the harness is broken" -- the
            # exact inverted verdict this pipeline exists to prevent. `n` smaller
            # than the arrays stays legal (that is the documented padded-tail
            # case, and the slice above is what handles it); only a short read is
            # refused. Same refusal-to-guess as `_first` above.
            if len(actions_raw) != n or len(targets) != n:
                raise ValueError(
                    f"{demo}: num_samples={n} but arrays are "
                    f"{len(actions_raw)}/{len(targets)} rows after slicing "
                    f"(actions/joint_targets) -- the shard is truncated"
                )
            actions = np.concatenate([targets[:, :6], actions_raw[:, 6:7]], axis=1).astype(
                np.float32
            )  # (n, 7): arm from targets, gripper raw (never binarized)

            arrays = initial_state_arrays(grp)
            np.savez(out_dir / f"{demo}.npz", actions=actions, **arrays)
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--episodes", type=str, default=None, help="comma-separated demo indices, e.g. 0,1,2"
    )
    args = parser.parse_args()
    episodes = [int(e) for e in args.episodes.split(",")] if args.episodes else None
    written = extract(args.shard, args.out_dir, episodes)
    print(f"EXTRACT_OK bundles={written}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
