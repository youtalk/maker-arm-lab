"""HDF5 shard(s) -> LeRobot v3 dataset (create or resume). Runs in lerobot_stack/.venv.

Observation state: the 6 URDF arm joint angles plus the gripper OPENING FRACTION
(spec section 4.2) -- the left jaw's prismatic position over its travel, clipped to
[0, 1]. The two jaws are driven independently in the sim (the vendored URDF's
``<mimic>`` is not honoured by the importer, Task D3), so the left jaw specifically is
the reference and the right jaw column is dropped.

Action stream: 6 post-IK arm joint targets + the raw binary gripper command
(spec section 5.4 item 4) -- what the ACT head outputs and the real robot's
joint-position interface consumes. The seventh action column is a COMMAND (>= 0 opens,
< 0 closes), not a position: it is deliberately left in its own units rather than
normalised into the state's, which is why the two features name it differently
(``gripper`` vs ``gripper_open``). fps follows the 30 Hz control rate.

Every exit path prints CONVERT_OK or CONVERT_FAIL on stdout. scripts/generate_dataset.py
parses that line, and a run that terminates without either reads to it as a HANG rather
than a failure -- a far worse outcome than a reported failure.
"""

import argparse
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from extract_replay_bundle import initial_state_arrays

TASK = "sort the lego bricks into the color-matched bins"
ARM_JOINTS = [
    "link_002_joint",
    "link_003_joint",
    "link_004_joint",
    "link_005_joint",
    "link_006_joint",
    "link_007_joint",
]
# Per-jaw prismatic travel of the vendored URDF (gripper_left_joint upper limit). Pinned here
# because this venv has no maker_arm_lab; tests/test_lerobot_schema_pin.py keeps it honest.
GRIPPER_TRAVEL = 0.0524125

FEATURES = {
    "action": {"dtype": "float32", "shape": (7,), "names": ARM_JOINTS + ["gripper"]},
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": ARM_JOINTS + ["gripper_open"],
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.overhead": {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    },
}


# PHASE_TO_INT pins (recorders.py): dropped outright when splitting.
SETTLE_INT, DONE_INT, ABANDON_INT = 0, 10, 11


def cycle_segments(phase: np.ndarray, brick: np.ndarray) -> list[np.ndarray]:
    """Frame-index segments, one per brick cycle.

    Drops SETTLE/DONE/ABANDON frames, then cuts each time a NEW non-negative
    brick value appears (differs from the last non-negative value seen).
    Retries never change the brick, so they stay inside their cycle; -1 frames
    (goals[i] and oracles[i] both None -- see recorders.py's CyclePhaseRecorder;
    NOT produced by an in-episode phase transition like RETREAT, since
    oracle_runtime._enter_phase re-selects the next goal synchronously, in the
    same post-step pass that ends the previous phase) stay with the open cycle
    if one is open, and kept frames before the first target attach to the
    first cycle.
    """
    keep = ~np.isin(phase, (SETTLE_INT, DONE_INT, ABANDON_INT))
    segments: list[list[int]] = []
    current: list[int] = []
    last_brick: int | None = None
    for t in np.nonzero(keep)[0]:
        b = int(brick[t])
        if b >= 0 and last_brick is not None and b != last_brick:
            segments.append(current)
            current = []
        if b >= 0:
            last_brick = b
        current.append(int(t))
    if current and last_brick is not None:
        segments.append(current)
    return [np.asarray(s) for s in segments]


def convert(
    shards: list[Path],
    root: Path,
    repo_id: str,
    fps: int,
    *,
    split_cycles: bool = False,
    sidecar_dir: Path | None = None,
) -> tuple[int, int]:
    """Convert `shards` into (or append them onto) the LeRobot dataset at `root`.

    When `sidecar_dir` is set, every sub-episode written under `split_cycles` gets a
    `{episode_index}.npz` sidecar holding its `cycle_phase`/`cycle_brick` slices plus
    `initial_state_arrays(grp)` (see extract_replay_bundle.py). That `initial_state` is
    the SOURCE DEMO's t=0 state -- pre-SETTLE, before any brick has been picked -- not
    the sub-episode's own first frame: for segment k>0 it describes a scene that never
    coexisted with the sub-episode (earlier bricks are still on the desk, not yet in a
    bin), and even segment 0's sidecar carries the pre-SETTLE pose rather than its own
    first recorded frame. It is not a replayable start state for the sub-episode by
    itself -- do not feed it to `scene.reset_to()` expecting it to reproduce where the
    sub-episode begins.
    """
    if sidecar_dir is not None and not split_cycles:
        raise ValueError("sidecar_dir requires split_cycles")
    # Resume iff root/meta/info.json (the local LeRobot dataset marker) exists. A bare
    # root.exists() check is not enough: the campaign supervisor pre-creates dataset_root as an
    # empty directory before the first conversion ever runs, so root.exists() is True with no
    # local metadata at all -- LeRobotDataset.resume then falls back to querying the HuggingFace
    # Hub for repo_id, which 404s (RepositoryNotFoundError) for a repo that was never pushed.
    if (root / "meta" / "info.json").exists():
        dataset = LeRobotDataset.resume(repo_id, root=root)
    else:
        if root.exists():
            if any(root.iterdir()):
                raise RuntimeError(
                    f"{root} exists, is non-empty, but has no meta/info.json -- refusing to "
                    "guess whether to create or resume; this looks like a partial or corrupt "
                    "dataset root"
                )
            # LeRobotDataset.create() requires root not to pre-exist; an empty pre-created
            # directory (e.g. the campaign supervisor's dataset_root before the first
            # conversion) is otherwise indistinguishable on disk from "nothing here yet", so
            # remove it and create fresh.
            root.rmdir()
        # A NEW root pairs one-to-one with a NEW sidecar_dir. A non-empty sidecar_dir here
        # means files from some earlier campaign (whose episode indices started at 0, same
        # as this fresh dataset's will) -- writing into it would silently overwrite or
        # interleave with that campaign's `{episode_index}.npz` files with nothing on disk
        # telling them apart. On resume (the branch above) this check does not run: an
        # existing sidecar_dir is expected to already hold this same dataset's earlier
        # episodes, and appending to it is correct.
        if sidecar_dir is not None and sidecar_dir.exists() and any(sidecar_dir.iterdir()):
            raise RuntimeError(
                f"{sidecar_dir} exists and is non-empty, but {root} is being created fresh -- "
                "refusing to guess whether these sidecars belong to this campaign or a previous "
                "one; use a fresh --sidecar_dir for a new dataset root, or resume into the "
                "existing --root that produced them"
            )
        dataset = LeRobotDataset.create(
            repo_id,
            fps=fps,
            features=FEATURES,
            root=root,
            robot_type="maker_arm",
            use_videos=True,
        )
    converted = 0
    for shard in shards:
        with h5py.File(shard, "r") as f:
            data = f["data"]
            for demo in sorted(data.keys(), key=lambda k: int(k.split("_")[1])):
                grp = data[demo]
                if not bool(grp.attrs.get("success", False)):
                    continue  # belt and braces: shards are success-only already
                actions = grp["actions"][:]  # (T, 7) raw pre-clamp oracle command
                targets = grp["joint_targets"][:]  # (T, 8)
                state = grp["obs/joint_pos_abs"][:]  # (T, 8): 6 arm + left/right jaw (m)
                wrist = grp["obs/images/wrist_rgb"][:]  # (T, 224, 224, 3) uint8
                overhead = grp["obs/images/overhead_rgb"][:]
                steps = min(len(actions), len(targets), len(state), len(wrist), len(overhead))
                # The right jaw is dropped, not averaged in: the two jaws are separately
                # actuated here, and the left one is the declared reference (module docstring).
                gripper_open = np.clip(state[:, 6] / GRIPPER_TRAVEL, 0.0, 1.0)
                state7 = np.concatenate([state[:, :6], gripper_open[:, None]], axis=1)
                if split_cycles:
                    if "obs/cycle_phase" not in grp or "obs/cycle_brick" not in grp:
                        raise RuntimeError(
                            f"{shard}:{demo} has no cycle channels -- --split_cycles needs "
                            "shards recorded with CyclePhaseRecorder/CycleBrickRecorder"
                        )
                    phase = grp["obs/cycle_phase"][:steps]
                    brick = grp["obs/cycle_brick"][:steps]
                    segments = cycle_segments(phase, brick)
                else:
                    segments = [np.arange(steps)]
                # Hoisted out of the segment loop: read+stack once per demo instead of once
                # per sub-episode, and raise (on a shard with cycle channels but no
                # initial_state group) before save_episode() writes any of this demo's
                # segments, instead of after some of them already landed.
                init_arrays = initial_state_arrays(grp) if sidecar_dir is not None else None
                for seg in segments:
                    # index of the episode about to be saved
                    episode_index = dataset.meta.total_episodes
                    for t in seg:
                        dataset.add_frame(
                            {
                                "action": np.concatenate([targets[t, :6], actions[t, 6:7]]).astype(
                                    np.float32
                                ),
                                "observation.state": state7[t].astype(np.float32),
                                "observation.images.wrist": wrist[t],
                                "observation.images.overhead": overhead[t],
                                "task": TASK,
                            }
                        )
                    dataset.save_episode()
                    if sidecar_dir is not None:
                        sidecar_dir.mkdir(parents=True, exist_ok=True)
                        np.savez(
                            sidecar_dir / f"{episode_index}.npz",
                            cycle_phase=phase[seg],
                            cycle_brick=brick[seg],
                            **init_arrays,
                        )
                    converted += 1
    dataset.finalize()
    return converted, dataset.meta.total_episodes


class TokenArgumentParser(argparse.ArgumentParser):
    """An ArgumentParser whose usage errors still print a token.

    argparse's own ``error()`` writes usage to stderr and exits 2 with nothing on stdout, so
    a bad flag would look identical to a wedged process to the supervisor that reads this
    script's stdout. ``--help`` is deliberately NOT covered: it exits through ``exit()``
    rather than ``error()``, and a machine-readable verdict printed after a help text would
    be nonsense.
    """

    def error(self, message: str) -> None:
        print(f"CONVERT_FAIL usage: {message}", flush=True)
        super().error(message)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser()
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo_id", default="youtalk/maker-arm-legosort")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--split_cycles", action="store_true")
    parser.add_argument(
        "--sidecar_dir",
        type=Path,
        default=None,
        help=(
            "Directory for one {episode_index}.npz replay-label sidecar per sub-episode "
            "(requires --split_cycles). Each sidecar's initial_state is its SOURCE DEMO's "
            "t=0 (pre-SETTLE) state, not the sub-episode's own start -- see convert()'s "
            "docstring. Must be empty (or a fresh path) when --root is being created; "
            "reuse it only to resume into that same --root."
        ),
    )
    return parser


def main() -> int:
    # Everything after this point is inside the token contract: argparse failures go through
    # TokenArgumentParser.error, and any other failure becomes CONVERT_FAIL below. Nothing
    # that can fail on user input runs at module scope, where no token could be printed.
    parser = build_parser()
    try:
        args = parser.parse_args()
        if args.sidecar_dir is not None and not args.split_cycles:
            parser.error("--sidecar_dir requires --split_cycles")
        converted, total = convert(
            args.shards,
            args.root,
            args.repo_id,
            args.fps,
            split_cycles=args.split_cycles,
            sidecar_dir=args.sidecar_dir,
        )
    except Exception as exc:  # noqa: BLE001 -- any failure must become CONVERT_FAIL, not a crash.
        print(f"CONVERT_FAIL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    print(f"CONVERT_OK episodes={converted} total={total}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
