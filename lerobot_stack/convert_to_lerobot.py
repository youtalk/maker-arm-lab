"""HDF5 shard(s) -> LeRobot v3 dataset (create or resume). Runs in lerobot_stack/.venv.

Observation state: the 6 URDF arm joint angles plus the gripper OPENING FRACTION
(spec section 4.2) -- the left jaw's prismatic position over its travel, clipped to
[0, 1]. The two jaws are driven independently in the sim (the vendored URDF's
``<mimic>`` is not honoured by the importer, Task D3), so the left jaw specifically is
the reference and the right jaw column is dropped.

Action stream: 6 post-IK arm joint targets + the raw gripper command (spec section 4.2 --
the same table as the state above; section 5 is an unsubdivided list, so the "5.4 item 4"
this line carried over from openyam-lab pointed nowhere) -- what the ACT head outputs and
the real robot's joint-position interface consumes. The seventh action column follows the
shard's gripper contract (``--gripper``, phase B design 5.4). Under ``binary`` (every corpus
before phase B) it is a COMMAND (>= 0 opens, < 0 closes), feature name ``gripper``. Under
``continuous`` it is the target jaw gap in meters, 0.0 shut to twice ``GRIPPER_TRAVEL``
open, feature name ``gripper_gap_m``. Either way it is left in its own units rather than
normalised into the state's opening fraction, which is why the state names its column
``gripper_open``. fps follows the 30 Hz control rate.

Every exit path prints CONVERT_OK or CONVERT_FAIL on stdout. scripts/generate_dataset.py
parses that line, and a run that terminates without either reads to it as a HANG rather
than a failure -- a far worse outcome than a reported failure.

``--context`` additionally reads a shard's ``obs/context_<name>`` dataset(s) and writes
them as the single feature ``observation.environment_state``. ``camera`` and ``privileged``
read one channel each; ``camera_box`` (MS6 heap shards) concatenates the camera pose and the
destination box pose into one 9-wide row, camera first, and ``camera_box_target`` adds the
7-wide target row (``obs/context_target``) after the box. The context choices never mix in one
dataset: one ``--context`` per conversion. ``--no_images`` drops both image features (the
privileged-state ceiling) and requires ``--context`` -- a dataset with neither images nor a
context channel has nothing to condition the policy on.

``--dagger`` also turns the ``success`` attribute filter OFF. A heap run records EVERY episode
(mj/recorder.py ``export_all``, MS6 design H7) and its ``success`` is True only when all of its
parts stowed, which almost never happens, so filtering on it throws the corpus away before the
positive-actor cut ever runs. Without ``--dagger`` the filter stays, because a six-brick shard
is success-only by construction.

``--action_space delta_prev`` (MS4 c2 design 4.2) replaces the joint part of ``action`` with
this frame's target minus the previous frame's target; the gripper column is unchanged.
On a segment's first frame, the anchor is the measured joint position. The policy server
adds the client's anchor back before sending an absolute command the clamp accepts.
A ``meta/action_space.json`` marker records which space a dataset was created with, and
resuming into it with the other space is refused.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import NamedTuple

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

# The gripper contract of a shard's action column (MS6 phase B design 5.4). "binary" is the
# raw >= 0 opens command every corpus before phase B was recorded with; "continuous" is the
# jaw gap in meters. The recorder writes it as the HDF5 file attribute `gripper_mode`, a
# dataset never mixes the two, and the name of the action's seventh feature says which.
GRIPPER_MODES = ("binary", "continuous")
GRIPPER_NAMES = {"binary": "gripper", "continuous": "gripper_gap_m"}

BASE_FEATURES = {
    "action": {"dtype": "float32", "shape": (7,), "names": ARM_JOINTS + ["gripper"]},
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": ARM_JOINTS + ["gripper_open"],
    },
}
IMAGE_FEATURES = {
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
# What each --context choice reads from a shard, in concatenation order. "camera_box" is the
# MS6 heap channel (design section 9): the overhead camera pose followed by the destination box
# pose, 6 + 3 = 9 wide, matching mj/context_names.py's CONTEXT_DIMS["camera_box"].
CONTEXT_SOURCES = {
    "camera": ("camera",),
    "privileged": ("privileged",),
    "camera_box": ("camera", "box"),
    "camera_box_target": ("camera", "box", "target"),
}
CONTEXT_CHANNELS = tuple(CONTEXT_SOURCES)
ACTION_SPACES = ("absolute", "delta_prev")
ACTION_SPACE_MARKER = "action_space.json"


class Counts(NamedTuple):
    """What one conversion did, straight onto the CONVERT_OK line.

    ``read`` and ``skipped`` let a supervisor tell "this corpus had nothing in it" apart from
    "the success filter ate it": a heap shard converted without ``--dagger`` reports a large
    ``skipped``, the same shard converted with it reports 0.
    """

    episodes: int
    total: int
    read: int
    skipped: int


def write_action_space_marker(root: Path, action_space: str, gripper: str = "binary") -> None:
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / ACTION_SPACE_MARKER).write_text(
        json.dumps({"action_space": action_space, "gripper": gripper})
    )


def read_action_space_marker(root: Path) -> dict:
    """The marker as written, with the pre-phase-B defaults for a marker that lacks a key."""
    marker = root / "meta" / ACTION_SPACE_MARKER
    recorded = json.loads(marker.read_text()) if marker.exists() else {}
    return {
        "action_space": recorded.get("action_space", "absolute"),
        "gripper": recorded.get("gripper", "binary"),
    }


def check_action_space_marker(root: Path, action_space: str, gripper: str = "binary") -> None:
    """Refuse to append actions of one space or gripper contract to a dataset of the other."""
    recorded = read_action_space_marker(root)
    if recorded["action_space"] != action_space:
        raise RuntimeError(
            f"{root} was converted with action_space={recorded['action_space']}, "
            f"refusing to append {action_space}"
        )
    if recorded["gripper"] != gripper:
        raise RuntimeError(
            f"{root} was converted with gripper={recorded['gripper']}, refusing to append {gripper}"
        )


def build_features(context_names: list[str] | None, images: bool, gripper: str = "binary") -> dict:
    """The dataset features for one conversion: state and action always, images unless
    dropped, and ``observation.environment_state`` when a context channel is selected."""
    features = dict(BASE_FEATURES)
    features["action"] = {**BASE_FEATURES["action"], "names": ARM_JOINTS + [GRIPPER_NAMES[gripper]]}
    if images:
        features.update(IMAGE_FEATURES)
    if context_names is not None:
        features["observation.environment_state"] = {
            "dtype": "float32",
            "shape": (len(context_names),),
            "names": list(context_names),
        }
    return features


def context_keys(context: str | None) -> list[str]:
    """The shard datasets one ``--context`` choice reads, in concatenation order."""
    return [] if context is None else [f"obs/context_{s}" for s in CONTEXT_SOURCES[context]]


def read_context_names(shard: Path, context: str) -> list[str]:
    """The ``names`` attributes of the first demo's context datasets, concatenated in order."""
    names: list[str] = []
    with h5py.File(shard, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for key in context_keys(context):
            if not demos or key not in f["data"][demos[0]]:
                raise RuntimeError(
                    f"{shard} has no {key} dataset -- --context {context} needs shards recorded "
                    "with a recorder that writes every channel it reads"
                )
            names += json.loads(f["data"][demos[0]][key].attrs["names"])
    return names


# PHASE_TO_INT pins (recorders.py): dropped outright when splitting.
SETTLE_INT, DONE_INT, ABANDON_INT = 0, 10, 11
CORRECTIVE_INTS = (12, 13, 14, 15, 16, 17)
"""PUSH_HOVER..PUSH_SWEEP and TIP_HOVER..TIP_SWEEP, also dropped when splitting. The only
correctives a shard holds since 2026-09-18 are the MS7a tray rakes (the densest-cell push and
the tip-up are off), recorded under the next pick's actor with no target: cycle_brick -1,
target_part -1 and an all-zero target row. Kept, each rake became a training episode of its
own; the policy never rakes (the serving side runs it, --scripted_rake), so they go."""


def cycle_segments(
    phase: np.ndarray, brick: np.ndarray, lead_keep: int = 0, actor: np.ndarray | None = None
) -> list[np.ndarray]:
    """Frame-index segments, one per brick cycle.

    ``lead_keep`` attaches the LAST that many SETTLE frames to the first cycle (openyam-lab's
    ``--lead_keep 60``, MS3 recipe design 5.3): a short hold at the start state stays in the
    corpus, the 240-frame fixed point does not. 0 keeps none.

    Drops SETTLE/DONE/ABANDON and the corrective (CORRECTIVE_INTS) frames, then cuts each
    time a NEW non-negative
    brick value appears (differs from the last non-negative value seen).
    Retries never change the brick, so they stay inside their cycle; -1 frames
    (goals[i] and oracles[i] both None -- see recorders.py's CyclePhaseRecorder;
    NOT produced by an in-episode phase transition like RETREAT, since
    oracle_runtime._enter_phase re-selects the next goal synchronously, in the
    same post-step pass that ends the previous phase) stay with the open cycle
    if one is open, and kept frames before the first target attach to the
    first cycle.

    ``actor`` (MS4 c2 design 6.5, ``--dagger``) is the recorder's ``obs/actor``: frames whose
    actor is not positive are dropped, and a segment is cut wherever the actor changes, so
    two takeovers on the same brick never share a chunk horizon across the policy's missing
    frames.
    """
    keep = ~np.isin(phase, (SETTLE_INT, DONE_INT, ABANDON_INT, *CORRECTIVE_INTS))
    if actor is not None:
        keep &= actor > 0
    segments: list[list[int]] = []
    current: list[int] = []
    last_brick: int | None = None
    last_actor: int | None = None
    for t in np.nonzero(keep)[0]:
        b = int(brick[t])
        a = None if actor is None else int(actor[t])
        new_brick = b >= 0 and last_brick is not None and b != last_brick
        new_actor = a is not None and last_actor is not None and a != last_actor
        if new_brick or new_actor:
            segments.append(current)
            current = []
        if b >= 0:
            last_brick = b
        last_actor = a
        current.append(int(t))
    if current and last_brick is not None:
        segments.append(current)
    if segments and lead_keep > 0:
        first = segments[0][0]
        lead = [t for t in range(max(0, first - lead_keep), first) if phase[t] == SETTLE_INT]
        segments[0] = lead + segments[0]
    return [np.asarray(s) for s in segments]


def convert(
    shards: list[Path],
    root: Path,
    repo_id: str,
    fps: int,
    *,
    split_cycles: bool = False,
    sidecar_dir: Path | None = None,
    lead_keep: int = 0,
    context: str | None = None,
    images: bool = True,
    action_space: str = "absolute",
    dagger: bool = False,
    gripper: str = "binary",
) -> Counts:
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
    if action_space not in ACTION_SPACES:
        raise ValueError(f"action_space must be one of {ACTION_SPACES}, got {action_space!r}")
    if gripper not in GRIPPER_MODES:
        raise ValueError(f"gripper must be one of {GRIPPER_MODES}, got {gripper!r}")
    if sidecar_dir is not None and not split_cycles:
        raise ValueError("sidecar_dir requires split_cycles")
    if lead_keep and not split_cycles:
        raise ValueError("lead_keep requires split_cycles")
    if dagger and not split_cycles:
        raise ValueError("dagger requires split_cycles")
    if dagger and lead_keep:
        raise ValueError("dagger refuses lead_keep: takeover frames have no SETTLE lead")
    if context is not None and context not in CONTEXT_CHANNELS:
        raise ValueError(f"context must be one of {CONTEXT_CHANNELS}, got {context!r}")
    if not images and context is None:
        raise ValueError("a dataset without images needs a context channel (--context)")
    context_names = read_context_names(shards[0], context) if context is not None else None
    keys = context_keys(context)
    # Resume iff root/meta/info.json (the local LeRobot dataset marker) exists. A bare
    # root.exists() check is not enough: the campaign supervisor pre-creates dataset_root as an
    # empty directory before the first conversion ever runs, so root.exists() is True with no
    # local metadata at all -- LeRobotDataset.resume then falls back to querying the HuggingFace
    # Hub for repo_id, which 404s (RepositoryNotFoundError) for a repo that was never pushed.
    if (root / "meta" / "info.json").exists():
        check_action_space_marker(root, action_space, gripper)
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
            features=build_features(context_names, images, gripper),
            root=root,
            robot_type="maker_arm",
            use_videos=True,
        )
        write_action_space_marker(root, action_space, gripper)
    converted = read = skipped = 0
    for shard in shards:
        with h5py.File(shard, "r") as f:
            recorded_gripper = str(f.attrs.get("gripper_mode", "binary"))
            if recorded_gripper != gripper:
                raise RuntimeError(
                    f"{shard} was recorded with gripper_mode={recorded_gripper}, "
                    f"--gripper is {gripper}"
                )
            data = f["data"]
            for demo in sorted(data.keys(), key=lambda k: int(k.split("_")[1])):
                grp = data[demo]
                read += 1
                # A six-brick shard is success-only already, so this is belt and braces there.
                # A heap shard is not: it records every episode and its `success` means "every
                # part stowed", so under --dagger the positive-actor cut is the filter and this
                # one would throw nearly the whole corpus away before it ran (module docstring).
                if not dagger and not bool(grp.attrs.get("success", False)):
                    skipped += 1
                    continue
                actions = grp["actions"][:]  # (T, 7) raw pre-clamp oracle command
                targets = grp["joint_targets"][:]  # (T, 8)
                state = grp["obs/joint_pos_abs"][:]  # (T, 8): 6 arm + left/right jaw (m)
                wrist = grp["obs/images/wrist_rgb"] if images else None  # (T, 224, 224, 3) uint8
                overhead = grp["obs/images/overhead_rgb"] if images else None
                lengths = [len(actions), len(targets), len(state)]
                if images:
                    lengths += [len(wrist), len(overhead)]
                if keys:
                    for key in keys:
                        if key not in grp:
                            raise RuntimeError(f"{shard}:{demo} has no {key} dataset")
                    ctx_steps = min(len(grp[key]) for key in keys)
                    ctx = np.concatenate([grp[key][:ctx_steps] for key in keys], axis=1)
                    lengths.append(ctx_steps)
                steps = min(lengths)
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
                    actor = None
                    if dagger:
                        if "obs/actor" not in grp:
                            raise RuntimeError(f"{shard}:{demo} has no obs/actor dataset")
                        actor = grp["obs/actor"][:steps]
                    segments = cycle_segments(phase, brick, lead_keep, actor)
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
                    previous = None
                    for t in seg:
                        joints = targets[t, :6]
                        if action_space == "delta_prev":
                            # MS4 c2 design 4.2: a per-step velocity of the COMMAND. The first
                            # frame of a segment anchors on the measured joints, every later
                            # frame on the previous frame's post-clamp target, so the command
                            # gap keeps growing when the arm lags (the c1 delta hugged the
                            # measured pose and stalled the arm).
                            anchor = state[t, :6] if previous is None else targets[previous, :6]
                            joints = targets[t, :6] - anchor
                        previous = t
                        frame = {
                            "action": np.concatenate([joints, actions[t, 6:7]]).astype(np.float32),
                            "observation.state": state7[t].astype(np.float32),
                            "task": TASK,
                        }
                        if images:
                            frame["observation.images.wrist"] = wrist[t]
                            frame["observation.images.overhead"] = overhead[t]
                        if keys:
                            frame["observation.environment_state"] = ctx[t].astype(np.float32)
                        dataset.add_frame(frame)
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
    return Counts(converted, dataset.meta.total_episodes, read, skipped)


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
        "--lead_keep",
        type=int,
        default=0,
        help="with --split_cycles: keep the last N SETTLE frames on each demo's first cycle",
    )
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
    parser.add_argument(
        "--context",
        choices=CONTEXT_CHANNELS,
        default=None,
        help="write obs/context_<name> as observation.environment_state (MS4 levers 1 and 3); "
        "camera_box concatenates the camera pose and the MS6 destination box pose, 9 wide",
    )
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="drop both image features (the privileged-state ceiling); requires --context",
    )
    parser.add_argument(
        "--action_space",
        choices=ACTION_SPACES,
        default="absolute",
        help=(
            "delta_prev: joint part = this frame's target minus the previous frame's target "
            "(the measured joints on a segment's first frame). The policy server adds the "
            "client's anchor back."
        ),
    )
    parser.add_argument(
        "--dagger",
        action="store_true",
        help="keep only the oracle takeover cycles of a DAgger shard (positive obs/actor), "
        "cut on actor change; also drops the success-attribute filter, since a heap run "
        "records every episode",
    )
    parser.add_argument(
        "--gripper",
        choices=GRIPPER_MODES,
        default="binary",
        help="the shards' gripper contract; must match the recorder's gripper_mode attribute",
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
        counts = convert(
            args.shards,
            args.root,
            args.repo_id,
            args.fps,
            split_cycles=args.split_cycles,
            lead_keep=args.lead_keep,
            sidecar_dir=args.sidecar_dir,
            context=args.context,
            images=not args.no_images,
            action_space=args.action_space,
            dagger=args.dagger,
            gripper=args.gripper,
        )
    except Exception as exc:  # noqa: BLE001 -- any failure must become CONVERT_FAIL, not a crash.
        print(f"CONVERT_FAIL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    print(
        f"CONVERT_OK episodes={counts.episodes} total={counts.total} "
        f"read={counts.read} skipped={counts.skipped}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
