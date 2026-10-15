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

``--drop_reopen_lead TOL_M`` (MS7h DAgger round, HD59; recipe 3.4: closed jaws that the teacher
opens at once teach the policy's re-open) drops, per takeover episode, the leading frames whose
measured jaw gap (left + right jaw) lies more than TOL_M under the commanded gap (``reopen_lead``).
It needs ``--dagger --gripper continuous`` and adds its counts to the CONVERT_OK line. Off by
default.

``--own_frames {labeled,positive}`` (MS7i rev3 design 5.3) converts the policy's own frames of a
``mj_eval_heap.py --dagger_dir --record_all`` shard. It reads the shard's payload
(``<shard>.meta.json``, the client's ``--out``), joins each census row to the frames of its
``obs/loop_cycle`` value (the frame count must equal the row's ``open_steps + closed_steps``, or
the conversion fails) and labels every policy frame (``own_labels``): ``TAG_NEGATIVE`` on the
``NEG_WINDOW`` frames that end at a failing event, ``TAG_POSITIVE`` on a clean grasp segment and
on a carry the policy ends in its served box, ``TAG_UNTAGGED`` otherwise. ``labeled`` writes every
policy frame with the int8 columns ``label`` and ``subtask``, which no policy reads (neither name
starts with ``observation`` or ``action``); ``positive`` writes only the positive frames in the
stock schema, so the root merges with a corpus. Takeover, handoff and rake frames are dropped and
counted: the takeovers come from the stock ``--split_cycles --dagger --drop_reopen_lead`` pass on
the same shards. ``meta/own_frames.json`` keeps one row per cycle for the audit sampler. Any
conversion of record_all shards also writes ``meta/subtask.npy`` (``SUBTASK_META``). It
refuses a shard that ``dag_mask.py`` rewrote (``obs/actor_raw``).
"""

import argparse
import json
import math
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

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


@dataclass
class ReopenLead:
    """--drop_reopen_lead: the tolerance, and what a conversion dropped with it."""

    tol: float
    frames: int = 0
    takeovers: int = 0  # takeover episodes that lost a lead
    emptied: int = 0  # takeovers that lost every frame, so were not written


# MS7i --own_frames. The label codes are the interface's TAG_* values; the rest are copies of
# core (this venv has no maker_arm_lab): hold.SUBTASKS, oracle.LIFT_MIN_RISE,
# loop.CENSUS_MIN_HELD_STEPS and the loop.RELEASE_PLACES a release fails or ends in a box on,
# pinned by tests/test_lerobot_schema_pin.py.
OWN_MODES = ("labeled", "positive")
TAG_UNTAGGED, TAG_POSITIVE, TAG_NEGATIVE = 0, 1, 2
SUBTASKS = ("reach", "grasp", "carry", "release")
CARRY = SUBTASKS.index("carry")
LIFT_MIN_RISE = 0.05
CENSUS_MIN_HELD_STEPS = 10
FAILING_PLACES = ("tray", "desk", "rim")
BOX_PLACES = ("right_box", "wrong_box")
NEG_WINDOW = 30  # frames that end at a failing event, the event's included (rev3 design 5.3)
REOPEN_WITHIN = 10  # steps from a close to the open that makes it a failing re-open
REOPEN_MIN_CLOSED = 3  # steps: a shorter close is a command flicker (the round 1 audit)
RELEASE_LEAD = 10  # steps: a positive carry opens first at most this long before its release
OFF_TARGET_M = 0.020  # m, the MS7h jaw table's off-target first close: off_z or the xy distance
EVENTS = ("off_target", "reopen", "no_lift", "takeover", "lost", "release")
OWN_META = "own_frames.json"
SUBTASK_META = "subtask.npy"
"""``meta/subtask.npy``: the recorded obs/subtask of every frame of a root, by frame index, kept
while every frame of the root has one (record_all shards), for Run A's prompt codes. A takeover
that starts after a lift keeps the loop's carry, which an offline pass over the takeover alone
would read as grasp."""


@dataclass
class OwnFrames:
    """--own_frames: the mode, the host override, and what a conversion counted and joined."""

    mode: str
    host: str | None = None
    counts: Counter = field(default_factory=Counter)
    rows: list = field(default_factory=list)  # one per census row, for meta/own_frames.json


def own_labels(rows, classes, reasons, final_box, cycle, actor, phase, subtask):
    """One episode's per-frame labels (int8 ``TAG_*``) and one dict per census row.

    ``rows``, ``classes`` and ``reasons`` are the episode's ``cycle_census``, ``cycle_classes``
    and ``cycle_reasons``, ``final_box`` its box per part; the rest are the shard's per-frame
    obs/loop_cycle, obs/actor, obs/cycle_phase and obs/subtask. A policy frame has phase -1 and
    actor 0; only policy frames get a label other than untagged. The join: the frames of row
    j are the frames with loop_cycle j + 1, contiguous and as many as its census steps, and no
    frame names a cycle without a row; anything else raises. A frame reads closed in the grasp
    and carry subtasks; a close is an open-to-closed edge (or a cycle that starts closed), a
    re-open a closed-to-open edge. The events (frame e, a policy frame of the cycle: one on a
    takeover's or a handoff's frame is the teacher's, and dropped): ``off_target`` the first
    close with off_z or the xy offset over ``OFF_TARGET_M``; ``reopen`` each re-open within
    ``REOPEN_WITHIN`` steps of its close, when that close lasted ``REOPEN_MIN_CLOSED`` steps or
    more and came before the first carry frame (a grasp that hesitates, not a command flicker or
    a release); ``no_lift`` the last close of a cycle that lifted
    nothing (max_rise under ``LIFT_MIN_RISE``) and did not stow; ``takeover`` the last frame
    before the teacher's first, or the last frame of a ``takeover_declined`` cycle; ``lost`` the
    frame after the last contact-held step of a lifted cycle, if it reads closed and the release
    puts the part in no box (a slip; a part the opening jaws let go of over a box is none);
    ``release`` the open after the lift onto ``FAILING_PLACES``. The positives: the grasp
    segment (cycle start to the first carry frame) of a cycle with a hold and a lift, no re-open
    before the carry and no rig frame, whose census hold began by the first carry frame (a
    cycle that starts closed can read a carry before any hold); the carry segment (the first
    carry frame to the cycle end) of such a cycle when the policy ends it alone (no handoff, no rig frame) with a release over a box, no
    open more than ``RELEASE_LEAD`` steps before that release, and its part, held by no later
    cycle, at rest in the served box. A negative window wins."""
    n = len(cycle)
    policy = (phase == -1) & (actor == 0)
    closed = np.isin(subtask, (SUBTASKS.index("grasp"), CARRY))
    label = np.full(n, TAG_UNTAGGED, np.int8)
    if n and int(cycle.max()) > len(rows):
        raise RuntimeError(f"obs/loop_cycle reaches {int(cycle.max())} with {len(rows)} rows")
    out = []
    for j, row in enumerate(rows):
        frames = np.flatnonzero(cycle == j + 1)
        steps = row["open_steps"] + row["closed_steps"]
        if len(frames) != steps or (len(frames) and frames[-1] - frames[0] + 1 != len(frames)):
            raise RuntimeError(f"cycle {j + 1}: {len(frames)} frames against {steps} census steps")
        info = {"ordinal": j + 1, "first": None, "last": None, "carry_start": None}
        info["events"] = {name: [] for name in EVENTS}
        out.append(info)
        if not len(frames):
            continue
        s, last = int(frames[0]), int(frames[-1])
        info["first"], info["last"] = s, last
        span = range(s, last + 1)
        shut = [t for t in span if closed[t] and (t == s or not closed[t - 1])]
        opened = [t for t in span if t > s and closed[t - 1] and not closed[t]]
        hits = []  # (event, frame), kept when the frame is a policy frame of the cycle
        if row["first_close_at"] is not None and row["off_z"] is not None:
            off_xy = math.hypot(row["off_x"], row["off_y"])
            if row["off_z"] > OFF_TARGET_M or off_xy > OFF_TARGET_M:
                hits.append(("off_target", s + row["first_close_at"]))
        carry = [t for t in span if subtask[t] == CARRY]
        for r in opened:
            c = max((t for t in shut if t < r), default=None)
            grasping = not carry or c is not None and c < carry[0]
            if c is not None and REOPEN_MIN_CLOSED <= r - c <= REOPEN_WITHIN and grasping:
                hits.append(("reopen", r))
        lifted = row["max_rise"] >= LIFT_MIN_RISE
        if not lifted and classes[j] != "stowed" and shut:
            hits.append(("no_lift", shut[-1]))
        teacher = np.flatnonzero(actor[s : last + 1] != 0)
        if len(teacher):
            hits.append(("takeover", s + int(teacher[0]) - 1))
        elif reasons[j] == "takeover_declined":
            hits.append(("takeover", last))
        if lifted and row["last_held_at"] is not None and row["release_place"] not in BOX_PLACES:
            e = s + row["last_held_at"] + 1
            if e <= last and closed[e]:
                hits.append(("lost", e))
        if row["release_place"] in FAILING_PLACES and row["release_at"] is not None:
            hits.append(("release", s + row["release_at"]))
        for name, e in hits:
            if s <= e <= last and policy[e]:
                info["events"][name].append(int(e))
                window = slice(max(s, e - NEG_WINDOW + 1), e + 1)
                label[window][policy[window]] = TAG_NEGATIVE
        if not carry:
            continue
        start = info["carry_start"] = carry[0]
        grasp, rest = slice(s, start), slice(start, last + 1)
        held = (  # the census hold began by the carry start, else the carry read is false
            row["held_steps"] >= CENSUS_MIN_HELD_STEPS
            and lifted
            and row["first_held_at"] is not None
            and s + row["first_held_at"] <= start
        )
        if held and not any(r < start for r in opened) and policy[grasp].all():
            label[grasp][label[grasp] != TAG_NEGATIVE] = TAG_POSITIVE
        part = row["last_held"]
        again = any(part in (later["first_held"], later["last_held"]) for later in rows[j + 1 :])
        if (
            held
            and row.get("handoff") is None
            and policy[rest].all()
            and row["release_at"] is not None
            and row["release_place"] in BOX_PLACES
            and not any(start < r < s + row["release_at"] - RELEASE_LEAD for r in opened)
            and part >= 0
            and not again
            and final_box[part] == row["box"]
        ):
            label[rest][label[rest] != TAG_NEGATIVE] = TAG_POSITIVE
    return label, out


def own_runs(keep: np.ndarray, cycle: np.ndarray) -> list[np.ndarray]:
    """Maximal runs of kept frames, consecutive in time and of one loop_cycle value."""
    idx = np.flatnonzero(keep)
    if not len(idx):
        return []
    cuts = np.flatnonzero((np.diff(idx) != 1) | (np.diff(cycle[idx]) != 0)) + 1
    return np.split(idx, cuts)


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
    frames. A positive actor that also appears negated is dropped too: the batch loop negates
    a failed cycle's frames when the teacher gives up at LIFT (``negate_new_failures``), and
    the give-up REOPEN recorded after that keeps the positive ordinal. Kept, those 20 frames
    (jaws opening over the failed part at lift height) became a training episode of their
    own, 187 of the 2553 in MS7a's r1/dataset_w1.
    """
    keep = ~np.isin(phase, (SETTLE_INT, DONE_INT, ABANDON_INT, *CORRECTIVE_INTS))
    if actor is not None:
        keep &= (actor > 0) & ~np.isin(actor, -actor[actor < 0])
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


def reopen_lead(gap: np.ndarray, command: np.ndarray, tol: float) -> int:
    """The leading frames of one takeover in which the teacher opens jaws the policy left closed:
    the measured gap lies more than ``tol`` under the commanded gap. The lead ends at the first
    frame where the jaws have opened to within ``tol``, or where the command falls under the gap
    (the teacher's first close). A takeover that starts open, or holding a part, loses nothing.
    The rule of the HD59 recorder's dag_mask.py, so the two drop the same frames."""
    opening = gap < command - tol
    return len(opening) if opening.all() else int(np.argmin(opening))


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
    reopen: ReopenLead | None = None,
    own: OwnFrames | None = None,
) -> Counts:
    """Convert `shards` into (or append them onto) the LeRobot dataset at `root`.

    When `sidecar_dir` is set, every sub-episode written under `split_cycles` gets a
    `{episode_index}.npz` sidecar holding its `cycle_phase`/`cycle_brick` slices.
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
    if reopen is not None and not (dagger and gripper == "continuous"):
        raise ValueError("reopen reads a gap command: it needs dagger and gripper continuous")
    if context is not None and context not in CONTEXT_CHANNELS:
        raise ValueError(f"context must be one of {CONTEXT_CHANNELS}, got {context!r}")
    if not images and context is None:
        raise ValueError("a dataset without images needs a context channel (--context)")
    if own is not None:
        if own.mode not in OWN_MODES:
            raise ValueError(f"own_frames must be one of {OWN_MODES}, got {own.mode!r}")
        if split_cycles or dagger or lead_keep or sidecar_dir is not None or reopen is not None:
            raise ValueError(
                "own_frames refuses split_cycles, dagger, lead_keep, sidecar_dir and reopen: "
                "the stock --dagger pass converts the takeovers"
            )
        if gripper != "continuous" or context != "camera_box_target":
            raise ValueError("own_frames needs gripper continuous and context camera_box_target")
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
        features = build_features(context_names, images, gripper)
        if own is not None and own.mode == "labeled":
            for name in ("label", "subtask"):
                features[name] = {"dtype": "int8", "shape": (1,), "names": [name]}
        dataset = LeRobotDataset.create(
            repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type="maker_arm",
            use_videos=True,
        )
        write_action_space_marker(root, action_space, gripper)
    converted = read = skipped = 0
    sidecar = root / "meta" / SUBTASK_META
    frames_before, subtasks, lacking = dataset.meta.total_frames, [], False
    if sidecar.exists() and not all(recorded_subtasks(shard) for shard in shards):
        raise RuntimeError(f"{sidecar}: a shard without obs/subtask would join a root that has it")
    for shard in shards:
        with h5py.File(shard, "r") as f:
            recorded_gripper = str(f.attrs.get("gripper_mode", "binary"))
            if recorded_gripper != gripper:
                raise RuntimeError(
                    f"{shard} was recorded with gripper_mode={recorded_gripper}, "
                    f"--gripper is {gripper}"
                )
            data = f["data"]
            payload = None
            if own is not None:
                meta = shard.with_name(shard.name.removesuffix(".hdf5") + ".meta.json")
                payload = json.loads(meta.read_text())
                if Path(payload.get("shard", "")).name != shard.name or not payload.get(
                    "record_all"
                ):
                    raise RuntimeError(f"{meta} is not the record_all payload of {shard.name}")
            for demo in sorted(data.keys(), key=lambda k: int(k.split("_")[1])):
                grp = data[demo]
                read += 1
                # A six-brick shard is success-only already, so this is belt and braces there.
                # A heap shard is not: it records every episode and its `success` means "every
                # part stowed", so under --dagger the positive-actor cut is the filter and this
                # one would throw nearly the whole corpus away before it ran (module docstring).
                if not dagger and own is None and not bool(grp.attrs.get("success", False)):
                    skipped += 1
                    continue
                actions = grp["actions"][:]  # (T, 7) raw pre-clamp oracle command
                recorded = grp["obs/subtask"][:] if "obs/subtask" in grp else None
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
                elif own is not None:
                    segments, columns, cycle, infos = own_segments(
                        own, shard, demo, grp, payload, steps
                    )
                else:
                    segments = [np.arange(steps)]
                for seg in segments:
                    if reopen is not None:
                        cut = reopen_lead(
                            state[seg, 6] + state[seg, 7], actions[seg, 6], reopen.tol
                        )
                        seg = seg[cut:]
                        reopen.frames += cut
                        reopen.takeovers += cut > 0
                        if not len(seg):
                            reopen.emptied += 1
                            continue
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
                        if own is not None and own.mode == "labeled":
                            for name, column in columns.items():
                                frame[name] = column[t : t + 1]
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    if recorded is None:
                        lacking = True
                    else:
                        subtasks.append(recorded[seg])
                    if own is not None and cycle[seg[0]] > 0:  # the audit's frame map
                        infos[cycle[seg[0]] - 1]["written"].append(
                            [episode_index, int(seg[0]), len(seg)]
                        )
                    if sidecar_dir is not None:
                        sidecar_dir.mkdir(parents=True, exist_ok=True)
                        np.savez(
                            sidecar_dir / f"{episode_index}.npz",
                            cycle_phase=phase[seg],
                            cycle_brick=brick[seg],
                        )
                    converted += 1
    dataset.finalize()
    # The sidecar exists only while every frame of the root has its recorded subtask.
    if subtasks and not lacking and (sidecar.exists() or frames_before == 0):
        before = np.load(sidecar) if sidecar.exists() else np.zeros(0, np.int8)
        np.save(sidecar, np.concatenate([before, *subtasks]).astype(np.int8))
    if own is not None:
        meta = root / "meta" / OWN_META
        before = json.loads(meta.read_text())["cycles"] if meta.exists() else []
        meta.write_text(json.dumps({"cycles": before + own.rows}))
    return Counts(converted, dataset.meta.total_episodes, read, skipped)


def recorded_subtasks(shard: Path) -> bool:
    """Every demo of ``shard`` records obs/subtask (a record_all shard)."""
    with h5py.File(shard, "r") as f:
        return all("obs/subtask" in demo for demo in f["data"].values())


def own_segments(own, shard, demo, grp, payload, steps):
    """--own_frames on one demo: the runs to write, the per-frame columns of ``labeled`` (label,
    subtask), obs/loop_cycle and the joined cycle rows, which also go to ``own.rows``; adds the
    demo's counts to ``own.counts``."""
    if "obs/actor_raw" in grp:
        raise RuntimeError(f"{shard}:{demo} has obs/actor_raw: dag_mask.py rewrote its actors")
    if "obs/loop_cycle" not in grp or "env_index" not in grp.attrs:
        raise RuntimeError(f"{shard}:{demo} was not recorded with --record_all")
    env_index = int(grp.attrs["env_index"])
    ep = next(e for e in payload["episodes"] if e["env_index"] == env_index)
    if steps != ep["steps"]:
        raise RuntimeError(f"{shard}:{demo} has {steps} frames, its episode {ep['steps']} steps")
    cycle = grp["obs/loop_cycle"][:steps]
    actor = grp["obs/actor"][:steps]
    phase = grp["obs/cycle_phase"][:steps]
    subtask = grp["obs/subtask"][:steps]
    rows = ep["cycle_census"]
    label, cycles = own_labels(
        rows,
        ep["cycle_classes"],
        ep["cycle_reasons"],
        ep["final_box"],
        cycle,
        actor,
        phase,
        subtask,
    )
    policy = (phase == -1) & (actor == 0)
    host = own.host or shard.parent.name
    for info in cycles:
        tags = label[cycle == info["ordinal"]][policy[cycle == info["ordinal"]]]
        info["labels"] = {
            "positive": int((tags == TAG_POSITIVE).sum()),
            "negative": int((tags == TAG_NEGATIVE).sum()),
            "untagged": int((tags == TAG_UNTAGGED).sum()),
        }
        info |= {"host": host, "shard": str(shard), "demo": demo, "written": []}
        for name, hits in info["events"].items():
            own.counts[f"ev_{name}"] += len(hits)
    own.rows.extend(cycles)
    c = own.counts
    c["cycles"] += len(rows)
    c["joined"] += sum(info["first"] is not None for info in cycles)
    lifted = [r["max_rise"] >= LIFT_MIN_RISE for r in rows]
    c["lift_no_carry"] += sum(up and i["carry_start"] is None for up, i in zip(lifted, cycles))
    c["own_pos"] += int((label[policy] == TAG_POSITIVE).sum())
    c["own_neg"] += int((label[policy] == TAG_NEGATIVE).sum())
    c["own_untagged"] += int((label[policy] == TAG_UNTAGGED).sum())
    c["rig_frames"] += int(((actor == 0) & (phase >= 0)).sum())
    c["takeover_frames"] += int((actor != 0).sum())
    keep = policy if own.mode == "labeled" else label == TAG_POSITIVE
    columns = {"label": label, "subtask": subtask.astype(np.int8)}
    return own_runs(keep, cycle), columns, cycle, cycles


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
            "Directory for one {episode_index}.npz sidecar of cycle_phase/cycle_brick labels "
            "per sub-episode (requires --split_cycles). Must be empty (or a fresh path) when "
            "--root is being created; reuse it only to resume into that same --root."
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
    parser.add_argument(
        "--drop_reopen_lead",
        type=float,
        default=None,
        metavar="TOL_M",
        help="with --dagger --gripper continuous: drop each takeover's leading frames whose jaw "
        "gap lies more than TOL_M under the commanded gap (the teacher opening closed jaws)",
    )
    parser.add_argument(
        "--own_frames",
        choices=OWN_MODES,
        default=None,
        help="MS7i: the policy frames of record_all shards, joined to <shard>.meta.json's census; "
        "labeled adds the int8 label and subtask columns, positive writes only positive frames",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="with --own_frames: the host each cycle row of meta/own_frames.json names "
        "(default: the shard's directory name)",
    )
    return parser


def main() -> int:
    # Everything after this point is inside the token contract: argparse failures go through
    # TokenArgumentParser.error, and any other failure becomes CONVERT_FAIL below. Nothing
    # that can fail on user input runs at module scope, where no token could be printed.
    parser = build_parser()
    try:
        args = parser.parse_args()
        tol = args.drop_reopen_lead
        reopen = None if tol is None else ReopenLead(tol)
        if args.sidecar_dir is not None and not args.split_cycles:
            parser.error("--sidecar_dir requires --split_cycles")
        if args.host is not None and args.own_frames is None:
            parser.error("--host names the host of --own_frames rows")
        own = None if args.own_frames is None else OwnFrames(args.own_frames, args.host)
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
            reopen=reopen,
            own=own,
        )
    except Exception as exc:  # noqa: BLE001 -- any failure must become CONVERT_FAIL, not a crash.
        print(f"CONVERT_FAIL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    dropped = ""
    if reopen is not None:
        dropped = (
            f" reopen_tol={reopen.tol} reopen_frames={reopen.frames}"
            f" reopen_takeovers={reopen.takeovers} reopen_emptied={reopen.emptied}"
        )
    if own is not None:
        keys = ["own_pos", "own_neg", "own_untagged", "cycles", "joined", "lift_no_carry"]
        keys += ["rig_frames", "takeover_frames"] + [f"ev_{name}" for name in EVENTS]
        dropped += f" own_frames={own.mode}" + "".join(f" {k}={own.counts[k]}" for k in keys)
    print(
        f"CONVERT_OK episodes={counts.episodes} total={counts.total} "
        f"read={counts.read} skipped={counts.skipped}{dropped}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
