"""The generation-run loop and its records, shared by both simulator backends.

``scripts/generate_batch.py`` (Isaac Lab) and ``scripts/mj_generate_batch.py`` (MuJoCo) differ
only in how they build the environment. Everything that decides what a run RECORDS lives here,
so the two backends' ``RUN_META`` / ``RUN_STATS`` lines and ``<shard>.meta.json`` sidecars carry
the same fields with the same meanings and a campaign ledger cannot tell them apart except by
the ``backend`` field. The duck-typed env surface is the one ``OracleRig`` reads; see
``maker_arm_lab.mj.env`` for the list.

The campaign target the ``--probe`` extrapolation is quoted against stays the MS1 number (50
successes) for the same reason it did in generate_batch.py: it is a probe convention, not a
campaign size.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch

from maker_arm_lab.core import desk_boxes, layout, oracle, perception, trajectory
from maker_arm_lab.core import loop as _loop
from maker_arm_lab.core import parts as _parts
from maker_arm_lab.core.heap_oracle import CYCLE_BUDGET
from maker_arm_lab.core.layout import sample_run_draw

CAMPAIGN_TARGET_SUCCESSES = 50
EVENTS = _loop.EVENT_NAMES  # MS7f: a heap run's events, counted per episode by its track

PARTS_SPLITS = ("train", "heldout", "all")
LEGACY_EPISODE_LENGTH_S = 180.0


def heap_episode_length_s(k: int, cycle_budget: int) -> float:
    """Design section 9: 30 s of settle and slack plus 20 s per pick the budget allows. With
    ``heap_cycle_budget`` it scales with the heap: 1030 s at 50 parts, 3030 s (90,900 steps) at
    the MS7f heap of 150, where the MS7e teacher's 14.3 s a cycle at 1.04 cycles a part needs
    about 2230 s (MS7f design section 3)."""
    return 30.0 + 20.0 * min(k, cycle_budget)


def heap_cycle_budget(k: int) -> int:
    """The recorder's default cycle budget for a heap of ``k`` parts: ``heap_oracle.CYCLE_BUDGET``
    (50) up to 50 parts, the budget of every corpus through MS7e, and above that the heap loop's
    own ceil(1.3 K) (``loop.cycle_budget``): 195 for the MS7f heap of 150 (MS7f design section
    5), so a recording and a read of one heap get the same cycles."""
    return CYCLE_BUDGET if k <= CYCLE_BUDGET else _loop.cycle_budget(k)


def heap_parts(args) -> list[_parts.Part] | None:
    """The run's part set, or None on the six-brick task (``--heap_k 0``).

    The split is seeded by ``--parts_seed`` so the held-out set is reproducible, and the
    run's draw from it by ``--layout_seed``, so two lanes never share a part set (design H4).
    The catalog holds the families ``--families`` names (MS7e). ``--heap_family KEY`` (a family
    heap read, MS7e design section 6.3) draws K // 2 parts from that family's side of the split
    and the rest from the split's other parts, then shuffles them, so the family's parts do
    not all drop first, to the bottom of the heap.
    """
    if args.heap_k <= 0:
        return None
    train, heldout = _parts.split(_parts.catalog(getattr(args, "families", ())), args.parts_seed)
    pool = {"train": train, "heldout": heldout, "all": train + heldout}[args.parts_split]
    rng = random.Random(args.layout_seed)
    key = getattr(args, "heap_family", None)
    if not key:
        return _parts.sample_parts(rng, pool, args.heap_k)
    own = [p for p in pool if p.family in _parts.FAMILY_KEYS[key]]
    rest = [p for p in pool if p.family not in _parts.FAMILY_KEYS[key]]
    half = args.heap_k // 2
    heap = _parts.sample_parts(rng, own, half) + _parts.sample_parts(rng, rest, args.heap_k - half)
    rng.shuffle(heap)
    return heap


def heldout_names(args) -> frozenset[str]:
    """The part kinds of the held-out split of the run's catalog (``--families``, ``--parts_seed``),
    the kinds neither the policy nor the classifier saw: the held-out row of MS7f design section
    4.4, meaningful under ``--parts_split all``."""
    catalog = _parts.catalog(getattr(args, "families", ()))
    return frozenset(p.name for p in _parts.split(catalog, args.parts_seed)[1])


def heap_classes(args) -> tuple[str, ...]:
    """The active class list: the eight MS7e classes whenever ``--families`` names a family or
    the box draw is on, else the four MS7a classes, so with every MS7e switch off the class
    map and the perception's class draws are MS7d's (MS7e design section 5). Under MS7f
    ``--categories`` it is the run's active categories (``parts.active_categories``)."""
    if getattr(args, "categories", False):
        return _parts.active_categories(getattr(args, "families", ()))
    on = getattr(args, "families", ()) or getattr(args, "box_draw", 0)
    return _parts.PART_CLASSES if on else _parts.BASE_CLASSES


UNBUILT_SWITCHES: tuple[str, ...] = ()
"""The MS7e switches whose behaviour is not in this commit: any value but off is a usage
error. The boxes and release-lean tracks remove theirs as they land. The gate supervisor keeps
a stdlib copy (``scripts/run_gate_eval.py``), pinned by a test."""


def families_arg(text: str) -> tuple[str, ...]:
    """``--families``: a comma list of ``parts.FAMILY_KEYS`` keys, in that order, once each."""
    keys = [k.strip() for k in text.split(",") if k.strip()]
    unknown = [k for k in keys if k not in _parts.FAMILY_KEYS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown family keys {unknown}; keys: {', '.join(_parts.FAMILY_KEYS)}"
        )
    return tuple(k for k in _parts.FAMILY_KEYS if k in keys)


def add_ms7e_arguments(parser) -> None:
    """The MS7e switches (MS7e design section 10.1), shared by the MuJoCo heap scripts. Each
    defaults off, and off is MS7d."""
    parser.add_argument(
        "--families",
        type=families_arg,
        default=(),
        help="MS7e: the part families the catalog and the split add to brick and slope, a "
        f"comma list of {', '.join(_parts.FAMILY_KEYS)}; any family also turns on the eight "
        "classes",
    )
    parser.add_argument(
        "--box_draw",
        type=int,
        default=0,
        help="MS7e B1/B2: draw the box count per episode, uniform from 4 to this; 0 is off",
    )
    parser.add_argument(
        "--release_lean",
        action="store_true",
        help="MS7e B2: the release leg leans toward the box, the camera moves back and the "
        "band widens",
    )
    parser.add_argument(
        "--carry_via",
        action="store_true",
        help="MS7e R12 T1, with --release_lean: an EXTEND that would pass the arm base walks "
        "an arc about it",
    )
    parser.add_argument(
        "--box_unseen",
        choices=("size", "rim"),
        default=None,
        help="MS7e: the unseen-box ladder row's box draw (design section 8.3); off by default",
    )
    parser.add_argument(
        "--heap_family",
        choices=tuple(_parts.FAMILY_KEYS),
        default=None,
        help="MS7e family heap read: half of each heap from this family, half from the other "
        "parts of the split; the key must be in --families",
    )
    parser.add_argument(
        "--passenger_check",
        action="store_true",
        help="MS7e teacher lever T2 (ledger R12): a LIFT that raised a part other than the one "
        "it carries reopens in the tray and retries, so no passenger rides into another box",
    )
    parser.add_argument(
        "--wall_band_cont",
        action="store_true",
        help="MS7e serving lever S3 (ledger R13), with --gripper continuous: the grasp model's "
        "wall test puts the fingers at the continuous jaws' landing gap, not the binary hold band",
    )
    parser.add_argument(
        "--thin_snap",
        action="store_true",
        help="MS7e serving lever S2 (ledger R13), with --pick_snap: a served row whose half "
        "height is at most loop.THIN_SNAP_MAX_HALF_HEIGHT snaps to desk contact, not row z + 3 mm",
    )
    parser.add_argument(
        "--snap_closed",
        action="store_true",
        help="MS7e flat study (2026-09-30), with --pick_snap: pick_snap's landing hold ends once "
        "the measured jaw gap is under the served width plus loop.SNAP_CLOSED_MARGIN",
    )


def check_ms7e(parser, args) -> None:
    """Refuse an MS7e switch whose behaviour is not in this commit (MS7e design section 10.2)
    with ``parser.error``, so the script prints its fail token instead of a no-op run."""
    unbuilt = [k for k in args.families if k not in _parts.READY_FAMILIES]
    if unbuilt:
        parser.error(f"--families {','.join(unbuilt)} is not built on this commit")
    for name in UNBUILT_SWITCHES:
        if getattr(args, name):
            parser.error(f"--{name} is not built on this commit")
    if args.heap_family and args.heap_family not in args.families:
        parser.error("--heap_family must name one of --families")
    b2 = getattr(args, "release_lean", False)
    lo, hi = desk_boxes.BOX_COUNT_MIN, desk_boxes.B2_CEILING if b2 else desk_boxes.B1_CEILING
    if args.box_draw and not (lo <= args.box_draw <= hi and getattr(args, "desk_boxes", 0) > 0):
        parser.error(
            f"--box_draw draws {lo} to at most {hi} boxes ({'B2' if b2 else 'B1'}) and needs "
            "--desk_boxes"
        )
    if getattr(args, "carry_via", False) and not b2:
        parser.error("--carry_via bends the B2 carry: it needs --release_lean")
    if args.box_unseen and not args.box_draw:
        parser.error("--box_unseen draws the unseen boxes of --box_draw")
    if getattr(args, "deep_box_floor", False) and getattr(args, "desk_boxes", 0) <= 0:
        parser.error("--deep_box_floor deepens the desk box floors: it needs --desk_boxes")
    if getattr(args, "passenger_check", False) and getattr(args, "desk_boxes", 0) <= 0:
        parser.error("--passenger_check keeps passengers out of other boxes: it needs --desk_boxes")
    if getattr(args, "wall_band_cont", False) and getattr(args, "gripper", "") != "continuous":
        parser.error("--wall_band_cont is the continuous jaws' band: it needs --gripper continuous")
    if getattr(args, "thin_snap", False) and not getattr(args, "pick_snap", False):
        parser.error("--thin_snap moves pick_snap's floor: it needs --pick_snap")
    if getattr(args, "snap_closed", False) and not getattr(args, "pick_snap", False):
        parser.error("--snap_closed ends pick_snap's landing hold: it needs --pick_snap")


def ms7e_switches(args) -> dict:
    """The MS7e switch block a payload and a run's meta record, and the gate supervisor's cache
    matches (``run_gate_eval.ms7e_block``). Every switch off is MS7d."""
    return {
        "families": list(getattr(args, "families", ())),
        "box_draw": int(getattr(args, "box_draw", 0)),
        "release_lean": bool(getattr(args, "release_lean", False)),
        "carry_via": bool(getattr(args, "carry_via", False)),
        "box_unseen": getattr(args, "box_unseen", None),
        "heap_family": getattr(args, "heap_family", None),
        "passenger_check": bool(getattr(args, "passenger_check", False)),
        "wall_band_cont": bool(getattr(args, "wall_band_cont", False)),
        "thin_snap": bool(getattr(args, "thin_snap", False)),
        "snap_closed": bool(getattr(args, "snap_closed", False)),
    }


MS7F_OFF: dict = {
    "categories": False,
    "deep_tray": False,
    "tray_refill": False,
    "box_empty": False,
    "classifier": None,
    "passenger_drop": False,
    "transport_speed": trajectory.TRANSPORT_SPEED,
}
"""The MS7f switches (MS7f design section 12.1, branch 1) at their off values: the "ms7f" block
of a run with every one off, what an MS7e run ran with. categories is the base track's;
deep_tray, tray_refill and box_empty track H's (design section 5); classifier and
passenger_drop track C's (design section 6); transport_speed track S's (design section 8), the
teacher walk's pace. The gate supervisor keeps a stdlib copy (``scripts/run_gate_eval.py``),
pinned by a test."""
MS7F_READY: frozenset[str] = frozenset({"categories"})
"""The MS7f switches whose behaviour is in this commit. Any other at a value but its off value is
a usage error with the script's fail token (MS7f design section 12.2). Each track adds its
switch in the commit that brings the behaviour; the supervisor's copy is pinned by a test."""


def add_ms7f_arguments(parser, transport_speed: bool = True) -> None:
    """The MS7f switches (MS7f design section 12.1, ``MS7F_OFF``), shared by the MuJoCo heap
    scripts. Each defaults off, and off is MS7e. ``transport_speed`` False leaves out
    ``--transport_speed``, which the recorder already takes (``add_batch_arguments``)."""
    parser.add_argument(
        "--categories",
        action="store_true",
        help="MS7f: sort by the six categories (parts.PART_CATEGORIES) over the drawn boxes, a "
        "category with more boxes than it needs taking a second one (desk_boxes.category_map); "
        "needs --box_draw",
    )
    parser.add_argument(
        "--deep_tray",
        action="store_true",
        help="MS7f H1: the tray's walls rise to the deep tray's height and every part drops in "
        "at the start",
    )
    parser.add_argument(
        "--tray_refill",
        action="store_true",
        help="MS7f H1': the tray holds a share of the heap at the start and is refilled between "
        "cycles; the waiting parts are parked out of reach",
    )
    parser.add_argument(
        "--box_empty",
        action="store_true",
        help="MS7f H2: a full box is emptied between cycles, its parts graded then and parked",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=None,
        help="MS7f C: a TorchScript category classifier read on the three views, in place of the "
        "stand-in",
    )
    parser.add_argument(
        "--passenger_drop",
        action="store_true",
        help="MS7f C: open the jaws at the lift when the wrist view reads a passenger over the "
        "tray interior",
    )
    if transport_speed:
        parser.add_argument(
            "--transport_speed",
            type=float,
            default=trajectory.TRANSPORT_SPEED,
            help="MS7f S: the teacher walk's pace, the multiplier on the transport legs "
            "(trajectory.TRANSPORT_SPEED)",
        )


def check_ms7f(parser, args, ready: frozenset[str] = MS7F_READY) -> None:
    """Refuse an MS7f switch whose behaviour is not in this commit (MS7f design section 12.2)
    with ``parser.error``, so the script prints its fail token instead of a no-op run."""
    for name, off in MS7F_OFF.items():
        if getattr(args, name, off) != off and name not in ready:
            parser.error(f"--{name} is not built on this commit")
    if getattr(args, "categories", False) and not getattr(args, "box_draw", 0):
        parser.error("--categories maps the categories onto the drawn boxes: it needs --box_draw")


def ms7f_switches(args) -> dict:
    """The MS7f switch block a payload and a run's meta record, and the gate supervisor's cache
    matches (``run_gate_eval.ms7f_block``). Every switch off is MS7e."""
    return {name: getattr(args, name, off) for name, off in MS7F_OFF.items()}


def cycle_actor(orc) -> int:
    """The actor ordinal of the frame about to be recorded (design H7).

    The pick cycle ordinal, which is the length of the oracle's pick order once HOVER has
    selected. A push belongs to the pick that follows it, so it carries that pick's ordinal.
    SETTLE, DONE and ABANDON frames read 0 and are not recorded.
    """
    if orc is None or orc.phase in (oracle.Phase.SETTLE, oracle.Phase.DONE, oracle.Phase.ABANDON):
        return 0
    corrective = orc.phase in oracle.PUSH_PHASES or orc.phase in oracle.TIP_PHASES
    return len(orc.pick_order) + (1 if corrective else 0)


def negate_new_failures(recorder, env_index: int, outcomes: list, seen: int) -> int:
    """Negate the frames of every cycle since ``seen`` that did not stow; returns the new count."""
    while seen < len(outcomes):
        if not outcomes[seen]:
            recorder.negate_actor(env_index, seen + 1)
        seen += 1
    return seen


def build_run(args) -> tuple[dict, layout.BinLayout, oracle.OracleNoise]:
    """The run-level draw (RNG'd from --layout_seed) with --permutation applied, plus the
    derived BinLayout and OracleNoise, from the parsed generate_batch-style ``args``."""
    perm = tuple(int(v) for v in args.permutation.split(","))
    if sorted(perm) != [0, 1, 2, 3]:
        raise ValueError(
            f"--permutation must be a permutation of 0,1,2,3, got {args.permutation!r}"
        )
    if not 0.0 <= args.induce_prob <= 1.0:
        raise ValueError(f"--induce_prob must be in [0, 1], got {args.induce_prob}")
    if args.induce_offset_min > args.induce_offset_max:
        raise ValueError(
            f"--induce_offset_min ({args.induce_offset_min}) must be <= "
            f"--induce_offset_max ({args.induce_offset_max})"
        )
    if not 0.0 <= args.hover_jitter_min <= args.hover_jitter_max:
        raise ValueError(
            f"--hover_jitter_min/max must satisfy 0 <= min <= max, got "
            f"{args.hover_jitter_min}/{args.hover_jitter_max}"
        )
    rng = random.Random(args.layout_seed)
    draw = sample_run_draw(rng)
    draw["slot_colors"] = perm
    bl = layout.BinLayout(
        bin_x=draw["bin_x"], bin_ys=tuple(draw["bin_ys"]), slot_colors=draw["slot_colors"]
    )
    noise = oracle.OracleNoise(
        hover_height=(0.12 - 0.02 * args.noise_scale, 0.12 + 0.02 * args.noise_scale),
        lateral_offset=0.003 * args.noise_scale,
        yaw_offset=0.0873 * args.noise_scale,
        gain_scale=(1.0 - 0.3 * args.noise_scale, 1.0),
        action_noise_std=0.05 * args.noise_scale,
        via_jitter=0.02 * args.noise_scale,
        # Outside --noise_scale on purpose: it removes a SHORTCUT (the fixed R,B,Y,G,R,B
        # color schedule) rather than widening a noise band, so no recipe string should
        # read as if it might have carried it.
        randomize_pick_order=args.randomize_pick_order,
        nearest_first_pick_order=args.nearest_first_pick_order,
        hover_jitter=(args.hover_jitter_min, args.hover_jitter_max),
    )
    return draw, bl, noise


def make_induction(args) -> oracle.FailureInduction | None:
    if args.induce_prob <= 0.0:
        return None
    return oracle.FailureInduction(
        prob=args.induce_prob,
        offset_weight=args.induce_offset_weight,
        offset_range=(args.induce_offset_min, args.induce_offset_max),
        noise_boost=args.induce_noise_boost,
    )


def run_meta(
    args,
    draw: dict,
    noise: oracle.OracleNoise,
    induction: oracle.FailureInduction | None,
    *,
    num_bricks: int,
    export_mode: str,
    backend: str,
    task: str,
    parts: list | None = None,
    episode_length_s: float | None = None,
) -> dict:
    return {
        "task": task,
        "backend": backend,
        "run_seed": args.run_seed,
        "layout_seed": args.layout_seed,
        "num_envs": args.num_envs,
        "num_bricks": num_bricks,
        "attempts": args.attempts,
        "probe": args.probe,
        "noise_scale": args.noise_scale,
        "start_pose_offset_rad": args.start_pose_offset_rad,
        "transport_speed": args.transport_speed,
        # What was CONFIGURED; each episode's own "pick_order" records what was picked.
        "randomize_pick_order": noise.randomize_pick_order,
        "nearest_first_pick_order": noise.nearest_first_pick_order,
        # Configured range; each episode's "hover_jitters" says how many it demonstrated.
        "hover_jitter": list(noise.hover_jitter),
        "episode_draws": bool(getattr(args, "episode_draws", False)),
        "frame_jitter": bool(getattr(args, "frame_jitter", False)),
        # MS6 heap runs (design section 9). heap_k 0 is the six-brick task.
        "heap_k": int(getattr(args, "heap_k", 0)),
        "parts_seed": int(getattr(args, "parts_seed", 0)),
        "parts_split": getattr(args, "parts_split", "train"),
        "cycle_budget": int(getattr(args, "cycle_budget", 0)),
        "tray": bool(getattr(args, "tray", False)),
        "tray_lean": float(getattr(args, "tray_lean", 0.0)),
        "gripper": getattr(args, "gripper", "binary"),
        # MS7a design, desk boxes: 0 is the MS6 scene, the shelf's one heap bin.
        "desk_boxes": int(getattr(args, "desk_boxes", 0)),
        # MS7e design section 10.1: the MS7e switches, every one off on an MS7d run.
        "ms7e": ms7e_switches(args),
        "ms7f": ms7f_switches(args),  # MS7f design section 12.1, every one off on an MS7e run
        "deep_box_floor": bool(getattr(args, "deep_box_floor", False)),
        # MS7c design section 7.1: the rows, the marker and the box row as recorded from the
        # perceived state at these knob scales; each episode counts its perceived selections
        # and its fallbacks (a target part the perception missed).
        "perceived": bool(getattr(args, "perceived", False)),
        "noise_scales": perception.format_scales(args.noise_scales)
        if getattr(args, "perceived", False)
        else None,
        "parts": None if parts is None else [p.name for p in parts],
        "episode_length_s": (
            episode_length_s
            if episode_length_s is not None
            else (
                heap_episode_length_s(len(parts), int(getattr(args, "cycle_budget", 50)))
                if parts is not None
                else LEGACY_EPISODE_LENGTH_S
            )
        ),
        "induction": None
        if induction is None
        else {
            "prob": induction.prob,
            "offset_weight": induction.offset_weight,
            "offset_range": list(induction.offset_range),
            "noise_boost": induction.noise_boost,
        },
        "cameras": not args.no_cameras,
        "permutation": draw["slot_colors"],
        "shard_name": args.shard_name,
        "out_dir": args.out_dir,
        "max_wall_minutes": args.max_wall_minutes,
        "dataset_export_mode": export_mode,
        "run_draw": draw,
        # Carried in the sidecar itself, not only in a docstring: a consumer reading
        # <shard>.meta.json alone has no other way to learn that "sort_attempts" is not a
        # success count.
        "outcome_fields_note": (
            "episodes[].sort_attempts is an ATTEMPT count, not a success count -- the oracle "
            "bumps it when a RETREAT phase ends, whether or not the brick it was carrying "
            "arrived, so it counts a brick dropped in transit and can read either side of "
            "the truth. Use episodes[].succeeded (the env's own 'success' termination, the "
            "tensor the recorder's export filter grades on) as the label, "
            "episodes[].bricks_delivered for partial credit, and run_stats.successes (the "
            "recorder's own exported_successful_episode_count) for the run total."
        ),
        "reproducibility_note": (
            "Episodes after the first per env conclude via the env's auto-reset, which takes "
            "no seed of its own -- per-episode replay comes from each demo's own recorded "
            "'initial_state' in the HDF5, not from a seed. This run is identified by run_seed "
            "(OracleRig's noise-draw RNG) and layout_seed (sample_run_draw's RNG), plus "
            "num_envs -- per-step action noise is drawn from Torch's process-global RNG, so "
            "which noise sample lands on which control step depends on how many envs are "
            "stepped together, not on the two seeds alone."
        ),
    }


def run_batch(env, rig, target_attempts: int, max_wall_minutes: float) -> dict:
    """Drive ``rig`` until ``target_attempts`` episodes concluded or the deadline passed.

    Returns the raw counters ``run_stats`` turns into RUN_STATS. Closes the env on every
    exit path (the recorder's export count is read before that), as generate_batch.py did.
    """
    episodes: list[dict] = []
    abandon_reasons: Counter = Counter()
    control_steps = 0
    clamped_steps = 0
    attempts = 0
    cycle_bricks: list[set[int]] = [set() for _ in range(rig.num_envs)]
    episode_steps = [0] * rig.num_envs  # control steps of each env's episode so far
    motion = trajectory.MotionStats(rig.num_envs)
    start = time.time()
    heap = getattr(env, "parts", None) is not None
    recorder = getattr(env, "recorder", None)
    seen_outcomes = [0] * rig.num_envs
    if heap:
        env.oracle_actor = torch.zeros(rig.num_envs, dtype=torch.int16)
    deadline = start + max_wall_minutes * 60
    try:
        while attempts < target_attempts and time.time() < deadline:
            if heap:
                last_step = env.episode_length_buf >= env.max_episode_length - 1
                for i in range(rig.num_envs):
                    actor = cycle_actor(rig.oracles[i])
                    orc = rig.oracles[i]
                    if bool(last_step[i]) and recorder is not None and orc is not None:
                        # The time-out cuts this cycle short of its RETREAT (design H7). When
                        # a cycle is actually in flight `actor` IS its ordinal; when the
                        # oracle already went idle (DONE/ABANDON reads 0) a trailing push
                        # that hit its cap left the NEXT ordinal dangling with no pick ever
                        # following it (no cycle_outcomes entry, so negate_new_failures never
                        # sees it either). Negating unconditionally is safe: negate_actor only
                        # flips frames that still carry that exact ordinal, so it is a no-op
                        # once a cycle already resolved or never happened.
                        recorder.negate_actor(i, actor if actor > 0 else len(orc.pick_order) + 1)
                        actor = 0
                    env.oracle_actor[i] = actor
            out = rig.step()
            control_steps += 1
            episode_steps = [n + 1 for n in episode_steps]
            if heap and recorder is not None:
                for i in range(rig.num_envs):
                    orc = rig.oracles[i]
                    if orc is not None:
                        seen_outcomes[i] = negate_new_failures(
                            recorder, i, orc.cycle_outcomes, seen_outcomes[i]
                        )
            for i in range(rig.num_envs):
                motion.observe(i, rig.q_cmd[i])
            # Per-env flag: on how many envs the arm's command had to be clamped this step.
            clamped_steps += int(env.action_manager.get_term("arm_action").clamped.sum().item())
            for i, b in enumerate(env.oracle_cycle_brick.tolist()):
                if b >= 0:
                    cycle_bricks[i].add(b)
            finished = out["dones"] | out["abandoned"]
            if bool(finished.any()):
                env_ids = finished.nonzero(as_tuple=False).squeeze(-1).tolist()
                done_ids = set(out["dones"].nonzero(as_tuple=False).squeeze(-1).tolist())
                for i in env_ids:
                    # episode_stats is frozen by the rig on the step the episode ended, so this
                    # read is safe for every env in the batch (see OracleRig.episode_stats).
                    stats = rig.episode_stats(i)
                    if i in done_ids:
                        outcome = "success" if stats["succeeded"] else "time_out"
                    else:
                        outcome = "abandoned"
                        abandon_reasons[stats["abandon"] or "unknown"] += 1
                    episodes.append(
                        {
                            "env": i,
                            "outcome": outcome,
                            "cycles": len(cycle_bricks[i]),
                            "steps": episode_steps[i],
                            **stats,
                        }
                    )
                    cycle_bricks[i] = set()
                    episode_steps[i] = 0
                    seen_outcomes[i] = 0
                attempts += len(env_ids)
                abandoned_ids = out["abandoned"].nonzero(as_tuple=False).squeeze(-1).tolist()
                if abandoned_ids:
                    # Watchdog/oracle abandon: explicit env-scoped reset. The episode exports
                    # with success=False -- dropped in the default 'succeeded' mode.
                    env.reset(env_ids=torch.tensor(abandoned_ids, device=env.device))
                for i in env_ids:
                    motion.reset(i)
                rig.rearm(env_ids)
    finally:
        wall_seconds = time.time() - start
        exported = env.recorder_manager.exported_successful_episode_count
        # A heap episode's "success" needs every part of the heap stowed, so at the phase's
        # own 0.90-per-part target `exported` undercounts a heap run's usable output badly.
        # `exported_written` is every episode the recorder wrote, success or not, which is
        # what a heap caller (mj_generate_batch.py) should gate BATCH_OK on instead.
        exported_written = getattr(env.recorder_manager, "exported_episode_count", exported)
        env.close()
    return {
        "episodes": episodes,
        "abandon_reasons": abandon_reasons,
        "control_steps": control_steps,
        "clamped_steps": clamped_steps,
        "attempts": attempts,
        "wall_seconds": wall_seconds,
        "exported": exported,
        "exported_written": exported_written,
        "motion": motion.summary(),
    }


def run_stats(
    counters: dict,
    *,
    target_attempts: int,
    num_envs: int,
    probe: bool,
    num_bricks: int | None = None,
) -> dict:
    episodes = counters["episodes"]
    abandon_reasons = counters["abandon_reasons"]
    control_steps = counters["control_steps"]
    clamped_steps = counters["clamped_steps"]
    attempts = counters["attempts"]
    wall_seconds = counters["wall_seconds"]
    exported = counters["exported"]
    dones_count = sum(1 for e in episodes if e["outcome"] in ("success", "time_out"))
    abandoned_count = sum(1 for e in episodes if e["outcome"] == "abandoned")
    stats = {
        "attempts": attempts,
        "attempts_target": target_attempts,
        # Authoritative success count (the recorder's own running total), NOT a sum over
        # episodes[]["outcome"].
        "successes": exported,
        "time_outs": dones_count - exported,
        "abandons": abandoned_count,
        "abandon_reasons": dict(abandon_reasons),
        "watchdog_trips": abandon_reasons.get("watchdog", 0),
        "control_steps": control_steps,
        # The denominator is env control steps, not control steps, because the numerator
        # counts one per env.
        "clamped_steps": clamped_steps,
        "clamp_fraction": clamped_steps / (control_steps * num_envs) if control_steps else 0.0,
        "wall_seconds": wall_seconds,
        "sim_steps_per_sec": control_steps / wall_seconds if wall_seconds > 0 else 0.0,
        "deadline_hit": attempts < target_attempts,
        "episodes_per_env": {
            str(i): c for i, c in sorted(Counter(e["env"] for e in episodes).items())
        },
        # The oracle's planning-lookahead counters, summed over the run's episodes.
        "raise_vetoed": sum(int(e.get("raise_vetoed", 0)) for e in episodes),
        "hover_lowered": sum(int(e.get("hover_lowered", 0)) for e in episodes),
        # The per-run audit of --randomize_pick_order: with six bricks the lowest-index
        # sequence occurs by chance once in 720 episodes.
        "off_schedule_episodes": sum(
            1 for e in episodes if list(e.get("pick_order", [])) != sorted(e.get("pick_order", []))
        ),
        "hover_jitters": sum(int(e.get("hover_jitters", 0)) for e in episodes),
        # The smooth-oracle probe's reads (design section 5 item 1): control steps per
        # episode across the batch, and the commanded target's static-frame fraction and
        # acceleration quantiles.
        "env_steps_per_episode": control_steps * num_envs / attempts if attempts else None,
        "motion": counters.get("motion", {}),
    }
    if num_bricks is not None and any("cycles" in e for e in episodes):
        # MS6 heap runs (design section 8).
        stowed = sum(int(e.get("bricks_delivered", 0)) for e in episodes)
        stats["parts_total"] = num_bricks * len(episodes)
        stats["parts_stowed"] = stowed
        stats["parts_out_of_reach"] = sum(int(e.get("parts_out_of_reach", 0)) for e in episodes)
        stats["cycles"] = sum(int(e.get("cycles", 0)) for e in episodes)
        stats["pushes"] = sum(int(e.get("pushes", 0)) for e in episodes)
        stats["rakes"] = sum(int(e.get("rakes", 0)) for e in episodes)
        stats["cycles_per_stowed"] = stats["cycles"] / stowed if stowed else None
        # MS7f design section 4.1: every step of every episode counts, the events take none.
        steps = sum(int(e.get("steps", 0)) for e in episodes)
        stats["steps_total"] = steps
        stats["stows_per_hour"] = _loop.stows_per_hour(stowed, steps)
        stats["events"] = {n: sum(e.get("events", {}).get(n, 0) for e in episodes) for n in EVENTS}
    if probe:
        # A heap probe grades on every episode the recorder wrote, not the fully-stowed
        # count: the same "exported_written" gate `exported_count` (mj_generate_batch.py)
        # uses, because a heap episode's own success bar is much higher than one probe run.
        probe_successes = counters["exported_written"] if num_bricks is not None else exported
        success_rate = probe_successes / attempts if attempts else 0.0
        wall_sec_per_episode = wall_seconds / attempts if attempts else 0.0
        stats["success_rate"] = success_rate
        stats["wall_sec_per_episode"] = wall_sec_per_episode
        stats[f"extrapolated_{CAMPAIGN_TARGET_SUCCESSES}_success_wall_hours"] = (
            (CAMPAIGN_TARGET_SUCCESSES / success_rate) * wall_sec_per_episode / 3600.0
            if success_rate > 0
            else None
        )
    return stats


def write_meta(
    out_dir: str | Path, shard_name: str, meta: dict, stats: dict, episodes: list
) -> Path:
    path = Path(out_dir) / f"{shard_name}.meta.json"
    path.write_text(
        json.dumps({"run_meta": meta, "run_stats": stats, "episodes": episodes}, indent=2)
    )
    return path


MAX_HEAP_K = 150
"""The largest heap a script takes: the MS7f heap of 150 parts (MS7f design section 5). A part
index above 127 does not fit the int8 of obs/cycle_brick and the rig's stash, so a heap over
127 parts writes both as int16 (``mj.recorder.MAX_INT8_PART``) and every smaller heap keeps
the int8 schema of every shard before it. No cycle budget, drop or tray is sized past 150."""


def _heap_k_type(parser):
    """A ``type=`` for ``--heap_k`` that rejects a heap over ``MAX_HEAP_K``."""

    def parse(value: str) -> int:
        k = int(value)
        if k > MAX_HEAP_K:
            parser.error(f"--heap_k {k} exceeds {MAX_HEAP_K}, the MS7f heap (MAX_HEAP_K)")
        return k

    return parse


def add_batch_arguments(parser) -> None:
    """The generate_batch.py flags both backends share (AppLauncher flags excluded)."""
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--shard_name", type=str, default="shard")
    parser.add_argument("--attempts", type=int, default=50)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--run_seed", type=int, default=0, help="Seeds OracleRig's noise-draw RNG.")
    parser.add_argument(
        "--layout_seed", type=int, default=0, help="Seeds sample_run_draw's run-level RNG."
    )
    parser.add_argument(
        "--permutation",
        type=str,
        default="0,1,2,3",
        help="Comma-separated color-slot permutation, overrides sample_run_draw's own shuffle "
        "-- the campaign supervisor schedules permutations without replacement.",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=1.0,
        help="Multiplier on every OracleNoise axis (0 = OracleNoise's own nominal minus the "
        "hover-height band collapsed to its center).",
    )
    parser.add_argument(
        "--randomize_pick_order",
        action="store_true",
        help="Pick the next brick uniformly at random among the ones still on the table instead "
        "of taking the lowest index (openyam-lab's largest gate lever). Off by default.",
    )
    parser.add_argument(
        "--nearest_first_pick_order",
        action="store_true",
        help="Pick the brick nearest the tool instead of the lowest index, so the target is "
        "readable from the cameras; refused together with --randomize_pick_order. Off by "
        "default.",
    )
    parser.add_argument(
        "--hover_jitter_min",
        type=float,
        default=0.0,
        help="Corrective approaches: lower radius (m) of the per-pick lateral HOVER offset.",
    )
    parser.add_argument(
        "--hover_jitter_max",
        type=float,
        default=0.0,
        help="Upper radius (m) of the per-pick lateral HOVER offset; 0 = off (the default).",
    )
    parser.add_argument(
        "--episode_draws",
        action="store_true",
        help="MuJoCo only: redraw bins, lights and cameras at every episode reset instead "
        "of once per run (MS4 lever 1). The color permutation stays per run.",
    )
    parser.add_argument(
        "--frame_jitter",
        action="store_true",
        help="MuJoCo only: resample a small overhead-camera jitter on every rendered frame.",
    )
    parser.add_argument(
        "--heap_k",
        type=_heap_k_type(parser),
        default=0,
        help="MuJoCo only: replace the six bricks with a heap of K mixed parts on the desk (MS6). "
        f"0 is the six-brick task; K must be <= {MAX_HEAP_K}, the MS7f heap.",
    )
    parser.add_argument(
        "--parts_seed", type=int, default=20260915, help="seeds the held-out split of the catalog"
    )
    parser.add_argument(
        "--parts_split",
        choices=PARTS_SPLITS,
        default="train",
        help="which side of the split a heap run draws its parts from",
    )
    parser.add_argument(
        "--cycle_budget",
        type=int,
        default=50,
        help="pick cycles a heap episode may spend (the MuJoCo recorder's default scales with "
        "the heap, batch_loop.heap_cycle_budget)",
    )
    parser.add_argument(
        "--tray",
        action="store_true",
        help="MuJoCo heap only: drop the heap into the walled tray (MS7a); off is the bare desk",
    )
    parser.add_argument(
        "--tray_lean",
        type=float,
        default=0.0,
        help="MS7a: lean the tray's walls outward by this many degrees (0 is vertical)",
    )
    parser.add_argument(
        "--gripper",
        choices=("binary", "continuous"),
        default="binary",
        help="MuJoCo only: the gripper contract of the recorded action's seventh column "
        "(MS6 phase B design 5.4): binary is raw >= 0 opens, continuous is the jaw gap in m",
    )
    parser.add_argument(
        "--desk_boxes",
        type=int,
        default=0,
        help="MuJoCo heap only: N desk boxes beside the arm replace the shelf and its heap bin "
        "(MS7a: 1 for rung 1, 4 for rung 2); the heap drops on the tray footprint. 0 is the MS6 "
        "scene.",
    )
    parser.add_argument(
        "--deep_box_floor",
        action="store_true",
        help="MS7e flat track: each desk box floor's collider reaches under the desk top, so "
        "a thin part released over the box cannot drop through it (core.desk_boxes)",
    )
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=None,
        help="override the episode length; the default is 180 s, or 30 + 20 * min(K, budget) on a heap",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Print extra throughput/extrapolation fields in RUN_STATS. Does not change "
        "--attempts.",
    )
    parser.add_argument(
        "--start_pose_offset_rad",
        type=float,
        default=None,
        help="half-width of the uniform per-joint offset applied to the arm's start pose at "
        "reset (the env's own default is 0.05).",
    )
    parser.add_argument(
        "--transport_speed",
        type=float,
        default=trajectory.TRANSPORT_SPEED,
        help="Multiplier on the pace of the transport legs (HOVER, LIFT, RAISE, EXTEND, "
        "RETREAT) under the minimum-jerk walk; 1.0 is the old staircase's pace, the default "
        "is the speed-ladder read. DESCEND is never scaled.",
    )
    parser.add_argument(
        "--induce_prob",
        type=float,
        default=0.0,
        help="Per-pick probability of a deliberate first-attempt grasp failure (0 = off).",
    )
    parser.add_argument("--induce_offset_weight", type=float, default=0.7)
    parser.add_argument("--induce_offset_min", type=float, default=0.015)
    parser.add_argument("--induce_offset_max", type=float, default=0.025)
    parser.add_argument("--induce_noise_boost", type=float, default=5.0)
    parser.add_argument(
        "--export_mode",
        type=str,
        choices=("succeeded", "all"),
        default="succeeded",
        help="'succeeded' exports only episodes whose 'success' termination fired; 'all' "
        "exports every concluded episode (instrument-validation shards only).",
    )
    parser.add_argument("--no_cameras", action="store_true", help="cameras off in the env cfg.")
    parser.add_argument("--max_wall_minutes", type=float, default=90.0)
