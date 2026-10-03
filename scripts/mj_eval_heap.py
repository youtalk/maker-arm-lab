"""Heap gate client on the MuJoCo backend: one process = one run-level draw (phase B design 5.3).

The heap twin of ``scripts/mj_eval_policy.py``: the same policy server over ZMQ and the same
``EVAL_OK`` / ``EVAL_FAIL`` token contract, with ``maker_arm_lab.core.loop`` as the rollout,
so ``scripts/run_gate_eval.py --backend mujoco --heap_k K`` drives it. Runs under the
project's own venv:

    uv run python scripts/mj_eval_heap.py --num_envs 5 --heap_k 50 --parts_split heldout \\
        --run_seed 20261400 --layout_seed 20261410 --permutation 0,1,2,3 --episode_draws \\
        --frame_jitter --context camera_box --server tcp://127.0.0.1:5555 --out /tmp/eval_000.json

``--oracle_driver`` replaces the policy with the oracle rig on every cycle: the preflight's
driver and the loop's end-to-end check, no server and no cameras needed.

``--dagger_dir`` is MS7f wave 3 (ledger D17): the same served loop, and on a failure trigger
(``loop.TakeoverConfig`` without the random draw, ``loop.HELD_NO_RISE_STEPS``) the teacher takes
the cycle over. Only the takeover frames are recorded, to ``<dagger_dir>/<shard_name>.hdf5``, for
``convert_to_lerobot.py --split_cycles --dagger``; the payload names the shard.

``--classifier_frames DIR`` records every frame the classifier reads on the policy's own cycles
as a render set file (``mj_render_set.PolicyFrames``), labels read from the simulator; the run
itself is the same.
"""

import argparse
import gzip
import json
import os
import random
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lerobot_stack"))
import policy_client
from maker_arm_lab.core import oracle, perception
from maker_arm_lab.core.batch_loop import (
    MAX_HEAP_K,
    PARTS_SPLITS,
    add_ms7e_arguments,
    add_ms7f_arguments,
    check_ms7e,
    check_ms7f,
    check_teacher_waits,
    heap_classes,
    heap_parts,
    heldout_names,
    ms7e_switches,
    ms7f_switches,
    teacher_waits,
)
from maker_arm_lab.core.eval_loop import EvalFailure
from maker_arm_lab.core.heap_oracle import CELL_FAILURE_CAP, HeapOracle
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.core.loop import (
    CARRY_VIEW_STEPS,
    CYCLE_STEP_BUDGET,
    HELD_NO_RISE_STEPS,
    MAX_CARRY_VOTES,
    MAX_CELL_FAILURE_CAP,
    TakeoverConfig,
    cycle_budget,
    episode_length_s,
    rollout,
)
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig


class _TokenEmittingParser(argparse.ArgumentParser):
    """A parser whose usage errors still honour the EVAL_OK / EVAL_FAIL contract (``token``: a
    script that runs this client's ``main``, ``scripts/mj_stretch_run.py``, sets its own, and its
    own ``max_heap_k``: a stretch pool is served through the tray refill, not dropped at once)."""

    token = "EVAL"
    max_heap_k = MAX_HEAP_K

    def error(self, message: str):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr, flush=True)
        print(
            f"{self.token}_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True
        )
        raise SystemExit(2)


def _scales(text: str) -> dict:
    """``--noise_scales``: ``perception.parse_scales`` with its reason in the usage error."""
    try:
        return perception.parse_scales(text)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from None


def parse_args(argv=None, parser_cls=_TokenEmittingParser) -> argparse.Namespace:
    parser = parser_cls(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument("--run_seed", type=int, default=0, help="env.reset seed and the rig rng")
    parser.add_argument("--layout_seed", type=int, default=0, help="seeds sample_run_draw")
    parser.add_argument("--permutation", type=str, default="0,1,2,3", help="bin-color permutation")
    parser.add_argument("--episode_draws", action="store_true", help="per-episode scene draws")
    parser.add_argument("--frame_jitter", action="store_true", help="per-frame camera jitter")
    parser.add_argument("--spatial_scale", type=float, default=1.0)
    parser.add_argument(
        "--context",
        choices=("none", "camera", "camera_box", "camera_box_target"),
        default="camera_box",
    )
    parser.add_argument("--heap_k", type=int, default=50, help="parts in the heap (> 0)")
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
    parser.add_argument("--parts_seed", type=int, default=20260915)
    parser.add_argument("--parts_split", choices=PARTS_SPLITS, default="heldout")
    parser.add_argument(
        "--cycle_budget",
        type=int,
        default=None,
        help="pick cycles per episode; the default is ceil(1.3 * K) (phase B design 5.1)",
    )
    parser.add_argument(
        "--gripper",
        choices=("binary", "continuous"),
        default="binary",
        help="the env's gripper contract; must match the checkpoint's handshake",
    )
    parser.add_argument(
        "--desk_boxes",
        type=int,
        default=0,
        help="N desk boxes beside the arm replace the shelf and its heap bin (MS7a: 1 for "
        "rung 1, 4 for rung 2); 0 is the MS6 scene",
    )
    add_ms7e_arguments(parser)
    add_ms7f_arguments(parser)
    parser.add_argument(
        "--deep_box_floor",
        action="store_true",
        help="MS7e flat track: each desk box floor's collider reaches under the desk top",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="MS7f H0: render the policy cameras under --oracle_driver too, to time a read's scene",
    )
    parser.add_argument("--server", type=str, default="tcp://127.0.0.1:5555")
    parser.add_argument("--out", type=str, default="", help="write the EVAL_OK payload here too")
    parser.add_argument(
        "--oracle_driver", action="store_true", help="the rig drives every cycle; no server"
    )
    parser.add_argument(
        "--dagger_dir",
        type=str,
        default="",
        help="MS7f wave 3: the teacher takes a failing cycle over (the failure triggers) and "
        "only its frames are recorded to <dagger_dir>/<shard_name>.hdf5; needs the server",
    )
    parser.add_argument("--shard_name", type=str, default="shard", help="with --dagger_dir")
    parser.add_argument(
        "--max_wall_minutes",
        type=float,
        default=None,
        help="the wall-clock cap; the default allows a generous 0.15 s per step of the episode "
        "(every cycle of the budget at its full step budget)",
    )
    parser.add_argument(
        "--select_at_release",
        action="store_true",
        help="select the next target when the release opens the jaws after a stow "
        "(loop.release_ready), not at the retreat point",
    )
    parser.add_argument(
        "--replan_on_switch",
        action="store_true",
        help="send the per-env replan flag: the server refills that env's action queue on "
        "the step a target row is written",
    )
    parser.add_argument(
        "--row_refresh",
        action="store_true",
        help="rewrite the target row from the live scene at a re-open that lifted nothing "
        "(the teacher's REOPEN refresh)",
    )
    parser.add_argument(
        "--squeeze_floor",
        action="store_true",
        help="cap a closing gap command at the teacher's close set point, w - 6 mm on the "
        "served target row's width (loop.floored_gap)",
    )
    parser.add_argument(
        "--close_gate",
        action="store_true",
        help="hold the jaws at the teacher's landing gap while the pick point is still "
        "descending onto the served target (loop.gated_gap); applied before --squeeze_floor",
    )
    parser.add_argument(
        "--square_wrist",
        action="store_true",
        help="turn the wrist roll command toward the served row's closing axis on the final "
        "approach, before the first hold (loop.squared_roll)",
    )
    parser.add_argument(
        "--scripted_place",
        action="store_true",
        help="MS7a fallback: at the cycle's first hold plus a 5 cm lift the teacher resumes at "
        "LIFT and carries the part to its own box (loop.SCRIPTED_PLACE_RISE); needs the server",
    )
    parser.add_argument(
        "--scripted_rake",
        action="store_true",
        help="MS7a tray: when the loop finds no target but the teacher would rake a part off a "
        "wall, the rig runs that rake and hands the env back to the policy; needs the server",
    )
    parser.add_argument(
        "--part_failures",
        action="store_true",
        help="MS7a: book a failed cycle against its part, not its 5 cm cell, so the loop skips "
        "a part after three failed cycles and its neighbours stay candidates",
    )
    parser.add_argument(
        "--pick_snap",
        action="store_true",
        help="MS7a: on the final approach move the commanded pick point onto the served row, xy "
        "toward its centre and, with the jaws open over it, down to the teacher's grasp height "
        "(loop.snap_shift)",
    )
    parser.add_argument(
        "--dwell_close",
        action="store_true",
        help="MS7a: command the teacher's close when the policy waits at grasp height with the "
        "jaws open for loop.DWELL_STEPS (loop.dwell_count)",
    )
    parser.add_argument(
        "--adopt_held",
        action="store_true",
        help="MS7a desk boxes: serve the box and marker of a lifted part that is not the target, "
        "as the teacher's adopt_held_part does",
    )
    parser.add_argument(
        "--rake_cap",
        type=int,
        default=0,
        help="MS7a rake prototype: rake one part up to N times (the teacher's cap is "
        "heap_oracle.RAKE_PART_CAP) and scale the episode's rake cap with it; 0 keeps both",
    )
    parser.add_argument(
        "--rake_crowd",
        action="store_true",
        help="MS7a rake prototype: when no rake start clears every neighbour, land the tips up "
        "to grasp.RAKE_CROWD_M into one",
    )
    parser.add_argument(
        "--park_done",
        action="store_true",
        help="MS7a tail forensics: hold an env whose episode has ended at its last command "
        "while the rest of the batch runs, instead of serving it the policy",
    )
    parser.add_argument(
        "--rake_turn",
        action="store_true",
        help="MS7a tail forensics: retry a rake the rig declined at the teacher's next jaw "
        "headings (across the sweep, then held) before it counts as spent",
    )
    parser.add_argument(
        "--perceived",
        action="store_true",
        help="MS7c: the loop's decisions read only the perceived state (core.perception) and the "
        "hold detector, never the simulator truth; needs --desk_boxes, --gripper continuous and "
        "the server",
    )
    parser.add_argument(
        "--noise_scales",
        type=_scales,
        default=None,
        help="MS7c, with --perceived: a scale per perception knob, e.g. 'heading=2' or "
        "'all=0,miss=1' (knobs: all, "
        + ", ".join(perception.KNOBS + perception.EXTRA_KNOBS)
        + "); a knob not named "
        "stays at 1, the nominal, and so does every knob when this is omitted",
    )
    parser.add_argument(
        "--contact_hold",
        action="store_true",
        help="MS7c, with --perceived: the pre-registered fallback, the serving rules' stop flag "
        "reads the contacts again; everything else stays perceived",
    )
    parser.add_argument(
        "--box_filter_xy",
        action="store_true",
        help="MS7d L1, with --perceived: the candidate filter tests only a detection's x and y "
        "against each perceived box's outer footprint plus desk_boxes.BOX_FILTER_MARGIN_M",
    )
    parser.add_argument(
        "--release_gap",
        action="store_true",
        help="MS7d L2, with --perceived: a release also completes at the jaw gap of the hold's "
        "end plus loop.RELEASE_OPEN_M",
    )
    parser.add_argument(
        "--rearm_hold",
        action="store_true",
        help="MS7d L8, with --perceived: the serving levers' stop flag clears after a hold that "
        "ends with no lift",
    )
    parser.add_argument(
        "--box_walls",
        action="store_true",
        help="MS7d L9, with --perceived: the grasp model and the rake read the walls of the "
        "perceived desk boxes",
    )
    parser.add_argument(
        "--marker_bias",
        action="store_true",
        help="MS7d L3, with --perceived: every detection is corrected by a bias estimate from the "
        "gripper marker in the overhead image",
    )
    parser.add_argument(
        "--wrist_fix",
        action="store_true",
        help="MS7d L4, with --perceived: a wrist look at the final approach corrects the serving "
        "row",
    )
    parser.add_argument(
        "--class_views",
        action="store_true",
        help="MS7d L5, with --perceived: a correlated class error model, and a third view on a "
        "disagreement of two",
    )
    parser.add_argument(
        "--hold_force",
        action="store_true",
        help="MS7d L6: a hold enters only while the pick-point force is under the bar",
    )
    parser.add_argument(
        "--force_cap",
        action="store_true",
        help="MS7d L7: the arm holds still on a step with a large pick-point force",
    )
    parser.add_argument(
        "--carry_guard",
        action="store_true",
        help="MS7g, with --perceived: keep a lifted part in the jaws until the pick point is over "
        "its served box, and hand a still guard to the rig's perceived carry (loop.guarded_gap); "
        "refuses --passenger_drop",
    )
    parser.add_argument(
        "--clear_first",
        action="store_true",
        help="MS7g, with --perceived: the selector ranks the grasps that sweep no standing "
        "neighbour before the height band, so it may choose another part (grasp.rank_grasps)",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="",
        help="dump env 0's wrist and overhead policy frames as PNGs here, every "
        "loop.FRAME_EVERY steps of each cycle (forensics; needs the policy server)",
    )
    parser.add_argument(
        "--classifier_frames",
        type=str,
        default="",
        help="MS7f: write every frame the classifier reads, labelled from the simulator, to "
        "DIR/policy_<run_seed>_<layout_seed>.npz in scripts/mj_render_set.py's layout, for "
        "lerobot_stack/train_part_classifier.py; needs --classifier and --perceived",
    )
    parser.add_argument(
        "--hold_trace",
        type=str,
        default="",
        help="MS7c: write the per-step hold trace here as gzipped JSON lines, one list per live "
        "env per step in core.hold.TRACE_FIELDS order (env, step, cycle, measured gap, "
        "commanded gap, pick z, contact-held part, its rise, and under --hold_force or "
        "--force_cap the pick-point force), for scripts/hold_agreement.py --traces; off by default",
    )
    args = parser.parse_args(argv)
    for name in ("run_seed", "layout_seed"):
        if not (0 <= getattr(args, name) < 2**32):
            parser.error(f"--{name} is outside numpy's seed range (0 <= seed < 2**32)")
    if args.heap_k <= 0:
        parser.error("--heap_k must be positive: this client evaluates heap episodes only")
    if args.heap_k > parser.max_heap_k:
        parser.error(f"--heap_k {args.heap_k} exceeds {parser.max_heap_k}, this client's heap cap")
    if not args.transport_speed > 0.0:
        parser.error("--transport_speed must be positive")
    if args.lift_budget < 0:
        parser.error("--lift_budget must be 0 (off) or more")
    if args.end_at_done:
        parser.error("--end_at_done cuts the recorder's idle tail: the heap client has none")
    if args.abandon_cycle:
        parser.error("--abandon_cycle is the recorder's: the heap loop survives a rig abandon")
    check_ms7e(parser, args)
    check_ms7f(parser, args)
    check_teacher_waits(parser, args)
    if args.carry_guard and args.passenger_drop:
        parser.error("--carry_guard refuses --passenger_drop, which opens over the tray")
    if args.classifier and not (args.perceived and args.categories):
        parser.error(
            "--classifier reads the categories on the perceived views: it needs --perceived and "
            "--categories"
        )
    if args.classifier_frames and not (args.classifier and args.perceived):
        parser.error(
            "--classifier_frames records the classifier's reads: it needs --classifier and "
            "--perceived"
        )
    if args.passenger_drop and not args.classifier:
        parser.error(
            "--passenger_drop reads the classifier's passenger head: it needs --classifier"
        )
    if args.lift_view_steps and not args.classifier:
        parser.error("--lift_view_steps delays the classifier's wrist read: it needs --classifier")
    if not 0 <= args.lift_view_steps < CARRY_VIEW_STEPS:
        parser.error(f"--lift_view_steps must lie in [0, {CARRY_VIEW_STEPS})")
    if args.missed_lift and not args.tray:
        parser.error("--missed_lift reads a carry off the tray: it needs --tray")
    if not 1 <= args.cell_failure_cap <= MAX_CELL_FAILURE_CAP:
        parser.error(f"--cell_failure_cap must lie in [1, {MAX_CELL_FAILURE_CAP}]")
    if args.cell_failure_cap != CELL_FAILURE_CAP and not args.perceived:
        parser.error("--cell_failure_cap sets the perceived selector's cap: it needs --perceived")
    if args.carry_votes and not (args.classifier and args.class_views):
        parser.error(
            "--carry_votes adds reads to the classifier's vote: it needs --classifier and "
            "--class_views"
        )
    if not 0 <= args.carry_votes <= MAX_CARRY_VOTES:
        parser.error(f"--carry_votes must lie in [0, {MAX_CARRY_VOTES}]")
    if args.carry_votes >= CARRY_VIEW_STEPS - args.lift_view_steps:
        parser.error(
            f"--carry_votes must be under {CARRY_VIEW_STEPS} - --lift_view_steps, a step for "
            "each read before the carry view"
        )
    if args.desk_boxes < 0:
        parser.error("--desk_boxes must be 0 or more")
    if args.dagger_dir and args.oracle_driver:
        parser.error("--dagger_dir takes cycles over from a policy: it needs the server")
    if args.perceived:
        if not args.desk_boxes:
            parser.error("--perceived needs --desk_boxes: a perceived box ends each cycle")
        if args.oracle_driver:
            parser.error("--perceived serves a policy; the oracle driver reads the truth")
        if args.gripper != "continuous":
            parser.error("--perceived needs --gripper continuous: the hold reads the jaw gap")
    for name in (
        "noise_scales",
        "contact_hold",
        "box_filter_xy",
        "release_gap",
        "rearm_hold",
        "box_walls",
        "marker_bias",
        "wrist_fix",
        "class_views",
        "lift_budget",
        "missed_lift",
        "stall_end",
        "carry_guard",
        "clear_first",
    ):
        if getattr(args, name) not in (None, False) and not args.perceived:
            parser.error(f"--{name} needs --perceived")
    if args.cycle_budget is None:
        args.cycle_budget = cycle_budget(args.heap_k)
    if args.max_wall_minutes is None:
        args.max_wall_minutes = args.cycle_budget * CYCLE_STEP_BUDGET * 0.15 / 60.0
    return args


def frame_writer(frames_dir: str):
    """``loop.rollout``'s ``on_frame``: ``c<ordinal>_s<step>_<camera>.png`` under ``frames_dir``."""
    from PIL import Image

    root = Path(frames_dir)
    root.mkdir(parents=True, exist_ok=True)

    def write(ordinal: int, step: int, wrist, overhead) -> None:
        for name, frame in (("wrist", wrist), ("overhead", overhead)):
            Image.fromarray(frame).save(root / f"c{ordinal:03d}_s{step:04d}_{name}.png")

    return write


def main(argv=None, parser_cls=_TokenEmittingParser, run=None, episode_s=None) -> int:
    """``run`` is the rollout (``loop.rollout``, or the stretch runner's chain of them) and
    ``episode_s`` overrides the env's episode length; the tokens are ``parser_cls.token``'s."""
    args = parse_args(argv, parser_cls)
    token = parser_cls.token
    policy = None
    env = None
    recorder = None
    close_error = None
    try:
        perm = tuple(int(v) for v in args.permutation.split(","))
        if sorted(perm) != [0, 1, 2, 3]:
            raise EvalFailure("permutation", args.permutation)
        draw = sample_run_draw(random.Random(args.layout_seed), args.spatial_scale)
        draw["slot_colors"] = perm
        parts = heap_parts(args)
        env = MjLegoSortEnv(
            args.num_envs,
            draw,
            num_bricks=len(parts),
            parts=parts,
            cameras=not args.oracle_driver or args.render,
            episode_draws=args.episode_draws,
            frame_jitter=args.frame_jitter,
            spatial_scale=args.spatial_scale,
            context=args.context,
            episode_length_s=episode_s or episode_length_s(args.cycle_budget),
            gripper_mode=args.gripper,
            desk_boxes=args.desk_boxes,
            tray=args.tray,
            tray_lean_deg=args.tray_lean,
            class_names=heap_classes(args),
            box_draw=args.box_draw,
            box_unseen=args.box_unseen,
            deep_box_floor=args.deep_box_floor,
            release_lean=args.release_lean,
            round_physics="round" in args.families,  # MS7e design section 6.4
            categories=args.categories,
            deep_tray=args.deep_tray,
            tray_refill=args.tray_refill,
            pile_pgs=args.pile_pgs,
            export_dir=args.dagger_dir or None,
            shard_name=args.shard_name,
            export_all=bool(args.dagger_dir),
        )
        env.reset(seed=args.run_seed)
        rig_rng = random.Random(args.run_seed)
        # MS7f --categories: the teacher's record of what it released per box, one per env and
        # shared by the oracle of every cycle, since the loop runs one episode per env.
        loads = [[0.0] * env.desk_boxes for _ in range(env.num_envs)]

        def factory(i, bl, episode_draw):
            orc = HeapOracle(
                bl,
                env.parts,
                env.destination_bins[i],
                episode_draw,
                rng=rig_rng,
                cycle_budget=args.cycle_budget,
                boxes=env.desk_box_poses[i],
                box_sizes=env.desk_box_sizes[i],
                tray=env.tray,
                tray_lean_deg=env.tray_lean_deg,
                wall_band_cont=args.wall_band_cont,
                box_sets=env.part_box_sets(i),
                box_load=loads[i],
                tray_wall_height=env.tray_wall_height,
            )
            return teacher_waits(orc, args)

        perceivers = None  # wave 3: the takeover rows are the perceived ones, as the corpora's
        if args.dagger_dir and args.perceived:
            perceivers = [
                perception.Perceiver(args.run_seed, i, args.noise_scales, env.class_names)
                for i in range(args.num_envs)
            ]
        rig = OracleRig(
            env,
            env.cfg.bin_layout,
            oracle.OracleNoise(),
            rig_rng,
            transport_speed=args.transport_speed,
            oracle_factory=factory,
            perceivers=perceivers,
            release_lean=args.release_lean,
            carry_via=args.carry_via,
            passenger_check=args.passenger_check,
            descend_speed=args.descend_speed,
            quiet_landing=args.quiet_landing,
            landing_steps=args.landing_steps,
            net_stall=args.net_stall,
            near_tracking=args.near_tracking,
        )
        policy = None if args.oracle_driver else policy_client.ServerPolicy(args.server)
        trace = [] if args.hold_trace else None
        classifier = perception.PartClassifier(args.classifier) if args.classifier else None
        if args.classifier_frames:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from mj_render_set import PolicyFrames

            out_dir, crop = args.classifier_frames, classifier.wrist_crop
            recorder = PolicyFrames(env, rig, out_dir, args, crop)
        payload = (run or rollout)(
            env,
            rig,
            policy,
            run_seed=args.run_seed,
            layout_seed=args.layout_seed,
            permutation=perm,
            cycle_budget=args.cycle_budget,
            max_wall_minutes=args.max_wall_minutes,
            takeover=(
                TakeoverConfig(random_interval=None, held_no_rise_steps=HELD_NO_RISE_STEPS)
                if args.dagger_dir
                else None
            ),
            on_frame=frame_writer(args.frames_dir) if args.frames_dir else None,
            on_classify=recorder,
            select_at_release=args.select_at_release,
            replan_on_switch=args.replan_on_switch,
            row_refresh=args.row_refresh,
            squeeze_floor=args.squeeze_floor,
            close_gate=args.close_gate,
            square_wrist=args.square_wrist,
            scripted_place=args.scripted_place,
            scripted_rake=args.scripted_rake,
            part_failures=args.part_failures,
            pick_snap=args.pick_snap,
            dwell_close=args.dwell_close,
            adopt_held=args.adopt_held,
            rake_cap=args.rake_cap,
            rake_crowd=args.rake_crowd,
            park_done=args.park_done,
            rake_turn=args.rake_turn,
            hold_trace=trace,
            perceived=args.perceived,
            noise_scales=args.noise_scales,
            contact_hold=args.contact_hold,
            box_filter_xy=args.box_filter_xy,
            release_gap=args.release_gap,
            rearm_hold=args.rearm_hold,
            box_walls=args.box_walls,
            marker_bias=args.marker_bias,
            wrist_fix=args.wrist_fix,
            class_views=args.class_views,
            hold_force=args.hold_force,
            force_cap=args.force_cap,
            thin_snap=args.thin_snap,
            snap_closed=args.snap_closed,
            heldout=heldout_names(args),
            box_empty=args.box_empty,
            book_miss=args.book_miss,
            lift_budget=args.lift_budget,
            classifier=classifier,
            passenger_drop=args.passenger_drop,
            lift_view_steps=args.lift_view_steps,
            missed_lift=args.missed_lift,
            cell_failure_cap=args.cell_failure_cap,
            carry_votes=args.carry_votes,
            stall_end=args.stall_end,
            carry_guard=args.carry_guard,
            clear_first=args.clear_first,
        )
        payload["parts_split"] = args.parts_split
        payload["parts_seed"] = args.parts_seed
        payload["desk_boxes"] = args.desk_boxes
        payload["tray"] = args.tray
        payload["tray_lean"] = args.tray_lean
        payload["ms7e"] = ms7e_switches(args)
        payload["ms7f"] = ms7f_switches(args)
        payload["deep_box_floor"] = args.deep_box_floor
        if args.dagger_dir:
            payload["shard"] = str(Path(args.dagger_dir) / f"{args.shard_name}.hdf5")
        if args.passenger_check:
            # Each rejected LIFT's passengers, as indices into payload["parts"].
            payload["passenger_rejects"] = [r for log in rig.passenger_log for r in log]
        if recorder is not None:
            payload["classifier_frames"] = recorder.summary()
    except EvalFailure as err:
        print(
            f"{token}_FAIL " + json.dumps({"reason": err.reason, "detail": err.detail}), flush=True
        )
        return 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        print(
            f"{token}_FAIL "
            + json.dumps({"reason": "exception", "detail": f"{err}\n{traceback.format_exc()}"}),
            flush=True,
        )
        return 1
    finally:
        try:
            if recorder is not None:
                recorder.flush()  # first: an early end keeps its frames too
            if policy is not None:
                policy.close()
            if env is not None:
                env.close()
        except Exception as err:  # noqa: BLE001 -- the token contract
            # The token is printed AFTER the try/except/finally, not here: a body that
            # already returned printed its own token and that return wins over this block,
            # so exactly one token is printed on either path (mj_dagger_batch.py's rule).
            close_error = f"{err}\n{traceback.format_exc()}"
    if close_error is not None:
        print(f"{token}_FAIL " + json.dumps({"reason": "close", "detail": close_error}), flush=True)
        return 1
    try:
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2))
        if trace is not None:
            with gzip.open(args.hold_trace, "wt") as f:
                f.writelines(json.dumps(row) + "\n" for row in trace)
    except Exception as err:  # noqa: BLE001 -- the rollout already ran; still print a token
        print(
            f"{token}_FAIL " + json.dumps({"reason": "write_out", "detail": str(err)}), flush=True
        )
        return 1
    print(f"{token}_OK " + json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
