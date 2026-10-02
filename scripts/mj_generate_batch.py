"""Batch generation on the MuJoCo backend: one process = one generation run.

The MuJoCo twin of ``scripts/generate_batch.py``: the same flags (minus Isaac's AppLauncher
ones), the same ``OracleRig``, the same ``RUN_META`` / ``RUN_STATS`` / ``BATCH_OK`` contract
and the same ``<shard_name>.hdf5`` + ``<shard_name>.meta.json`` outputs, so
``scripts/generate_dataset.py --backend mujoco`` drives it exactly as it drives the Isaac
script. Runs under the project's own venv:

    uv run python scripts/mj_generate_batch.py --out_dir /tmp/b0 --attempts 12 --num_envs 12 --probe

Everything that decides what gets recorded is in ``maker_arm_lab.core.batch_loop``; this file
only builds the environment. ``--headless`` and ``--enable_cameras`` are accepted and ignored
so a command line written for the Isaac script still runs here.

MS6 heap runs (``--heap_k`` > 0) build the heap env and drive it with ``HeapOracle`` instead
of the six-brick oracle:

    uv run python scripts/mj_generate_batch.py --out_dir /tmp/h0 --attempts 12 --num_envs 12 \\
        --heap_k 50 --parts_split heldout --probe

``--perceived`` (MS7c, a heap with desk boxes) records the target row, the frozen marker and
the box row from ``core.perception`` at ``--noise_scales``; the teacher still acts on the truth.
"""

import argparse
import json
import os
import random
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

from maker_arm_lab.core import perception
from maker_arm_lab.core.batch_loop import (
    MS7F_READY,
    add_batch_arguments,
    add_ms7e_arguments,
    add_ms7f_arguments,
    build_run,
    check_ms7e,
    check_ms7f,
    check_teacher_waits,
    heap_classes,
    heap_cycle_budget,
    heap_episode_length_s,
    heap_parts,
    install_heap_events,
    make_induction,
    run_batch,
    run_meta,
    run_stats,
    teacher_waits,
    write_meta,
)
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig

TASK = "MakerArm-LegoSort-Datagen-mujoco"
HEAP_TASK = "MakerArm-LegoHeap-Datagen-mujoco"
NUM_BRICKS = 6


def exported_count(counters: dict, heap: bool) -> int:
    """The count BATCH_OK reports and gates on.

    A heap episode's "exported" only rises when every part of the heap got stowed, but the
    phase's own target is a 0.90 stowed fraction per part, so a heap run counts every episode
    the recorder wrote instead (``exported_written``). The six-brick run is unchanged: it
    keeps counting only fully-successful episodes (``exported``).
    """
    return counters["exported_written"] if heap else counters["exported"]


class _TokenEmittingParser(argparse.ArgumentParser):
    """A parser whose usage errors still print the BATCH_FAIL token."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr, flush=True)
        print(f"BATCH_FAIL usage: {message}", flush=True)
        raise SystemExit(2)


def parse_args(argv=None) -> argparse.Namespace:
    parser = _TokenEmittingParser(description=__doc__)
    add_batch_arguments(parser)
    parser.set_defaults(cycle_budget=None)  # heap_cycle_budget(K) once --heap_k is read
    add_ms7e_arguments(parser)
    add_ms7f_arguments(parser, transport_speed=False)  # --transport_speed is a batch flag
    parser.add_argument(
        "--perceived",
        action="store_true",
        help="MS7c: record the target row, the marker and the box row as core.perception "
        "reports them at --noise_scales; the teacher still acts on the truth. Needs --heap_k "
        "> 0 and --desk_boxes > 0",
    )
    parser.add_argument(
        "--noise_scales",
        type=perception.parse_scales,
        default="",
        help='the perception knob scales under --perceived, e.g. "pos=2,miss=0" or "all=0"; '
        "a knob not named stays at 1 (nominal)",
    )
    parser.add_argument(
        "--headless", action="store_true", help="accepted for Isaac-command parity; no effect"
    )
    parser.add_argument(
        "--enable_cameras",
        action="store_true",
        help="accepted for parity; cameras follow --no_cameras",
    )
    args = parser.parse_args(argv)
    for name in ("run_seed", "layout_seed"):
        if not (0 <= getattr(args, name) < 2**32):
            parser.error(f"--{name} is outside numpy's seed range (0 <= seed < 2**32)")
    if args.cycle_budget is None:
        args.cycle_budget = heap_cycle_budget(args.heap_k)
    check_ms7e(parser, args)
    # The recorder's walk pace predates MS7f (the MS3 speed ladder): it is built here.
    # --book_miss is the heap loop's; the rig driving here keeps its own failure memory.
    # --missed_lift reads the serving loop's carries; the teacher here holds its own.
    # --cell_failure_cap is the serving loop's selector; the teacher keeps CELL_FAILURE_CAP.
    serving = {"book_miss", "missed_lift", "cell_failure_cap"}
    check_ms7f(parser, args, (MS7F_READY - serving) | {"transport_speed"})
    check_teacher_waits(parser, args)
    if args.lift_budget:
        parser.error("--lift_budget cuts a served cycle: the recorder serves no policy")
    if args.end_at_done and args.attempts != args.num_envs:
        parser.error("--end_at_done ends one episode an env: it needs --attempts == --num_envs")
    for name in ("classifier", "passenger_drop", "lift_view_steps"):
        if getattr(args, name):
            # MS7f C serves the policy loop's views; the recorder's teacher acts on the truth.
            parser.error(f"--{name} reads the serving loop's views: the recorder has none")
    if args.perceived and (args.heap_k <= 0 or args.desk_boxes <= 0):
        parser.error(
            "--perceived records a heap sorted into desk boxes: it needs --heap_k > 0 "
            "and --desk_boxes > 0"
        )
    return args


def run(args: argparse.Namespace) -> tuple[bool, int, int]:
    """Returns ``(ok, exported, attempts)``."""
    os.makedirs(args.out_dir, exist_ok=True)
    draw, bl, noise = build_run(args)
    parts = heap_parts(args)
    episode_length_s = args.episode_length_s
    if episode_length_s is None:
        episode_length_s = (
            heap_episode_length_s(len(parts), args.cycle_budget) if parts is not None else 180.0
        )
    env = MjLegoSortEnv(
        args.num_envs,
        draw,
        num_bricks=NUM_BRICKS if parts is None else len(parts),
        parts=parts,
        cameras=not args.no_cameras,
        start_pose_offset_rad=0.05
        if args.start_pose_offset_rad is None
        else args.start_pose_offset_rad,
        episode_length_s=episode_length_s,
        export_dir=args.out_dir,
        shard_name=args.shard_name,
        # Heap runs export every episode: the actor channel marks the cycles to keep (design H7).
        export_all=args.export_mode == "all" or parts is not None,
        episode_draws=args.episode_draws,
        frame_jitter=args.frame_jitter,
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
    )
    env.reset(seed=args.run_seed)
    induction = make_induction(args)
    rig_rng = random.Random(args.run_seed)
    factory = None
    if parts is not None:

        def factory(i, layout_i, episode_draw):
            orc = HeapOracle(
                layout_i,
                env.parts,
                env.destination_bins[i],
                episode_draw,
                induction=induction,
                rng=rig_rng,
                cycle_budget=args.cycle_budget,
                boxes=env.desk_box_poses[i],
                box_sizes=env.desk_box_sizes[i],
                tray=env.tray,
                tray_lean_deg=env.tray_lean_deg,
                wall_band_cont=args.wall_band_cont,
                box_sets=env.part_box_sets(i),  # MS7f: one teacher, one load record an episode
                tray_wall_height=env.tray_wall_height,
            )
            orc.abandon_cycle = args.abandon_cycle
            return teacher_waits(orc, args)

    perceivers = None
    if args.perceived:  # MS7c design section 7.1: the teacher acts on the truth
        perceivers = [
            perception.Perceiver(args.run_seed, i, args.noise_scales, env.class_names)
            for i in range(args.num_envs)
        ]
    rig = OracleRig(
        env,
        bl,
        noise,
        rig_rng,
        induction=induction,
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
    install_heap_events(env, rig, args.box_empty)  # MS7f: the refill and the box emptying
    meta = run_meta(
        args,
        draw,
        noise,
        induction,
        num_bricks=NUM_BRICKS if parts is None else len(parts),
        export_mode="EXPORT_ALL"
        if (args.export_mode == "all" or parts is not None)
        else "EXPORT_SUCCEEDED_ONLY",
        backend="mujoco",
        task=HEAP_TASK if parts is not None else TASK,
        parts=parts,
        episode_length_s=episode_length_s,
    )
    print("RUN_META " + json.dumps(meta), flush=True)
    counters = run_batch(env, rig, args.attempts, args.max_wall_minutes, args.end_at_done)
    stats = run_stats(
        counters,
        target_attempts=args.attempts,
        num_envs=args.num_envs,
        probe=args.probe,
        # None on the six-brick task: keeps RUN_STATS byte-for-byte with today's script.
        # run_stats's heap-counter block only fires when this is not None.
        num_bricks=env.cfg.num_bricks if parts is not None else None,
    )
    print("RUN_STATS " + json.dumps(stats), flush=True)
    write_meta(args.out_dir, args.shard_name, meta, stats, counters["episodes"])
    exported = exported_count(counters, heap=parts is not None)
    ok = args.probe or exported > 0
    return ok, exported, counters["attempts"]


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        ok, exported, attempts = run(args)
    except Exception:  # noqa: BLE001 -- any failure must become BATCH_FAIL, not a crash.
        traceback.print_exc()
        ok, exported, attempts = False, 0, 0
    print(f"BATCH_OK exported={exported} attempts={attempts}" if ok else "BATCH_FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
