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
"""

import argparse
import json
import os
import random
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

from maker_arm_lab.core.batch_loop import (
    add_batch_arguments,
    build_run,
    heap_episode_length_s,
    heap_parts,
    make_induction,
    run_batch,
    run_meta,
    run_stats,
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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_batch_arguments(parser)
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
    )
    env.reset(seed=args.run_seed)
    induction = make_induction(args)
    rig_rng = random.Random(args.run_seed)
    factory = None
    if parts is not None:

        def factory(i, layout_i, episode_draw):
            return HeapOracle(
                layout_i,
                env.parts,
                int(env.destination_bins[i]),
                episode_draw,
                induction=induction,
                rng=rig_rng,
                cycle_budget=args.cycle_budget,
            )

    rig = OracleRig(
        env,
        bl,
        noise,
        rig_rng,
        induction=induction,
        transport_speed=args.transport_speed,
        oracle_factory=factory,
    )
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
    counters = run_batch(env, rig, args.attempts, args.max_wall_minutes)
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
