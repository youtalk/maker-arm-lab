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
    make_induction,
    run_batch,
    run_meta,
    run_stats,
    write_meta,
)
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig

TASK = "MakerArm-LegoSort-Datagen-mujoco"
NUM_BRICKS = 6


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
    env = MjLegoSortEnv(
        args.num_envs,
        draw,
        num_bricks=NUM_BRICKS,
        cameras=not args.no_cameras,
        start_pose_offset_rad=0.05
        if args.start_pose_offset_rad is None
        else args.start_pose_offset_rad,
        export_dir=args.out_dir,
        shard_name=args.shard_name,
        export_all=args.export_mode == "all",
    )
    env.reset(seed=args.run_seed)
    induction = make_induction(args)
    rig = OracleRig(env, bl, noise, random.Random(args.run_seed), induction=induction)
    meta = run_meta(
        args,
        draw,
        noise,
        induction,
        num_bricks=NUM_BRICKS,
        export_mode="EXPORT_ALL" if args.export_mode == "all" else "EXPORT_SUCCEEDED_ONLY",
        backend="mujoco",
        task=TASK,
    )
    print("RUN_META " + json.dumps(meta), flush=True)
    counters = run_batch(env, rig, args.attempts, args.max_wall_minutes)
    stats = run_stats(
        counters, target_attempts=args.attempts, num_envs=args.num_envs, probe=args.probe
    )
    print("RUN_STATS " + json.dumps(stats), flush=True)
    write_meta(args.out_dir, args.shard_name, meta, stats, counters["episodes"])
    ok = args.probe or counters["exported"] > 0
    return ok, counters["exported"], counters["attempts"]


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
