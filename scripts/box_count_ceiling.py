"""MS7e box count ceiling (design section 4.4 item 4): how often the box draw places each count.

For every count and every seed in ``--seed`` .. ``--seed + --seeds - 1``, one
``core.desk_boxes.draw_boxes`` from ``random.Random(seed)`` on the given band. A count is
feasible if at least ``--feasible`` of the seeds place it; the ceiling is the largest feasible
count. The counts run in ascending order and stop at the first infeasible one: a larger count
puts at least as many boxes on each side. Pure geometry, CPU only, no MuJoCo.

    PYTHONPATH=$PWD/source .venv/bin/python scripts/box_count_ceiling.py --box_min 0.07 \\
        --band_x=0.05,0.21 --procs 8 --out /tmp/ceiling.json

Prints exactly one ``CEILING_OK {json}`` or ``CEILING_FAIL {json}`` line on every exit path,
argparse errors included, and exits through ``os._exit``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from multiprocessing import Pool

from maker_arm_lab.core import desk_boxes


class _TokenEmittingParser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        print("CEILING_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def _pair(text: str) -> tuple[float, float]:
    lo, hi = (float(v) for v in text.split(","))
    return lo, hi


def parse_args(argv=None) -> argparse.Namespace:
    parser = _TokenEmittingParser(description=__doc__)
    parser.add_argument("--band_x", type=_pair, default=desk_boxes.DESK_BOX_BAND_X)
    parser.add_argument("--band_y", type=_pair, default=desk_boxes.DESK_BOX_BAND_Y)
    parser.add_argument("--box_min", type=float, default=desk_boxes.BOX_MIN)
    parser.add_argument("--box_max", type=float, default=desk_boxes.BOX_MAX)
    parser.add_argument("--aspect", type=float, default=desk_boxes.BOX_MAX_ASPECT)
    parser.add_argument("--depth", type=_pair, default=desk_boxes.BOX_DEPTH)
    parser.add_argument("--back", type=float, default=desk_boxes.FRAME_BACK_X)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--size_draws", type=int, default=desk_boxes.SIZE_DRAWS)
    parser.add_argument("--counts", type=lambda t: [int(v) for v in t.split(",")], default=[4])
    parser.add_argument("--seed", type=int, default=20290000)
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--feasible", type=float, default=0.99)
    parser.add_argument("--procs", type=int, default=1)
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args(argv)


def _one(job):
    args, n, seed = job
    t0 = time.perf_counter()
    try:
        poses, sizes = desk_boxes.draw_boxes(
            random.Random(seed),
            n,
            band_x=args.band_x,
            band_y=args.band_y,
            sides=(args.box_min, args.box_max),
            aspect=args.aspect,
            depth=args.depth,
            back=args.back,
            sequential=args.sequential,
            size_draws=args.size_draws,
        )
    except RuntimeError:
        return n, None, None, time.perf_counter() - t0
    return n, poses, sizes, time.perf_counter() - t0


def run(args) -> dict:
    rows = {}
    for n in sorted(args.counts):
        with Pool(args.procs) as pool:
            mine = pool.map(_one, [(args, n, args.seed + k) for k in range(args.seeds)])
        placed = [r for r in mine if r[1] is not None]
        sides = [v for r in placed for size in r[2] for v in size[:2]]
        yaws = [abs(p[2]) for r in placed for p in r[1]]
        rows[n] = {
            "placed": len(placed),
            "seeds": len(mine),
            "feasible": len(placed) >= args.feasible * len(mine),
            "side_mean": round(statistics.fmean(sides), 4) if sides else None,
            "yaw_over_30_share": (
                round(sum(y > math.radians(30.0) for y in yaws) / len(yaws), 3) if yaws else None
            ),
            "draw_ms_mean": round(1000.0 * statistics.fmean(r[3] for r in mine), 2),
            "draw_ms_max": round(1000.0 * max(r[3] for r in mine), 1),
        }
        if not rows[n]["feasible"]:
            break  # a larger count puts at least as many boxes on each side
    feasible = [n for n, row in rows.items() if row["feasible"]]
    return {
        "band_x": list(args.band_x),
        "band_y": list(args.band_y),
        "box_min": args.box_min,
        "box_max": args.box_max,
        "aspect": args.aspect,
        "depth": list(args.depth),
        "back": args.back,
        "sampler": "sequential" if args.sequential else "rejection",
        "size_draws": args.size_draws,
        "draw_tries": desk_boxes.DRAW_TRIES,
        "seed": args.seed,
        "seeds": args.seeds,
        "ceiling": max(feasible) if feasible else None,
        "rows": rows,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
        print("CEILING_OK " + json.dumps(result), flush=True)
        return 0
    except Exception as err:  # noqa: BLE001 -- any failure must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        print("CEILING_FAIL " + json.dumps({"reason": "exception", "detail": detail}), flush=True)
        return 1


if __name__ == "__main__":
    os._exit(main())
