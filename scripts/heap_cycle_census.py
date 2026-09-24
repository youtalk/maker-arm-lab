"""The cycle census of a heap gate read: how far the cycles that stowed nothing got.

Stdlib only, runs under system python3 like the other supervisors. Reads the ``eval_NNN.json``
payloads ``scripts/run_gate_eval.py --heap_k`` leaves in its out_dir and sums what
``core/loop.py`` books per cycle. It reports the rows MS6 phase B design 5.10 asks for: the
census classes, the successful cycles that stowed a part other than the target, the first
close distance against the outcome, and the step count per end reason. With desk boxes it adds
the MS7a rows: ``wrong_box`` among the classes, parts stowed per class and per box, and the
parts at rest in a box that is not their own. Prints exactly one of
``CENSUS_OK {json}`` or ``CENSUS_FAIL {json}`` on stdout.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

CLOSE_CM_EDGES = (1.0, 2.0, 3.0)  # cm: the bins of the first close distance
TRAY_INTERIOR_XY = ((0.13, 0.26), (-0.10, 0.10))
"""m: the MS7a tray interior, ``layout.tray_bounds()`` (pinned by tests/test_heap_cycle_census.py)."""
WALL_BAND_M = 0.03
"""m: a part left this close to a tray wall is left at a wall: ``grasp.RAKE_BAND_M``, the band
the teacher rakes from (pinned by the same test)."""


def at_tray_wall(x: float, y: float) -> bool:
    """Inside the tray interior and within WALL_BAND_M of one of its walls."""
    (x0, x1), (y0, y1) = TRAY_INTERIOR_XY
    inside = x0 <= x <= x1 and y0 <= y <= y1
    return inside and min(x - x0, x1 - x, y - y0, y1 - y) < WALL_BAND_M


def _close_bin(cm: float | None) -> str:
    if cm is None:
        return "none"
    for edge in CLOSE_CM_EDGES:
        if cm < edge:
            return f"<{edge:g}"
    return f">={CLOSE_CM_EDGES[-1]:g}"


def read_payloads(paths: list[Path]) -> dict:
    classes: Counter = Counter()
    closes: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # bin -> [stowed, cycles]
    steps: dict[str, list[int]] = defaultdict(list)  # end reason -> cycle steps
    summary = {
        "draws": len(paths),
        "episodes": 0,
        "parts_total": 0,
        "parts_stowed": 0,
        "cycles": 0,
        "rakes": 0,
        "left_parts": 0,
        "left_at_wall": 0,
    }
    target_stows = other_stows = unaligned = 0
    # MS7a desk boxes: parts stowed in their own box and parts in total, per class and per box.
    by_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_box: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    wrong_box_parts = 0
    for path in paths:
        for ep in json.loads(Path(path).read_text())["episodes"]:
            if "cycle_classes" not in ep:
                raise KeyError(f"{path}: no cycle_classes, the payload predates the census")
            summary["episodes"] += 1
            summary["parts_total"] += int(ep["num_bricks"])
            summary["parts_stowed"] += int(ep["bricks_stowed"])
            summary["cycles"] += int(ep["cycles"])
            summary["rakes"] += int(ep.get("rakes", 0))  # MS7a tray: they run between cycles
            # The parts the episode left out of every box (part, x, y, z, rakes, failures).
            left = ep.get("left_parts", [])
            summary["left_parts"] += len(left)
            summary["left_at_wall"] += sum(1 for row in left if at_tray_wall(row[1], row[2]))
            classes.update(ep["cycle_classes"])
            for name, row in ep.get("stowed_by_class", {}).items():
                by_class[name][0] += row["stowed"]
                by_class[name][1] += row["total"]
            for m, row in enumerate(ep.get("stowed_by_box", [])):
                by_box[m][0] += row["stowed"]
                by_box[m][1] += row["total"]
            wrong_box_parts += int(ep.get("wrong_box_parts", 0))
            for ok, cm in zip(ep["cycle_outcomes"], ep["cycle_first_close_cm"]):
                row = closes[_close_bin(cm)]
                row[0] += bool(ok)
                row[1] += 1
            for reason, n in zip(ep["cycle_reasons"], ep["cycle_steps"]):
                steps[reason].append(int(n))
            won = [t for t, ok in zip(ep["cycle_targets"], ep["cycle_outcomes"]) if ok]
            if len(won) != len(ep["stowed_parts"]):
                unaligned += 1  # a cycle stowed two parts: its stows cannot be paired
                continue
            hits = sum(t == p for t, p in zip(won, ep["stowed_parts"]))
            target_stows += hits
            other_stows += len(won) - hits
    total = summary["cycles"]
    summary["stowed_fraction"] = (
        summary["parts_stowed"] / summary["parts_total"] if summary["parts_total"] else 0.0
    )
    summary["classes"] = {
        name: {"cycles": n, "share": round(n / total, 4)} for name, n in classes.most_common()
    }
    summary["stows"] = {
        "target": target_stows,
        "other_part": other_stows,
        "episodes_not_paired": unaligned,
    }
    summary["first_close_cm"] = {k: {"stowed": v[0], "cycles": v[1]} for k, v in closes.items()}
    if by_class:
        summary["stowed_by_class"] = {
            k: {"stowed": v[0], "total": v[1]} for k, v in by_class.items()
        }
        summary["stowed_by_box"] = {
            str(m): {"stowed": v[0], "total": v[1]} for m, v in sorted(by_box.items())
        }
        summary["wrong_box_parts"] = wrong_box_parts
    summary["cycle_steps"] = {
        reason: {
            "cycles": len(v),
            "median": statistics.median(v),
            "p90": sorted(v)[int(0.9 * (len(v) - 1))],
            "max": max(v),
        }
        for reason, v in sorted(steps.items())
    }
    return summary


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print(f"CENSUS_FAIL usage: {message}", flush=True)
        super().error(message)


def main(argv=None) -> int:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--evals", nargs="+", required=True, help="eval_NNN.json paths or globs")
    parser.add_argument("--out", type=Path, default=None, help="write the summary JSON here too")
    args = parser.parse_args(argv)
    try:
        paths = sorted(Path(p) for pattern in args.evals for p in glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no payload matches {args.evals}")
        summary = read_payloads(paths)
        if args.out is not None:
            args.out.write_text(json.dumps(summary, indent=2))
    except Exception as exc:  # noqa: BLE001 -- every failure must become a token
        print("CENSUS_FAIL " + json.dumps({"reason": f"{type(exc).__name__}: {exc}"}), flush=True)
        traceback.print_exc()
        return 1
    print("CENSUS_OK " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
