"""Summarize scripts/mj_heap_drop_sweep.py's samples.jsonl into one row per grid cell.

Stdlib only. Groups samples by their grid cell (tray size, wall height, drop band, settle
steps), averages the design's five counts and the settle speed across that cell's layout
seeds, and marks the cell that matches ``core/layout.py`` today as the baseline. Prints a
table and one ``SWEEP_READ {json}`` line.

    python3 scripts/heap_sweep_read.py /tmp/sweep0/samples.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict

COUNT_FIELDS = (
    "outside",
    "wall_blocked",
    "blocked",
    "no_grasp",
    "stacked",
    "unreachable",
    "speed_max",
)
CELL_FIELDS = (
    "tray_w_mm",
    "tray_h_mm",
    "wall_height_mm",
    "drop_min_mm",
    "drop_per_part_mm",
    "settle_steps",
)
# core/layout.py today: TRAY_INTERIOR, TRAY_WALL_HEIGHT, HEAP_DROP_Z_MIN, HEAP_DROP_Z_PER_PART,
# HEAP_SETTLE_STEPS, in millimeters. A literal so this script stays stdlib-only.
BASELINE_CELL = (130.0, 200.0, 20.0, 30.0, 1.2, 480)


def read_samples(paths: list[str]) -> list[dict]:
    samples = []
    for path in paths:
        with open(path) as f:
            samples += [json.loads(line) for line in f if line.strip()]
    return samples


def group_cells(samples: list[dict]) -> dict[tuple, list[dict]]:
    cells = defaultdict(list)
    for s in samples:
        cells[tuple(s[k] for k in CELL_FIELDS)].append(s)
    return cells


def summarize_cell(key: tuple, samples: list[dict]) -> dict:
    row = dict(zip(CELL_FIELDS, key))
    row["n"] = len(samples)
    row["heap_k"] = samples[0]["heap_k"]
    row["baseline"] = key == BASELINE_CELL
    for field in COUNT_FIELDS:
        values = [s[field] for s in samples]
        row[f"{field}_mean"] = sum(values) / len(values)
        row[f"{field}_min"] = min(values)
        row[f"{field}_max"] = max(values)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="+", help="samples.jsonl files")
    args = parser.parse_args(argv)
    samples = read_samples(args.samples)
    cells = [summarize_cell(key, rows) for key, rows in group_cells(samples).items()]
    cells.sort(key=lambda c: tuple(c[k] for k in CELL_FIELDS))
    header = ["tray", "wall_mm", "drop", "settle", "n", "base"]
    header += [f"{f}_mean" for f in COUNT_FIELDS]
    print(" ".join(f"{h:>12s}" for h in header))
    for c in cells:
        tray = f"{c['tray_w_mm']:g}x{c['tray_h_mm']:g}"
        drop = f"{c['drop_min_mm']:g}:{c['drop_per_part_mm']:g}"
        row = [
            tray,
            c["wall_height_mm"],
            drop,
            c["settle_steps"],
            c["n"],
            "*" if c["baseline"] else "",
        ]
        row += [round(c[f"{f}_mean"], 2) for f in COUNT_FIELDS]
        print(" ".join(f"{v!s:>12s}" for v in row))
    summary = {"samples": len(samples), "cells": cells}
    print("SWEEP_READ " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
