"""Design Q1: what is the largest tray interior the arm can pick over?

A solver read, not a physics read: the arm's own collisions with the desk and the tray walls
are not in it (that is ``mj_heap_drop_sweep.py`` and the bottom-open pick probe). This script
sweeps a grid over the desk plane and, at every point, asks ``core.global_ik`` for the descend
pose at the level tilt (``MIN_DESCEND_TILT``), at every heading in an evenly spaced set over a
half turn. A point is ``pickable`` once every heading converges inside the solver's own
tolerance, ``partial`` once some heading does, honoring the joint limits exactly as
``OracleRig`` does -- ``GlobalIK.solve`` is the same call the oracle's rig makes for a hover or
a descend, at the same restart count and the same tolerance, since both are baked into the
Rust solver rather than passed in.

From the pickable set it finds the largest axis-aligned rectangle by area, plus the best
rectangle at three aspect ratios: the current tray's (``layout.TRAY_INTERIOR``), a square, and
the widest one the pickable set admits. Each is reported with its interior size, its center,
its four corners' radii from the arm base (to read against ``layout.SCATTER_MAX_RADIUS``) and
its area against the current tray's.

    .venv/bin/python scripts/reach_map.py --out /tmp/reach.json --step 0.02

Writes the full grid and the rectangles to ``--out``, prints a text plot (one character per
grid point) to stdout, and prints exactly one ``REACH_OK {json}`` or ``REACH_FAIL {json}``
line before exiting through ``os._exit``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

from maker_arm_lab.core import layout
from maker_arm_lab.core.global_ik import (
    MIN_DESCEND_TILT,
    GlobalIK,
    GraspTarget,
    descend_pick_offset_or_raise,
    tool_axis_world,
)
from maker_arm_lab.core.grasp import OPEN_HALF_GAP
from maker_arm_lab.core.parts import BRICK_HEIGHT

# The shipping convention documented on ``GraspTarget.lean_azimuth`` itself: 0.0, +x. The
# oracle's rig (``oracle_runtime.LEAN_AZIMUTH``) commands the same value for every descend.
LEAN_AZIMUTH = 0.0


def descend_target(
    x: float, y: float, half_height: float, tilt: float, heading: float
) -> GraspTarget:
    """The ``GraspTarget`` for a descend onto a brick centered at ``(x, y, half_height)``.

    ``grasp_center = pick + offset_z * tool_axis`` (``oracle_runtime.grasp_center_for``, with
    its face-offset term dropped: that term is ``PICK_OFFSET_MEASURED[0] * face``, and
    ``PICK_OFFSET_MEASURED[0]`` is exactly 0.0 on this arm, so it vanishes rather than being
    approximated away).
    """
    offset_z, _margin = descend_pick_offset_or_raise(tilt, half_height)
    axis = tool_axis_world(tilt, LEAN_AZIMUTH)
    position = (
        x + offset_z * float(axis[0]),
        y + offset_z * float(axis[1]),
        half_height + offset_z * float(axis[2]),
    )
    return GraspTarget(position=position, tilt=tilt, jaw_heading=heading, lean_azimuth=LEAN_AZIMUTH)


def headings_over_half_turn(n: int) -> list[float]:
    """``n`` closing headings evenly spaced over a half turn (the jaw is symmetric under one)."""
    return [k * math.pi / n for k in range(n)]


def probe_point(
    ik: GlobalIK, x: float, y: float, half_height: float, tilt: float, headings: list[float]
) -> list[bool]:
    """Whether the solver converges at ``(x, y)`` for each heading, in order."""
    converged = []
    for heading in headings:
        target = descend_target(x, y, half_height, tilt, heading)
        converged.append(ik.solve(target) is not None)
    return converged


def build_axis(lo: float, hi: float, step: float) -> list[float]:
    n = round((hi - lo) / step) + 1
    return [lo + i * step for i in range(n)]


# --- the largest-rectangle scan -------------------------------------------------------------


def maximal_rectangles(mask: list[list[bool]]) -> list[tuple[int, int, int, int]]:
    """Every maximal all-True rectangle in ``mask``, as inclusive ``(row0, row1, col0, col1)``.

    The classic largest-rectangle-in-histogram scan, run once per row and every popped bar
    kept as a candidate rather than only the row's single best. A maximal-rectangle scan over
    the boolean mask is enough: it does not enumerate every narrower sub-rectangle a run also
    contains, but every candidate it returns is genuinely all-True, which is what the rectangle
    picks below need.
    """
    if not mask or not mask[0]:
        return []
    ncols = len(mask[0])
    heights = [0] * ncols
    rects: list[tuple[int, int, int, int]] = []
    for row_idx, row in enumerate(mask):
        for col in range(ncols):
            heights[col] = heights[col] + 1 if row[col] else 0
        stack: list[int] = []
        for col in range(ncols + 1):
            cur = heights[col] if col < ncols else 0
            while stack and heights[stack[-1]] >= cur:
                top = stack.pop()
                h = heights[top]
                left = stack[-1] + 1 if stack else 0
                right = col - 1
                if h > 0:
                    rects.append((row_idx - h + 1, row_idx, left, right))
            stack.append(col)
    return rects


def rectangle_info(
    rect: tuple[int, int, int, int], xs: list[float], ys: list[float], step: float
) -> dict:
    """A rectangle's report, in real units. Each grid point owns a ``step``-wide strip, so a
    rectangle spanning ``col0..col1`` (inclusive) is padded by ``step / 2`` on every side --
    otherwise a one-row or one-column rectangle would report a zero-area footprint."""
    row0, row1, col0, col1 = rect
    x0, x1 = xs[col0] - step / 2.0, xs[col1] + step / 2.0
    y0, y1 = ys[row0] - step / 2.0, ys[row1] + step / 2.0
    width, height = x1 - x0, y1 - y0
    corners = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
    current_tray_area = layout.TRAY_INTERIOR[0] * layout.TRAY_INTERIOR[1]
    area = width * height
    return {
        "x_range": [x0, x1],
        "y_range": [y0, y1],
        "size_m": [width, height],
        "center": [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
        "corner_radii_m": [math.hypot(cx, cy) for cx, cy in corners],
        "area_m2": area,
        "area_vs_current_tray": area / current_tray_area,
    }


def best_rectangles(mask: list[list[bool]], xs: list[float], ys: list[float], step: float) -> dict:
    """The counting below works in grid-cell counts (``col1 - col0 + 1``, and so on), which is
    ratio- and area-equivalent to real units up to the constant ``step`` factor; the padding
    that turns a count into a physical size (see ``rectangle_info``) is applied only when a
    winning rectangle is reported."""
    candidates = maximal_rectangles(mask)
    if not candidates:
        return {"overall": None, "current_tray_ratio": None, "square": None, "widest": None}

    def ratio_of(rect: tuple[int, int, int, int]) -> float:
        row0, row1, col0, col1 = rect
        return (col1 - col0 + 1) / (row1 - row0 + 1)

    def area_of(rect: tuple[int, int, int, int]) -> float:
        row0, row1, col0, col1 = rect
        return (col1 - col0 + 1) * (row1 - row0 + 1)

    def width_of(rect: tuple[int, int, int, int]) -> float:
        _row0, _row1, col0, col1 = rect
        return col1 - col0 + 1

    def best_at_ratio(target_ratio: float) -> tuple[int, int, int, int]:
        return min(
            candidates,
            key=lambda r: (round(abs(math.log(ratio_of(r) / target_ratio)), 6), -area_of(r)),
        )

    overall = max(candidates, key=area_of)
    current_ratio = layout.TRAY_INTERIOR[0] / layout.TRAY_INTERIOR[1]
    widest = max(candidates, key=lambda r: (width_of(r), area_of(r)))
    return {
        "overall": rectangle_info(overall, xs, ys, step),
        "current_tray_ratio": rectangle_info(best_at_ratio(current_ratio), xs, ys, step),
        "square": rectangle_info(best_at_ratio(1.0), xs, ys, step),
        "widest": rectangle_info(widest, xs, ys, step),
    }


# --- the grid sweep --------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    xs = build_axis(args.x_min, args.x_max, args.step)
    ys = build_axis(args.y_min, args.y_max, args.step)
    headings = headings_over_half_turn(args.headings)
    tilt = MIN_DESCEND_TILT
    ik = GlobalIK()

    points = []
    pickable_mask = [[False] * len(xs) for _ in ys]
    partial_mask = [[False] * len(xs) for _ in ys]
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            converged = probe_point(ik, x, y, args.half_height, tilt, headings)
            pickable = all(converged)
            partial = any(converged)
            pickable_mask[iy][ix] = pickable
            partial_mask[iy][ix] = partial
            points.append(
                {"x": x, "y": y, "converged": converged, "pickable": pickable, "partial": partial}
            )

    n_pickable = sum(row.count(True) for row in pickable_mask)
    n_partial = sum(row.count(True) for row in partial_mask) - n_pickable
    counts = {
        "total": len(points),
        "pickable": n_pickable,
        "partial": n_partial,
        "none": len(points) - n_pickable - n_partial,
    }
    rectangles = best_rectangles(pickable_mask, xs, ys, args.step)
    return {
        "grid": {
            "x_min": args.x_min,
            "x_max": args.x_max,
            "y_min": args.y_min,
            "y_max": args.y_max,
            "step": args.step,
            "headings": args.headings,
            "half_height": args.half_height,
            "tilt_rad": tilt,
            "jaw_open_half_gap_m": OPEN_HALF_GAP,
        },
        "counts": counts,
        "points": points,
        "rectangles": rectangles,
        "text_plot": text_plot(pickable_mask, partial_mask),
    }


def text_plot(pickable_mask: list[list[bool]], partial_mask: list[list[bool]]) -> str:
    """One line per row of ``y``, one character per grid point: ``#`` pickable, ``o`` partial, ``.`` none."""
    lines = []
    for pickable_row, partial_row in zip(pickable_mask, partial_mask):
        line = "".join("#" if p else ("o" if q else ".") for p, q in zip(pickable_row, partial_row))
        lines.append(line)
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x_min", type=float, default=0.05)
    parser.add_argument("--x_max", type=float, default=0.32)
    parser.add_argument("--y_min", type=float, default=-0.20)
    parser.add_argument("--y_max", type=float, default=0.20)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--headings", type=int, default=8)
    parser.add_argument("--half_height", type=float, default=BRICK_HEIGHT / 2)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
        result = run(args)
        with open(args.out, "w") as f:
            json.dump(result, f)
    except SystemExit as e:  # argparse
        print("REACH_FAIL " + json.dumps({"error": f"argparse exit {e.code}"}), flush=True)
        return 2
    except Exception as e:  # noqa: BLE001 -- any failure must become REACH_FAIL.
        traceback.print_exc()
        print("REACH_FAIL " + json.dumps({"error": repr(e)}), flush=True)
        return 1

    print(result["text_plot"], flush=True)
    print(
        "REACH_OK "
        + json.dumps(
            {"counts": result["counts"], "rectangles": result["rectangles"], "out": args.out}
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
