"""Grasp synthesis on part geometry for the heap teacher (MS6 phase A design 7.1).

Torch and stdlib only. A part pose gives three body axes in the world. Every axis near the
horizontal is a closing axis: the jaws close across the part's size along it, at that axis's
heading. The approach is from above at an opening MATCHED to that closing axis, so what the
descend has to clear is not the part's neighborhood but the two LANDING spots of the jaws:
the finger tips come down ``landing_half_gap(width)`` out from the part center on both sides
of the closing axis. The jaws used to descend fully open, which asked for 52.4 mm of clear
floor on both sides whatever the part; on the three MS6 read heaps that refused 84.1 % of
every part sighting in the dish of that time (MS6 rung 1).
The closing stroke then sweeps everything between a tip and the part toward the part, which
the physics tolerates (a brick target is stowed with a brick or a plate in the sweep band,
lone-part reads 2026-09-16). A standing neighbor under a landing spot stalls the descend
early. Grasps rank by height band, then by how many standing neighbors the stroke sweeps,
then by clearance. The heap lies on the bare desk since 2026-09-18, so the only obstacles
are other parts; the dish rim this module once modeled as a hard stop is gone.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from maker_arm_lab.core import layout
from maker_arm_lab.core.parts import MAX_GRASP_WIDTH
from maker_arm_lab.core.quat import quat_apply

AXIS_MAX_TILT_RAD = math.radians(15.0)
"""A body axis this far from horizontal still counts as a closing axis."""
CLEARANCE_MIN = 0.0
"""Landing clearance (m) a grasp needs to be selected. A landing spot on a neighbor is a
negative clearance, and the model's own boundary is the pick boundary (wall reads,
2026-09-16, on the dish of that time)."""
# The open jaw, measured in the MuJoCo model in the grasp frame (2026-09-16). Each finger's
# collision hull ends in a flat tip 8 mm across the face and 2 mm along jaw travel, with a
# vertical inner face ``OPEN_HALF_GAP`` from the tool axis when the jaws are open. Above the
# tip the hull widens: about 0.46 mm of thickness and 0.32 mm of half face per mm of height.
OPEN_HALF_GAP = 0.0524
JAW_APPROACH_CLEARANCE = 0.004
"""Slack (m) each jaw inner face keeps over the grasp width while the tips descend.

The rig holds this opening from HOVER to CLOSE instead of descending fully open
(``OracleRig.actions``), so the landing spots come down beside the part rather than
``OPEN_HALF_GAP`` out from it. 4 mm is the value the bottom-pick probe drove on a lone 2x4
brick at every wall distance from 70 mm down to 20 mm, stowing 12 of 12 (the lone-part wall
table of ``docs/ms6-lever-verification.md``).
"""
JAW_HOLD_BAND = 0.008
"""How far past the commanded opening the bang-bang hold coasts (m).

The gripper channel is binary by contract, so an intermediate opening is held by flipping
the command around it, and the jaw target ramps at the URDF's jaw velocity, about 3.3 mm per
control step. The model follows the JAWS, not the set point, so it carries the flip band
plus one control step of travel. This is the direction that makes the model stricter than
the hardware, which is the safe side for a clearance test.
"""
TIP_THICKNESS = 0.002
TIP_HALF_FACE = 0.004
TIP_THICKNESS_PER_M = 0.46
TIP_HALF_FACE_PER_M = 0.32
# The jaws must reach this far down the target's side faces to hold it. A lone plate picked
# with 2.4 mm of overlap is pressed into the desk and lost, a brick with 7 mm is held
# (lone-part reads, 2026-09-16). A neighbor whose top is under the target's top by more than
# this is the surface the descend stalls on, not an obstacle the jaws have to clear.
MIN_GRIP_OVERLAP = 0.005
HEIGHT_BAND = 0.005
"""Parts within one band rank as equally high; the desk's bricks and plates tie."""
PUSH_CELL_M = 0.05
"""Grid cell of the failure memory and of the densest-cell push (design 7.2)."""
PUSH_HALF_LENGTH_M = 0.04


@dataclass(frozen=True)
class Grasp:
    part: int
    heading: float  # world heading of the closing axis, rad
    width: float  # part size along the closing axis, m
    half_height: float  # half the part size along its most vertical axis, m
    clearance: float  # m from the landing tips to the nearest standing neighbor
    height: float  # part center z, m
    swept: int = 0  # standing neighbors the closing stroke pushes toward the part


def landing_half_gap(width: float) -> float:
    """The jaw inner face's distance (m) from the tool axis while the tips come down.

    Never more than ``OPEN_HALF_GAP``: a part too wide for the jaws to straddle asks for the
    full opening, which is all the jaws have.
    """
    return min(OPEN_HALF_GAP, 0.5 * width + JAW_APPROACH_CLEARANCE + JAW_HOLD_BAND)


def body_axes_world(quat: torch.Tensor) -> torch.Tensor:
    """(3, 3): row k is body axis k in the world for a (w, x, y, z) quaternion."""
    return quat_apply(quat.reshape(1, 4), torch.eye(3, dtype=torch.float32))


def jaw_plane_normal(tilt_rad: float = 0.0) -> torch.Tensor:
    """The jaw plane's normal in the world: horizontal at tilt_rad = 0.0, leaning toward +x.

    ``candidate_grasps`` and ``HeapOracle.standing`` (core/heap_oracle.py) both call this
    through ``off_plane_axes``, so a non-zero tilt cannot make the two disagree about which
    body axis is a closing axis (MS6 rework section 7).
    """
    return torch.tensor([-math.sin(tilt_rad), 0.0, math.cos(tilt_rad)], dtype=torch.float32)


def off_plane_axes(axes: torch.Tensor, tilt_rad: float = 0.0) -> torch.Tensor:
    """(3,) each body axis's magnitude off the jaw plane's normal, at ``tilt_rad``."""
    return (axes @ jaw_plane_normal(tilt_rad)).abs()


def candidate_grasps(
    part: int,
    size: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
    tilt_rad: float = 0.0,
    closing_axes: tuple[int, ...] = (0, 1, 2),
) -> list[Grasp]:
    """Every closing axis within AXIS_MAX_TILT_RAD of the jaw plane, under MAX_GRASP_WIDTH.

    The jaw plane is horizontal at ``tilt_rad = 0.0``. ``closing_axes`` are the body axes the
    part's shape allows (``parts.Part.closing_axes``): a slope offers only its y axis.
    """
    axes = body_axes_world(quat)
    off_plane = off_plane_axes(axes, tilt_rad)
    half_height = 0.5 * float(size[int(off_plane.argmax())])
    found = []
    for k in closing_axes:
        if float(off_plane[k]) > math.sin(AXIS_MAX_TILT_RAD):
            continue
        width = float(size[k])
        if width > MAX_GRASP_WIDTH:
            continue
        heading = math.atan2(float(axes[k, 1]), float(axes[k, 0]))
        found.append(Grasp(part, heading, width, half_height, float("nan"), float(pos[2])))
    return found


def neighbor_extents(sizes: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,) horizontal radius and (B,) vertical half extent of each part's world-aligned box.

    The world-aligned half extents are ``0.5 * |R|^T size``; the radius circumscribes the box's
    footprint. The radius is the cheap lower bound ``jaw_clearance`` screens neighbors with
    before it measures the exact footprint.
    """
    rxy, hz = [], []
    for b in range(sizes.shape[0]):
        half = 0.5 * body_axes_world(quat[b]).abs().T @ sizes[b]
        rxy.append(math.hypot(float(half[0]), float(half[1])))
        hz.append(float(half[2]))
    return torch.tensor(rxy), torch.tensor(hz)


_CORNERS = torch.tensor(
    [[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)]
)


def footprint(
    size: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor
) -> list[tuple[float, float]]:
    """The part's top-down footprint: the convex hull of its eight box corners, world xy."""
    corners = (_CORNERS * size) @ body_axes_world(quat) + pos
    return convex_hull([(float(x), float(y)) for x, y in corners[:, :2].tolist()])


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain, counterclockwise, no collinear points."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def half(seq):
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2 and _cross(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]

    return half(pts) + half(pts[::-1])


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_segment_distance(p, a, b) -> float:
    ax, ay = b[0] - a[0], b[1] - a[1]
    length2 = ax * ax + ay * ay
    t = (
        0.0
        if length2 == 0.0
        else max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / length2))
    )
    return math.hypot(p[0] - a[0] - t * ax, p[1] - a[1] - t * ay)


def polygon_distance(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Signed distance between two convex polygons: the gap, or minus the penetration depth."""
    # Separating axes: every edge normal of both polygons. The largest signed separation
    # over the axes is the penetration depth when they overlap; positive means they do not.
    worst = -math.inf
    for poly in (a, b):
        n = len(poly)
        for k in range(n):
            ex, ey = poly[(k + 1) % n][0] - poly[k][0], poly[(k + 1) % n][1] - poly[k][1]
            norm = math.hypot(ex, ey)
            if norm == 0.0:
                continue
            nx, ny = -ey / norm, ex / norm
            pa = [nx * x + ny * y for x, y in a]
            pb = [nx * x + ny * y for x, y in b]
            worst = max(worst, min(pa) - max(pb), min(pb) - max(pa))
    if worst >= 0.0:
        # Disjoint: the exact gap is attained at a vertex of one and an edge of the other.
        best = math.inf
        for p in a:
            for k in range(len(b)):
                best = min(best, _point_segment_distance(p, b[k], b[(k + 1) % len(b)]))
        for p in b:
            for k in range(len(a)):
                best = min(best, _point_segment_distance(p, a[k], a[(k + 1) % len(a)]))
        return best
    return worst


def landing_footprints(
    pos: torch.Tensor,
    heading: float,
    height_above_tips: float,
    half_gap: float = OPEN_HALF_GAP,
) -> list[list[tuple[float, float]]]:
    """The two jaw landing spots as world-xy rectangles, at a height above the tips."""
    h = max(0.0, height_above_tips)
    thickness = TIP_THICKNESS + TIP_THICKNESS_PER_M * h
    half_face = TIP_HALF_FACE + TIP_HALF_FACE_PER_M * h
    c, s = math.cos(heading), math.sin(heading)
    boxes = []
    for side in (1.0, -1.0):
        corners = []
        for lx in (side * half_gap, side * (half_gap + thickness)):
            for ly in (-half_face, half_face):
                corners.append((float(pos[0]) + c * lx - s * ly, float(pos[1]) + s * lx + c * ly))
        boxes.append(convex_hull(corners))
    return boxes


def sweep_footprints(
    pos: torch.Tensor, heading: float, width: float, outer: float = OPEN_HALF_GAP
) -> list[list[tuple[float, float]]]:
    """The band each closing tip crosses, from ``outer`` along the axis to the part's face."""
    c, s = math.cos(heading), math.sin(heading)
    boxes = []
    for side in (1.0, -1.0):
        corners = []
        for lx in (side * width / 2, side * outer):
            for ly in (-TIP_HALF_FACE, TIP_HALF_FACE):
                corners.append((float(pos[0]) + c * lx - s * ly, float(pos[1]) + s * lx + c * ly))
        boxes.append(convex_hull(corners))
    return boxes


def jaw_clearance(
    pos: torch.Tensor,
    heading: float,
    width: float,
    half_height: float,
    other_pos: torch.Tensor,
    other_hz: torch.Tensor,
    other_footprints: list[list[tuple[float, float]]],
) -> tuple[float, int]:
    """``(clearance, swept)`` for a grasp at ``pos`` closing along ``heading``.

    ``clearance`` is the horizontal distance from the landing tips to the nearest standing
    neighbor, negative when a footprint overlaps a landing spot and infinite with none. A
    neighbor stands when its top reaches into the target's grip band, the top
    ``MIN_GRIP_OVERLAP`` of its side faces. ``swept`` counts the standing neighbors whose
    footprint crosses a closing stroke.
    """
    bottom = float(pos[2]) - half_height
    band = float(pos[2]) + half_height - MIN_GRIP_OVERLAP
    half_gap = landing_half_gap(width)
    standing = torch.nonzero(other_pos[:, 2] + other_hz > band).flatten().tolist()
    clearance = math.inf
    swept = 0
    if standing:
        sweeps = sweep_footprints(pos, heading, width, half_gap)
        for k in standing:
            reach = float(other_pos[k, 2] + other_hz[k]) - bottom
            spots = landing_footprints(pos, heading, reach, half_gap)
            clearance = min(clearance, min(polygon_distance(t, other_footprints[k]) for t in spots))
            if any(polygon_distance(b, other_footprints[k]) < 0.0 for b in sweeps):
                swept += 1
    return clearance, swept


def rank_grasps(
    sizes: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
    candidates: torch.Tensor,
    tilt_rad: float = 0.0,
    closing_axes: Sequence[tuple[int, ...]] | None = None,
) -> list[Grasp]:
    """Every grasp of every candidate part with its clearance, best first.

    Other parts enter through ``neighbor_extents`` and ``footprint``. ``closing_axes`` is
    one axis tuple per part (``parts.Part.closing_axes``); None lets every part offer all
    three. The order is the height band (highest first), then the swept count, then the
    clearance, then the narrower width: a wide grasp closes on one side first and slides the
    part away (docs/ms6-matched-jaw-opening.md section 5), so it is the last resort at a tie.
    """
    _, hz = neighbor_extents(sizes, quat)
    prints = [footprint(sizes[b], pos[b], quat[b]) for b in range(sizes.shape[0])]
    ranked: list[Grasp] = []
    for b in torch.nonzero(candidates).flatten().tolist():
        others = torch.ones(sizes.shape[0], dtype=torch.bool)
        others[b] = False
        other_pos, other_hz = pos[others], hz[others]
        other_prints = [prints[k] for k in torch.nonzero(others).flatten().tolist()]
        axes_b = (0, 1, 2) if closing_axes is None else closing_axes[b]
        for g in candidate_grasps(b, sizes[b], pos[b], quat[b], tilt_rad, axes_b):
            clearance, swept = jaw_clearance(
                pos[b], g.heading, g.width, g.half_height, other_pos, other_hz, other_prints
            )
            ranked.append(replace(g, clearance=clearance, swept=swept))
    ranked.sort(key=lambda g: (-math.floor(g.height / HEIGHT_BAND), g.swept, -g.clearance, g.width))
    return ranked


def select_grasp(ranked: list[Grasp]) -> Grasp | None:
    """The first ranked grasp that clears its neighbors, or None."""
    for g in ranked:
        if g.clearance >= CLEARANCE_MIN:
            return g
    return None


def densest_cell(xy: torch.Tensor, cell: float = PUSH_CELL_M) -> tuple[float, float]:
    """Center of the ``cell``-sized grid cell holding the most points."""
    cells = Counter((math.floor(x / cell), math.floor(y / cell)) for x, y in xy.tolist())
    (cx, cy), _ = cells.most_common(1)[0]
    return ((cx + 0.5) * cell, (cy + 0.5) * cell)


PUSH_START_GAP_M = 0.004
"""Gap (m) between the pushed part's footprint and the closed tips' landing spot."""
PUSH_REACH_MARGIN_M = TIP_HALF_FACE
"""The closed tips' center stays this far (m) inside the reach band."""


def clamp_to_reach(
    point: tuple[float, float], margin: float = PUSH_REACH_MARGIN_M
) -> tuple[float, float]:
    """Pull a point along its own radius from the base until it is ``margin`` inside the band.

    The band is ``layout.SCATTER_MIN_RADIUS`` to ``layout.SCATTER_MAX_RADIUS``. A point
    at the base itself has no radius and is returned as it is.
    """
    radius = math.hypot(*point)
    if radius == 0.0:
        return point
    low = layout.SCATTER_MIN_RADIUS + margin
    high = layout.SCATTER_MAX_RADIUS - margin
    target = min(max(radius, low), high)
    if target == radius:
        return point
    scale = target / radius
    return (point[0] * scale, point[1] * scale)


def tip_footprint(point: tuple[float, float]) -> list[tuple[float, float]]:
    """The closed tips' footprint at ``point``, as a square of the tip's half face."""
    return convex_hull(
        [
            (point[0] + sx * TIP_HALF_FACE, point[1] + sy * TIP_HALF_FACE)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        ]
    )


def push_segment(
    center: tuple[float, float], heading: float, half_length: float = PUSH_HALF_LENGTH_M
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Start and end of a straight sweep through ``center`` along ``heading``, in reach."""
    dx, dy = half_length * math.cos(heading), half_length * math.sin(heading)
    return clamp_to_reach((center[0] - dx, center[1] - dy)), clamp_to_reach(
        (center[0] + dx, center[1] + dy)
    )
