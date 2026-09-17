"""Grasp synthesis on part geometry for the heap teacher (MS6 phase A design 7.1).

Torch and stdlib only. A part pose gives three body axes in the world. Every axis near the
horizontal is a closing axis: the jaws close across the part's size along it, at that axis's
heading. The approach is from above with the jaws fully open, so what the descend has to
clear is not the part's neighborhood but the two LANDING spots of the open jaws: the finger
tips come down ``OPEN_HALF_GAP`` out from the part center on both sides of the closing axis.
The closing stroke then sweeps everything between a tip and the part toward the part, which
the physics tolerates (a brick target is stowed with a brick or a plate in the sweep band,
lone-part reads 2026-09-16). A tray wall under a landing spot is a hard stop: a jaw whose
inner face lands past the wall's inner face never picks, one that lands inside picks even
when it brushes the wall top (wall reads 2026-09-16). A standing neighbor under a landing
spot stalls the descend early. Grasps rank by height band, then by how many standing
neighbors the stroke sweeps, then by clearance.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace

import torch

from maker_arm_lab.core import layout
from maker_arm_lab.core.parts import MAX_GRASP_WIDTH
from maker_arm_lab.core.quat import quat_apply

AXIS_MAX_TILT_RAD = math.radians(15.0)
"""A body axis this far from horizontal still counts as a closing axis."""
CLEARANCE_MIN = 0.0
"""Landing clearance (m) a grasp needs to be selected.

The model's own boundary already matches the physics: a jaw whose tip lands 0.6 mm inside
the wall face picks, one that lands 2.6 mm past it does not (wall reads, 2026-09-16).
"""
# The open jaw, measured in the MuJoCo model in the grasp frame (2026-09-16). Each finger's
# collision hull ends in a flat tip 8 mm across the face and 2 mm along jaw travel, with a
# vertical inner face ``OPEN_HALF_GAP`` from the tool axis when the jaws are open. Above the
# tip the hull widens: about 0.46 mm of thickness and 0.32 mm of half face per mm of height.
OPEN_HALF_GAP = 0.0524
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
    clearance: float  # m from the landing tips to the nearest wall or standing neighbor
    height: float  # part center z, m
    swept: int = 0  # standing neighbors the closing stroke pushes toward the part
    wall: float = math.inf  # m from the landing tips to the nearest tray wall


def body_axes_world(quat: torch.Tensor) -> torch.Tensor:
    """(3, 3): row k is body axis k in the world for a (w, x, y, z) quaternion."""
    return quat_apply(quat.reshape(1, 4), torch.eye(3, dtype=torch.float32))


def candidate_grasps(
    part: int, size: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor
) -> list[Grasp]:
    """Every closing axis within AXIS_MAX_TILT_RAD of horizontal and under MAX_GRASP_WIDTH."""
    axes = body_axes_world(quat)
    vertical = axes[:, 2].abs()
    half_height = 0.5 * float(size[int(vertical.argmax())])
    found = []
    for k in range(3):
        if float(vertical[k]) > math.sin(AXIS_MAX_TILT_RAD):
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
    pos: torch.Tensor, heading: float, height_above_tips: float
) -> list[list[tuple[float, float]]]:
    """The two open-jaw landing spots as world-xy rectangles, at a height above the tips."""
    h = max(0.0, height_above_tips)
    thickness = TIP_THICKNESS + TIP_THICKNESS_PER_M * h
    half_face = TIP_HALF_FACE + TIP_HALF_FACE_PER_M * h
    c, s = math.cos(heading), math.sin(heading)
    boxes = []
    for side in (1.0, -1.0):
        corners = []
        for lx in (side * OPEN_HALF_GAP, side * (OPEN_HALF_GAP + thickness)):
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
    wall_footprints: list[list[tuple[float, float]]],
) -> tuple[float, int, float]:
    """``(clearance, swept, wall)`` for a grasp at ``pos`` closing along ``heading``.

    ``clearance`` is the horizontal distance from the landing tips to the nearest wall or
    standing neighbor, negative when a footprint overlaps a landing spot. A neighbor stands
    when its top reaches into the target's grip band, the top ``MIN_GRIP_OVERLAP`` of its
    side faces. ``swept`` counts the standing neighbors whose footprint crosses a closing
    stroke. ``wall`` is the wall-only clearance of the whole stroke, landing spot
    included, measured with the tip itself: the slanted hull above the tip slides off a wall
    edge, so only the tip's own footprint counts, but a tip that lands past a wall has to
    cross it on the way in.
    """
    bottom = float(pos[2]) - half_height
    band = float(pos[2]) + half_height - MIN_GRIP_OVERLAP
    strokes = sweep_footprints(pos, heading, width, OPEN_HALF_GAP + TIP_THICKNESS)
    wall = min((polygon_distance(t, w) for t in strokes for w in wall_footprints), default=math.inf)
    standing = torch.nonzero(other_pos[:, 2] + other_hz > band).flatten().tolist()
    clearance = wall
    swept = 0
    if standing:
        sweeps = sweep_footprints(pos, heading, width)
        for k in standing:
            reach = float(other_pos[k, 2] + other_hz[k]) - bottom
            spots = landing_footprints(pos, heading, reach)
            clearance = min(clearance, min(polygon_distance(t, other_footprints[k]) for t in spots))
            if any(polygon_distance(b, other_footprints[k]) < 0.0 for b in sweeps):
                swept += 1
    return clearance, swept, wall


def rank_grasps(
    sizes: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
    candidates: torch.Tensor,
    wall_footprints: list[list[tuple[float, float]]],
) -> list[Grasp]:
    """Every grasp of every candidate part with its clearance, best first.

    Other parts enter through ``neighbor_extents`` and ``footprint``; the walls come from
    ``tray_wall_obstacles``. The order is the height band (highest first), then the swept
    count, then the clearance.
    """
    _, hz = neighbor_extents(sizes, quat)
    prints = [footprint(sizes[b], pos[b], quat[b]) for b in range(sizes.shape[0])]
    ranked: list[Grasp] = []
    for b in torch.nonzero(candidates).flatten().tolist():
        others = torch.ones(sizes.shape[0], dtype=torch.bool)
        others[b] = False
        other_pos, other_hz = pos[others], hz[others]
        other_prints = [prints[k] for k in torch.nonzero(others).flatten().tolist()]
        for g in candidate_grasps(b, sizes[b], pos[b], quat[b]):
            clearance, swept, wall = jaw_clearance(
                pos[b],
                g.heading,
                g.width,
                g.half_height,
                other_pos,
                other_hz,
                other_prints,
                wall_footprints,
            )
            ranked.append(replace(g, clearance=clearance, swept=swept, wall=wall))
    ranked.sort(key=lambda g: (-math.floor(g.height / HEIGHT_BAND), g.swept, -g.clearance))
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
PUSH_WALL_MARGIN_M = TIP_HALF_FACE
"""The closed tips' center stays this far (m) inside the tray interior: the tip touches a
wall face at most, and the slanted hull above it slides off the wall edge (wall reads)."""
PUSH_MIN_GAIN_M = 0.005
"""A push must bring its part at least this much (m) closer to the tray center."""


def clamp_to_tray(
    point: tuple[float, float], margin: float = PUSH_WALL_MARGIN_M
) -> tuple[float, float]:
    """Pull a point back along its own radius until it is ``margin`` inside the rim."""
    cx, cy = layout.TRAY_CENTER
    dx, dy = point[0] - cx, point[1] - cy
    radius = math.hypot(dx, dy)
    limit = layout.TRAY_RADIUS - margin
    if radius <= limit or radius == 0.0:
        return point
    scale = limit / radius
    return (cx + dx * scale, cy + dy * scale)


def push_toward_center(
    part_xy: tuple[float, float],
    part_footprint: list[tuple[float, float]],
    blocked: list[list[tuple[float, float]]],
    half_length: float = PUSH_HALF_LENGTH_M,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """A sweep that moves a wall-side part toward the tray center, or None.

    The closed tips land behind the part (on its far side from the tray center), a
    ``PUSH_START_GAP_M`` off its footprint, clear of every footprint in ``blocked`` and
    ``PUSH_WALL_MARGIN_M`` inside the walls, then sweep through the part's center toward the
    tray center. When the straight-in approach has no room, the start swings to either side,
    up to 115 degrees, as long as the sweep still brings the part ``PUSH_MIN_GAIN_M`` closer
    to the tray center: a part in a corner is pushed out along one wall.
    """
    cx, cy = layout.TRAY_CENTER
    dx, dy = part_xy[0] - cx, part_xy[1] - cy
    radius = math.hypot(dx, dy)
    away = math.atan2(dy, dx) if radius > 1e-6 else 0.0
    for turn in (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0):
        heading = away + turn
        c, s = math.cos(heading), math.sin(heading)
        start = None
        for k in range(1, 30):
            point = (part_xy[0] + 0.002 * k * c, part_xy[1] + 0.002 * k * s)
            if clamp_to_tray(point) != point:
                break
            tip = tip_footprint(point)
            if polygon_distance(tip, part_footprint) < PUSH_START_GAP_M:
                continue
            if any(polygon_distance(tip, b) < 0.0 for b in blocked):
                continue
            start = point
            break
        if start is None:
            continue
        end = clamp_to_tray((part_xy[0] - half_length * c, part_xy[1] - half_length * s))
        if math.hypot(end[0] - cx, end[1] - cy) > radius - PUSH_MIN_GAIN_M:
            continue
        return start, end
    return None


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
    """Start and end of a straight sweep through ``center`` along ``heading``, inside the tray."""
    dx, dy = half_length * math.cos(heading), half_length * math.sin(heading)
    return clamp_to_tray((center[0] - dx, center[1] - dy)), clamp_to_tray(
        (center[0] + dx, center[1] + dy)
    )


# ponytail: the rim is modelled as a vertical stop at its xy shadow, which refuses some
# grasps the 45 degree slope would accept. Upgrade to a sloped stop if rung R2 reads a
# clearance loss at the rim.
def tray_wall_obstacles() -> list[list[tuple[float, float]]]:
    """The dish rim segments as world-xy footprints, for ``rank_grasps``.

    The rim leans outward, so its xy footprint is the leaning slab's shadow. The model still
    treats it as a stop that stands where the shadow is.
    """
    prints = []
    for _, center, size, quat in layout.tray_static_boxes():
        prints.append(
            footprint(
                torch.tensor(size, dtype=torch.float32),
                torch.tensor(center, dtype=torch.float32),
                torch.tensor(quat, dtype=torch.float32),
            )
        )
    return prints
