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
are other parts, unless the MS7a tray is on: its walls enter through ``wall_allows``, which
keeps a finger off a wall and, near one, the closing axis along it.
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
WALL_BAND_RAD = math.radians(20.0)
"""MS7a design, the tray: near a wall, a closing axis must run within this of the wall's
direction, so no finger lands between the part and the wall. # INITIAL: the tray teacher
ladder reads it."""
WALL_BAND_MARGIN = 0.005
"""How far (m) past a grasp's own finger reach a wall still bands its heading.

A grasp is near a wall when the part centre is closer to the wall than the outer face of a
finger pointed at it, ``landing_half_gap(width) + TIP_THICKNESS``, plus this margin. The
design wrote ``OPEN_HALF_GAP`` for that reach, the fully open jaw of MS6 phase A. On the
0.13 x 0.20 m tray that bands the headings of every part within 57 mm of two walls both
ways, half the tray, where no heading can then survive. # INITIAL"""


WALL_MAX_GRASP_WIDTH = 0.060
"""Along a wall the jaws may close end to end on a part up to this long (m), past
MAX_GRASP_WIDTH.

A 1x6 (47.8 mm) or 2x7 (55.8 mm) brick flush along a wall has no grasp under
MAX_GRASP_WIDTH, because its short axis points a finger into the wall, and no room behind it
for the rake's tips. Driven on 2026-09-23 (MS7a tray ladder), both flush against the left
wall at yaw 0 and 5 degrees closed end to end along the wall, lifted and stowed. The landing
gap of a 60 mm grasp is 42 mm, inside OPEN_HALF_GAP. # INITIAL"""


@dataclass(frozen=True)
class Wall:
    """A static wall the jaws must not land on: its top-down footprint, its direction of
    length (rad, a line, so ``heading`` and ``heading + pi`` are the same wall) and its top."""

    footprint: tuple[tuple[float, float], ...]
    heading: float
    top: float


def tray_walls() -> tuple[Wall, ...]:
    """The four walls of ``layout.tray_static_boxes`` for the grasp model."""
    walls = []
    for _, center, size in layout.tray_static_boxes():
        corners = [
            (center[0] + sx * size[0] / 2, center[1] + sy * size[1] / 2)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        ]
        heading = 0.0 if size[0] >= size[1] else math.pi / 2
        walls.append(Wall(tuple(convex_hull(corners)), heading, center[2] + size[2] / 2))
    return tuple(walls)


def wall_distance(xy: tuple[float, float], wall: Wall) -> float:
    """Distance (m) from a point to the wall's footprint, negative inside it."""
    return polygon_distance([xy], list(wall.footprint))


def along_wall(heading: float, wall: Wall) -> bool:
    """Is a closing axis at ``heading`` within WALL_BAND_RAD of the wall's direction?"""
    off = abs((heading - wall.heading + math.pi / 2) % math.pi - math.pi / 2)
    return off <= WALL_BAND_RAD


def hugs_wall(
    pos: torch.Tensor, heading: float, width: float, half_height: float, walls: Sequence[Wall]
) -> bool:
    """Is this grasp within its finger reach of a wall that stands in its grip band, with
    its closing axis along that wall (``wall_allows``'s band, from the passing side)?"""
    band = float(pos[2]) + half_height - MIN_GRIP_OVERLAP
    xy = (float(pos[0]), float(pos[1]))
    near = landing_half_gap(width) + TIP_THICKNESS + WALL_BAND_MARGIN
    return any(
        wall.top > band and wall_distance(xy, wall) < near and along_wall(heading, wall)
        for wall in walls
    )


def wall_allows(
    pos: torch.Tensor, heading: float, width: float, half_height: float, walls: Sequence[Wall]
) -> bool:
    """Can the jaws close along ``heading`` on a part at ``pos`` beside these walls?

    A wall counts only where it stands in the part's grip band, the same rule a neighbor
    obeys in ``jaw_clearance``: a part resting higher than the wall top is gripped above it.
    Two tests then, per standing wall:

    * the band: within its finger reach of the wall (``WALL_BAND_MARGIN``), the closing axis
      runs along the wall (``along_wall``);
    * the stroke: each finger's path from the part face out to the tip's OUTER face, the
      tip's own 8 mm face across, stays off the wall. Only the tip counts, not the hull above
      it: a tip that lands inside the wall face picks with the hull resting on the wall top,
      one whose outer face lands past it does not (the MS6 wall reads of 2026-09-16,
      docs/ms6-teacher-inspection.md). So a wall is a neighbor of unbounded length that the
      finger clears by its thickness.
    """
    band = float(pos[2]) + half_height - MIN_GRIP_OVERLAP
    xy = (float(pos[0]), float(pos[1]))
    reach = landing_half_gap(width) + TIP_THICKNESS
    strokes = sweep_footprints(pos, heading, width, reach)
    for wall in walls:
        if wall.top <= band:
            continue
        if wall_distance(xy, wall) < reach + WALL_BAND_MARGIN and not along_wall(heading, wall):
            return False
        if any(polygon_distance(s, list(wall.footprint)) < CLEARANCE_MIN for s in strokes):
            return False
    return True


@dataclass(frozen=True)
class Grasp:
    part: int
    heading: float  # world heading of the CLOSING axis (the jaw travel), rad
    width: float  # part size along the closing axis, m
    half_height: float  # half the part size along its most vertical axis, m
    clearance: float  # m from the landing tips to the nearest standing neighbor
    height: float  # part center z, m
    swept: int = 0  # standing neighbors the closing stroke pushes toward the part


CLOSE_SQUEEZE_M = 0.003
"""How far past the part's half width the jaws may still close (m).

The contact model penetrates, and by a varying amount: a held grip settles 2.0 to 4.25 mm
inside the part's half width across the catalog, so this is a floor on the RUNAWAY, not a
model of the grip. Swept on the lone-part wall table, counting the rows whose grip survived
the lift: 2.0 mm holds 10 of 12 and cuts off plate_1x4, whose grip needs 2.6 mm; 3.0 mm
holds 11 of 12; 4.0 mm holds 10 of 12 and 5.0 mm only 8, both starving the wider grasps.
Read the FORCE in that table and not its `pick` column, which also counts the parts that
were gripped and then lost on the way to the bin.

For a narrow grasp the floor falls near zero and the jaws behave as they always did, which
is what the 7.8 mm rows want; it binds on the wide thin parts, which is where the extrusion
was.
"""


def landing_half_gap(width: float) -> float:
    """The jaw inner face's distance (m) from the tool axis while the tips come down.

    Never more than ``OPEN_HALF_GAP``: a part too wide for the jaws to straddle asks for the
    full opening, which is all the jaws have.
    """
    return min(OPEN_HALF_GAP, 0.5 * width + JAW_APPROACH_CLEARANCE + JAW_HOLD_BAND)


def close_gap(width: float) -> float:
    """The teacher's close set point (m): the jaw gap the carrying legs hold on a grasp of
    ``width``, its half width less ``CLOSE_SQUEEZE_M`` on each side and never below zero
    (``OracleRig._matched_close_command``). The continuous corpus records exactly this gap
    at every CLOSE, so a policy-side floor at the same value leaves recorded frames alone."""
    return 2.0 * max(0.0, 0.5 * width - CLOSE_SQUEEZE_M)


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
    walls: Sequence[Wall] = (),
) -> list[Grasp]:
    """Every closing axis within AXIS_MAX_TILT_RAD of the jaw plane, under MAX_GRASP_WIDTH,
    that the ``walls`` allow (``wall_allows``; none on the bare desk). Beside a wall an axis
    along it may reach WALL_MAX_GRASP_WIDTH (``hugs_wall``).

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
        wide = width > MAX_GRASP_WIDTH
        if wide and not (walls and width <= WALL_MAX_GRASP_WIDTH):
            continue
        heading = math.atan2(float(axes[k, 1]), float(axes[k, 0]))
        if walls and not wall_allows(pos, heading, width, half_height, walls):
            continue
        if wide and not hugs_wall(pos, heading, width, half_height, walls):
            continue
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
    walls: Sequence[Wall] = (),
) -> list[Grasp]:
    """Every grasp of every candidate part with its clearance, best first.

    Other parts enter through ``neighbor_extents`` and ``footprint``. ``closing_axes`` is
    one axis tuple per part (``parts.Part.closing_axes``); None lets every part offer all
    three. A grasp the ``walls`` refuse never ranks, so no forced pick lands a finger on a
    wall. The order is the height band (highest first), then the swept count, then the
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
        for g in candidate_grasps(b, sizes[b], pos[b], quat[b], tilt_rad, axes_b, walls):
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


RAKE_BAND_M = 0.03
"""MS7a design, the rake: a candidate with no grasp whose centre lies this close (m) to a
wall is raked off it. # INITIAL: the tray teacher ladder reads it."""
RAKE_START_GAP_M = 0.002
"""Gap (m) between the raked part's footprint and the closed tips where they land."""
RAKE_WEDGE_GAP_M = 0.0005
"""Gap (m) the wedged-in tips keep to the wall face (``rake_segment``'s last resort)."""
RAKE_SWINGS_RAD = (0.0, 0.5, -0.5, 1.0, -1.0)
"""Sweep directions tried, off the wall's normal, when the straight one has no room behind
the part: a swung sweep catches the part's end and turns it off the wall (the wall push of
MS6 phase A swung the same way, git f93e010^:core/grasp.py)."""


def rake_footprint(point: tuple[float, float], height: float = 0.0) -> list[tuple[float, float]]:
    """The closed tips' footprint at ``height`` above them, as a square that holds them at
    any jaw heading.

    The two closed tips are 4 mm along the closing axis and 8 mm across at the tips, and
    the hull widens above them (TIP_THICKNESS_PER_M per tip, TIP_HALF_FACE_PER_M per side).
    The rake tries more than one jaw heading (``HeapOracle.report``), so its footprint is
    the square of the larger half extent. At the 20 mm tray wall top that is 11.2 mm: tips
    that land closer to a wall rest the hull on the wall top and stop 20 mm up, over the
    parts (driven on 2026-09-23 at the near wall: the pick point stalled at 21.8 mm).
    """
    half = max(
        TIP_THICKNESS + TIP_THICKNESS_PER_M * height,
        TIP_HALF_FACE + TIP_HALF_FACE_PER_M * height,
    )
    return convex_hull(
        [(point[0] + sx * half, point[1] + sy * half) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
    )


def _clear_of_walls(point: tuple[float, float], walls: Sequence[Wall], gap: float = 0.0) -> bool:
    """Does the rake's hull, at each wall's top, clear that wall by ``gap``?"""
    return all(
        polygon_distance(rake_footprint(point, w.top), list(w.footprint)) >= gap for w in walls
    )


def rake_segment(
    part_xy: tuple[float, float],
    part_print: list[tuple[float, float]],
    length: float,
    wall: Wall,
    walls: Sequence[Wall],
    blocked: Sequence[list[tuple[float, float]]],
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    """``(start, end, heading)`` of a rake that sweeps a part off ``wall``, or None.

    The closed tips land behind the part, on the wall side, ``RAKE_START_GAP_M`` off its
    footprint, with their hull clear of every wall at its top (``rake_footprint``) and
    their tips clear of every footprint in ``blocked``, and sweep along ``heading`` through
    the part's centre to ``length`` past it, away from the wall. The straight sweep runs
    along the wall's normal; with no room behind the part the sweep swings off it
    (``RAKE_SWINGS_RAD``), and of the sweeps that fit the one that ends farthest from every
    wall wins. With no room on any swing, the tips wedge in: they land on the
    straight line at the wall face, on the part's wall-side edge. Their hull then rests on
    the wall top, and the sweep comes down onto the part on its way off the wall: driven on
    2026-09-23, that moved a flush 1x3 slope 5 and 8 mm in two rakes, and freed a 1x2 slope
    lying at 30 degrees, which then stowed. The end is pulled back until the hull clears
    every wall, and both ends stay in reach.
    """
    cx = sum(x for x, _ in wall.footprint) / len(wall.footprint)
    cy = sum(y for _, y in wall.footprint) / len(wall.footprint)
    nx, ny = -math.sin(wall.heading), math.cos(wall.heading)
    if (part_xy[0] - cx) * nx + (part_xy[1] - cy) * ny < 0.0:
        nx, ny = -nx, -ny
    normal = math.atan2(ny, nx)
    found = []
    for swing in RAKE_SWINGS_RAD:
        heading = normal + swing
        c, s = math.cos(heading), math.sin(heading)
        start = None
        for k in range(1, 61):  # back off from the centre, 1 mm at a time
            point = (part_xy[0] - 0.001 * k * c, part_xy[1] - 0.001 * k * s)
            if polygon_distance(rake_footprint(point), part_print) >= RAKE_START_GAP_M:
                start = point
                break
        if start is None or clamp_to_reach(start) != start or not _clear_of_walls(start, walls):
            continue
        if any(polygon_distance(rake_footprint(start), b) < 0.0 for b in blocked):
            continue
        end = _rake_end(part_xy, heading, length, walls)
        if end is not None:
            found.append((start, end, heading))
    if found:
        # The swing that leaves the part farthest from every wall, the straight one at a tie:
        # a swing toward a corner only hands the part to the next wall.
        return max(found, key=lambda f: min(wall_distance(f[1], w) for w in walls))
    c, s = math.cos(normal), math.sin(normal)
    start = None
    for k in range(-10, 61):  # from 10 mm inside the centre toward the wall
        point = (part_xy[0] - 0.001 * k * c, part_xy[1] - 0.001 * k * s)
        tips = rake_footprint(point)
        if any(polygon_distance(tips, list(w.footprint)) < RAKE_WEDGE_GAP_M for w in walls):
            break
        start = point
    if start is None or clamp_to_reach(start) != start:
        return None
    if any(polygon_distance(rake_footprint(start), b) < 0.0 for b in blocked):
        return None
    end = _rake_end(part_xy, normal, length, walls)
    return None if end is None else (start, end, normal)


def _rake_end(
    part_xy: tuple[float, float], heading: float, length: float, walls: Sequence[Wall]
) -> tuple[float, float] | None:
    """``length`` past the part's centre along ``heading``, pulled back until the hull clears
    every wall, and in reach; None once that is under half of ``length``."""
    c, s = math.cos(heading), math.sin(heading)
    reach = length
    while reach >= 0.5 * length:
        point = clamp_to_reach((part_xy[0] + reach * c, part_xy[1] + reach * s))
        if _clear_of_walls(point, walls):
            return point
        reach -= 0.002
    return None
