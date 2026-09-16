"""Grasp synthesis on part geometry for the heap teacher (MS6 phase A design 7.1).

Torch and stdlib only. A part pose gives three body axes in the world. Every axis near the
horizontal is a closing axis: the jaws close across the part's size along it, at that axis's
heading. The approach is from above through the runtime's near-level descend ladder, so no
tilt is chosen here. Clearance is measured with two virtual jaw boxes against the footprint
of every other part that stands taller than the finger tips, and against the tray walls.
Parts below the tips are the surface the descend stalls on, not obstacles. Grasps rank by
height, then clearance.
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
CLEARANCE_MIN = 0.003
"""Jaw-box clearance (m) a grasp needs to be selected."""
# The virtual jaw box, in the grasp frame: x along the closing axis, origin at the part
# center. The finger body is 30 mm along jaw travel. Its footprint is what a neighbor can foul.
JAW_BOX_THICKNESS = 0.030
JAW_BOX_HALF_FACE = 0.020
"""Half the jaw plate's face, perpendicular to the closing axis.

The vendored URDF's jaw meshes (``part_034_solid_034.stl``, ``part_041_solid_041.stl``)
measure at most 17.07 x 40.52 x 22.50 mm, so the plate's half extent is about 20 mm. The
MS6 design table's 0.0625 m modelled a 125 mm face, which no jaw on this arm has: against
the 130 mm tray interior it left only a 10 mm band of x pickable for a y-closing grasp.
Corrected on the owner's decision, 2026-09-16, with the measurement above as the authority.
"""
# How far under the pick point the finger tips reach at the shipped pick offset: 34.2 mm of
# tip reach against a 32.1 mm offset (robots.frames), rounded up. A neighbor whose top is under
# this depth is the surface the descend stalls on, not an obstacle the jaws have to clear.
TIP_BELOW_CENTER = 0.003
PUSH_CELL_M = 0.05
"""Grid cell of the failure memory and of the densest-cell push (design 7.2)."""
PUSH_HALF_LENGTH_M = 0.04
WALL_SPHERE_SPACING = 0.02


@dataclass(frozen=True)
class Grasp:
    part: int
    heading: float  # world heading of the closing axis, rad
    width: float  # part size along the closing axis, m
    half_height: float  # half the part size along its most vertical axis, m
    clearance: float  # m to the nearest neighbor surface, nan until ranked
    height: float  # part center z, m


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
    footprint, so a part's rotation about z never matters to the check.
    """
    rxy, hz = [], []
    for b in range(sizes.shape[0]):
        half = 0.5 * body_axes_world(quat[b]).abs().T @ sizes[b]
        rxy.append(math.hypot(float(half[0]), float(half[1])))
        hz.append(float(half[2]))
    return torch.tensor(rxy), torch.tensor(hz)


def jaw_clearance(
    pos: torch.Tensor,
    heading: float,
    width: float,
    other_pos: torch.Tensor,
    other_rxy: torch.Tensor,
    other_hz: torch.Tensor,
) -> float:
    """Horizontal distance (m) from the two jaw footprints to the nearest standing neighbor.

    A neighbor stands when its top is above the finger tips (``TIP_BELOW_CENTER`` under the
    part center). Negative when a footprint overlaps a jaw box. ``inf`` with no standing
    neighbor.
    """
    standing = other_pos[:, 2] + other_hz > float(pos[2]) - TIP_BELOW_CENTER
    if not bool(standing.any()):
        return math.inf
    c, s = math.cos(heading), math.sin(heading)
    rel = other_pos[standing] - pos
    radius = other_rxy[standing]
    local_x = c * rel[:, 0] + s * rel[:, 1]
    local_y = -s * rel[:, 0] + c * rel[:, 1]
    best = math.inf
    for side in (1.0, -1.0):
        dx = (local_x - side * (width / 2 + JAW_BOX_THICKNESS / 2)).abs() - JAW_BOX_THICKNESS / 2
        dy = local_y.abs() - JAW_BOX_HALF_FACE
        outside = torch.hypot(dx.clamp(min=0.0), dy.clamp(min=0.0))
        inside = torch.maximum(dx, dy).clamp(max=0.0)
        distance = torch.where((dx > 0) | (dy > 0), outside, inside) - radius
        best = min(best, float(distance.min()))
    return best


def rank_grasps(
    sizes: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
    candidates: torch.Tensor,
    extra_pos: torch.Tensor,
    extra_rxy: torch.Tensor,
    extra_hz: torch.Tensor,
) -> list[Grasp]:
    """Every grasp of every candidate part with its clearance, highest part first.

    Other parts enter through ``neighbor_extents``. The ``extra_*`` arrays are static
    obstacles, the tray walls from ``tray_wall_obstacles``.
    """
    rxy, hz = neighbor_extents(sizes, quat)
    ranked: list[Grasp] = []
    for b in torch.nonzero(candidates).flatten().tolist():
        others = torch.ones(sizes.shape[0], dtype=torch.bool)
        others[b] = False
        other_pos = torch.cat([pos[others], extra_pos])
        other_rxy = torch.cat([rxy[others], extra_rxy])
        other_hz = torch.cat([hz[others], extra_hz])
        for g in candidate_grasps(b, sizes[b], pos[b], quat[b]):
            clearance = jaw_clearance(pos[b], g.heading, g.width, other_pos, other_rxy, other_hz)
            ranked.append(replace(g, clearance=clearance))
    ranked.sort(key=lambda g: (-g.height, -g.clearance))
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


def push_segment(
    center: tuple[float, float], heading: float, half_length: float = PUSH_HALF_LENGTH_M
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Start and end of a straight sweep through ``center`` along ``heading``."""
    dx, dy = half_length * math.cos(heading), half_length * math.sin(heading)
    return ((center[0] - dx, center[1] - dy), (center[0] + dx, center[1] + dy))


def tray_wall_obstacles() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The tray walls as rows of discs, for ``rank_grasps``'s static obstacles."""
    rows = []
    for _, center, size in layout.tray_static_boxes():
        axis = 0 if size[0] >= size[1] else 1
        length = size[axis]
        count = math.ceil(length / WALL_SPHERE_SPACING) + 1
        for k in range(count):
            point = list(center)
            point[axis] = center[axis] - length / 2 + length * k / (count - 1)
            rows.append(point)
    pos = torch.tensor(rows, dtype=torch.float32)
    rxy = torch.full((pos.shape[0],), 0.5 * math.hypot(layout.TRAY_WALL, WALL_SPHERE_SPACING))
    return pos, rxy, torch.full((pos.shape[0],), layout.TRAY_WALL_HEIGHT / 2)
