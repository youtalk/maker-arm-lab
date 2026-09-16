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
from dataclasses import dataclass

import torch

from maker_arm_lab.core.parts import MAX_GRASP_WIDTH
from maker_arm_lab.core.quat import quat_apply

AXIS_MAX_TILT_RAD = math.radians(15.0)
"""A body axis this far from horizontal still counts as a closing axis."""
CLEARANCE_MIN = 0.003
"""Jaw-box clearance (m) a grasp needs to be selected."""
# The virtual jaw box, in the grasp frame: x along the closing axis, origin at the part
# center. The finger body is 30 mm along jaw travel and reaches 62.5 mm across the tool axis
# (the collision box robots/frames.py quotes). Its footprint is what a neighbor can foul.
JAW_BOX_THICKNESS = 0.030
JAW_BOX_HALF_FACE = 0.0625
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
