"""World layout for the LegoSort scene. Torch/stdlib only (CPU-testable).

Convention: arm base at the world origin, desk top surface at z = 0.
All lengths in meters. Real 2x4 LEGO brick: 31.8 x 15.8 x 9.6 mm.
"""

import itertools
import math
import random
from dataclasses import dataclass

import torch

BRICK_SIZE = (0.0318, 0.0158, 0.0096)
BRICK_MASS = 0.003  # kg; real brick ~2.3 g, slightly rounded up for solver health
BRICK_COLOR_ORDER = ("red", "blue", "yellow", "green")
BRICK_RGBA = {
    "red": (0.8, 0.1, 0.1, 1.0),
    "blue": (0.1, 0.2, 0.8, 1.0),
    "yellow": (0.9, 0.8, 0.1, 1.0),
    "green": (0.1, 0.7, 0.2, 1.0),
}

DESK_SIZE = (0.9, 0.9, 0.04)
DESK_POS = (0.30, 0.0, -0.02)  # top surface at z=0

# Brick scatter zone on the desk, in front of the arm.
#
# Carried over from the OpenYAM LegoSort layout (openyam-lab M1 Task 10). The
# Maker Arm's reach and elbow geometry differ; `scripts/release_envelope_check.py`
# (Task D4) re-validates every bin position and the scatter band on this arm and
# is the authority for changing these numbers.
SCATTER_X = (0.13, 0.29)
SCATTER_Y = (-0.18, 0.18)
SPAWN_Z = 0.02
MIN_SEPARATION = 0.06

# Reach clip: bricks never spawn further than this from the arm's base axis.
#
# THIS IS A TRACKING BOUND, NOT AN IK BOUND, and the difference is the whole story. The
# question is not "can the solver find a pose for this brick" -- it can, well past this
# radius -- but "does the arm actually ARRIVE there". Those two answers differ, and only the
# second one produces a usable episode.
#
# History, because the number moved a long way and the reason matters. Ruling D3-13 first set
# this to 0.315 from an IK-FEASIBILITY sweep: 35 brick positions x 6 jaw headings with a
# 64-restart confirmation, which found every fully-covered position inside radius 0.3199 m,
# a live gate trial's UNREACHABLE brick at 0.3257 m, and the fully-uncovered corner
# (0.290, +-0.180) at 0.3413 m solving 0 of 6 headings. That bound is real and still true --
# it is just not the binding one.
#
# Ruling D3-19 replaced it with the MEASURED TRACKABLE radius, after one trial failed
# identically in four consecutive gates (parts 10, 11, 13, 14): the arm stopped ~13 mm high
# on a brick at radius 0.3082 m, inside the old clip, with the IK reporting
# worst_pos_resid=0.00um and the path reporting continuous. Driving real descends on a grid
# of (azimuth, radius) -- /tmp/d3_radius.py, Task D3 part 15 -- gives the boundary:
#
#     azimuth     0.250  0.265  0.275  0.280  0.290  0.295  0.300  0.3082  0.315
#       -54 deg   clean   -- IK infeasible beyond here --
#       -40 deg   clean  clean         clean         clean
#       -25 deg   clean  clean         clean         clean
#     -19.8 deg                clean         clean         clean   FAIL    FAIL
#       -10 deg   clean  clean         clean         FAIL
#         0 deg          clean  clean         FAIL           FAIL    FAIL    FAIL
#        10 deg   clean  clean         clean         FAIL
#        25 deg   clean  clean         clean         clean
#        30 deg                                             clean   clean   IK infeasible
#        40 deg   clean  clean         clean         clean
#
# READ THAT SHAPE BEFORE CHANGING THIS NUMBER. The failures are a WEDGE around azimuth 0,
# not an annulus: the arm tracks to 0.3082 m at +30 deg and to at least 0.295 m at
# |azimuth| >= 25 deg, but gives out between 0.280 and 0.290 near azimuth 0. A radius bound
# cannot express a wedge, so this clip pays for the whole annulus to remove one -- 19.18 % of
# band area against the 2.87 % the old 0.315 cost, both by uniform sampling over the box.
# 0.280 is the largest radius every measured azimuth tracked to.
#
# The cheaper-looking 0.300 (7.48 %) was considered and REJECTED as fitted to the gate: it
# excludes the one failing trial and nothing else, leaving the measured failures at
# |azimuth| <= 10 deg inside the band for D5 and D7 to inherit. The extra cost buys the
# difference between a number fitted to the arm and one fitted to a test.
#
# Two causes were measured and EXCLUDED, so do not re-derive them: static torque (G(q) at the
# descend pose never exceeds ~52 % of the binding joint's motor rating anywhere in the band,
# including outside the old clip) and the joint trim (at trim_gain 0.0 the failures persist
# unchanged AND nothing saturates, so the saturation seen with the trim on was a consequence
# of it chasing an unreachable pose, not the cause). The wedge's mechanism is still unnamed;
# the likeliest shape of it is that global_ik's lean azimuth is pinned to 0.0, so the descend
# pose leans in a fixed WORLD direction regardless of where the brick is, which would make
# the geometry genuinely asymmetric in azimuth. That is untested.
#
# Applied as an unconditional invariant in `mdp.events._sample_brick_poses`, NOT as a
# preference like MIN_SEPARATION. A brick the arm cannot track is worse than a permanent
# failure floor: the success-only export discards the failed episode, so datagen spends time
# manufacturing waste AND biases what survives toward the easy part of the band.
#
# Does NOT affect the overhead camera. `overhead_view_region` frames SCATTER_X / SCATTER_Y,
# the declared box, not this radius -- so the frame still covers corners bricks no longer
# spawn in, which is correct and is why the framing tests are unchanged.
#
# Spec section 5 item 5 permits this ("the Maker Arm's reach ... may require moving the bins
# and the scatter band. The oracle state machine is not modified; only layout numbers are").
# `scripts/release_envelope_check.py` (Task D4) re-validates it with its own reach sweep, and
# should sweep TRACKING, not just IK, or it will re-derive the bound that was not binding.
SCATTER_MAX_RADIUS = 0.280

# Largest brick x at which the arm can complete a top-down pick -- i.e. descend
# to the desk with the tool axis on world -z.
#
# Carried over from the OpenYAM LegoSort layout (openyam-lab M1 Task 10). The
# Maker Arm's reach and elbow geometry differ; `scripts/release_envelope_check.py`
# (Task D4) re-validates every bin position and the scatter band on this arm and
# is the authority for changing these numbers.
#
# THIS IS WHY SCATTER_X's UPPER BOUND IS 0.29. Do not widen one without the
# other -- test_scatter_zone_is_pickable ties them together.
TOP_DOWN_X_LIMIT = 0.29

# Shelf pedestal on the desk; bins sit on its top face.
SHELF_SIZE = (0.10, 0.54, 0.15)
SHELF_POS = (0.44, 0.0, 0.075)

BIN_INTERIOR = (0.09, 0.09, 0.04)
BIN_WALL = 0.005
_BIN_Y = (-0.21, -0.07, 0.07, 0.21)
_BIN_X = 0.44
_BIN_FLOOR_TOP_Z = 0.155  # shelf top 0.15 + floor slab 0.005

X_LIMIT_MARGIN = 0.10
# Margin between the bins' robot-facing wall and TOP_DOWN_X_LIMIT. Carried over
# from the OpenYAM layout along with TOP_DOWN_X_LIMIT (see above); the same
# `scripts/release_envelope_check.py` sweep re-validates it on this arm. The
# limit moves with the bins.

_MIN_SLOT_GAP = 0.11  # outer bin width 0.10 + 10 mm clearance


@dataclass(frozen=True)
class BinLayout:
    """One generation run's bin geometry and color assignment.

    ``slot_colors[s]`` is the color index (into BRICK_COLOR_ORDER) whose bin
    occupies physical slot ``s`` (slots ordered by y, left to right).
    """

    bin_x: float = _BIN_X
    bin_ys: tuple[float, float, float, float] = _BIN_Y
    slot_colors: tuple[int, int, int, int] = (0, 1, 2, 3)


DEFAULT_BIN_LAYOUT = BinLayout()


def clip_to_reach(x: float, y: float, max_radius: float = SCATTER_MAX_RADIUS):
    """Pull a scatter point back along its own radius if it is outside the arm's reach.

    Lives here rather than in the sampler that calls it because `mdp.events` imports
    isaaclab and cannot be unit-tested; this is the invariant worth testing, so it goes in
    the isaaclab-free module, the same split `core.checker` documents for the gate metric.

    Radial rather than clamping x or y alone: the unreachable region is an annulus corner,
    so pulling along the radius is the smallest move that restores reachability, and it
    leaves the point's azimuth -- which is what the base yaw has to cover -- untouched.
    """
    radius = math.hypot(x, y)
    if radius <= max_radius or radius == 0.0:
        return x, y
    scale = max_radius / radius
    return x * scale, y * scale


def bin_front_x(bl: BinLayout) -> float:
    """x of the bins' robot-facing outer wall face."""
    return bl.bin_x - BIN_INTERIOR[0] / 2 - BIN_WALL


def top_down_x_limit(bl: BinLayout) -> float:
    return bin_front_x(bl) - X_LIMIT_MARGIN


def scatter_x(bl: BinLayout) -> tuple[float, float]:
    return (SCATTER_X[0], top_down_x_limit(bl))


def color_slot(bl: BinLayout, color_index: int) -> int:
    return bl.slot_colors.index(color_index)


def sample_bin_layout(rng: random.Random, slot_colors: tuple[int, int, int, int]) -> BinLayout:
    """Run-level draw: x in [-0.01, +0.02] around nominal, per-slot y +-0.02.

    Rejection-samples the y jitter until adjacent slot centers are >=
    _MIN_SLOT_GAP apart (outer bin width is 0.10 m, so anything closer risks
    wall contact).
    """
    bin_x = _BIN_X + rng.uniform(-0.01, 0.02)
    while True:
        bin_ys = tuple(y + rng.uniform(-0.02, 0.02) for y in _BIN_Y)
        if all(b - a >= _MIN_SLOT_GAP for a, b in itertools.pairwise(bin_ys)):
            return BinLayout(bin_x=bin_x, bin_ys=bin_ys, slot_colors=slot_colors)


def bin_interior_centers(bl: BinLayout | None = None) -> torch.Tensor:
    """(4,3) centers of each color's bin interior AABB, ordered as BRICK_COLOR_ORDER."""
    bl = bl or DEFAULT_BIN_LAYOUT
    z = _BIN_FLOOR_TOP_Z + BIN_INTERIOR[2] / 2
    return torch.tensor([[bl.bin_x, bl.bin_ys[color_slot(bl, c)], z] for c in range(4)])


def bin_half_extents() -> torch.Tensor:
    return torch.tensor([BIN_INTERIOR[0] / 2, BIN_INTERIOR[1] / 2, BIN_INTERIOR[2] / 2])


def overhead_view_region() -> tuple[tuple[float, float], tuple[float, float], float]:
    """Axis-aligned region a fixed overhead camera has to keep in frame.

    Returns ``((x_min, x_max), (y_min, y_max), z_top)`` -- the union of

    * the brick scatter zone, widened by a brick's worst-case xy half-diagonal
      because ``randomize_brick_poses`` randomises yaw over the full circle, so
      a brick centred on the zone boundary sticks out by that much; and
    * the four bins' OUTER boxes (interior half-extent plus one wall on each
      side), not their interiors.

    ``z_top`` is the tallest feature inside it: the bin rims.

    Lives here rather than in the camera cfg so it is testable without Isaac
    Lab -- the whole point of deriving the camera pose is that it follows when
    this file's constants move, and only a CPU test can guard that in CI.
    """
    half_diag = 0.5 * math.hypot(BRICK_SIZE[0], BRICK_SIZE[1])
    centers = bin_interior_centers()
    half = bin_half_extents()
    outer_x = float(half[0]) + BIN_WALL
    outer_y = float(half[1]) + BIN_WALL
    x_range = (
        min(SCATTER_X[0] - half_diag, float(centers[:, 0].min()) - outer_x),
        max(SCATTER_X[1] + half_diag, float(centers[:, 0].max()) + outer_x),
    )
    y_range = (
        min(SCATTER_Y[0] - half_diag, float(centers[:, 1].min()) - outer_y),
        max(SCATTER_Y[1] + half_diag, float(centers[:, 1].max()) + outer_y),
    )
    z_top = float((centers[:, 2] + half[2]).max())
    return x_range, y_range, z_top


def overhead_view_pose(fov_deg: float, margin: float = 1.15) -> tuple[float, float, float]:
    """World position of a straight-down camera framing :func:`overhead_view_region`.

    ``fov_deg`` is the square field of view (Isaac Lab renders square pixels
    only, so horizontal and vertical are equal); ``margin`` is the slack around
    the region.

    The height is solved at the region's ``z_top``, NOT at the desk. A
    downward-looking camera magnifies whatever is closest to it, so a height
    that just covers the region at z = 0 leaves the bin rims' outer corners
    outside the frame. This is the one non-obvious step in the derivation.
    """
    x_range, y_range, z_top = overhead_view_region()
    half_span = 0.5 * max(x_range[1] - x_range[0], y_range[1] - y_range[0]) * margin
    return (
        0.5 * (x_range[0] + x_range[1]),
        0.5 * (y_range[0] + y_range[1]),
        z_top + half_span / math.tan(math.radians(fov_deg) / 2.0),
    )


def bin_static_boxes(
    bl: BinLayout | None = None,
) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """(name, center, size) for 4 bins x (floor + 4 walls), as static colliders."""
    bl = bl or DEFAULT_BIN_LAYOUT
    boxes = []
    ix, iy, iz = BIN_INTERIOR
    for slot, y in enumerate(bl.bin_ys):
        color = BRICK_COLOR_ORDER[bl.slot_colors[slot]]
        floor_z = _BIN_FLOOR_TOP_Z - BIN_WALL / 2
        wall_z = _BIN_FLOOR_TOP_Z + iz / 2
        outer_x = ix + 2 * BIN_WALL
        boxes.append(
            (f"bin_{color}_floor", (bl.bin_x, y, floor_z), (outer_x, iy + 2 * BIN_WALL, BIN_WALL))
        )
        boxes.append(
            (
                f"bin_{color}_wall_front",
                (bl.bin_x + ix / 2 + BIN_WALL / 2, y, wall_z),
                (BIN_WALL, iy + 2 * BIN_WALL, iz),
            )
        )
        boxes.append(
            (
                f"bin_{color}_wall_back",
                (bl.bin_x - ix / 2 - BIN_WALL / 2, y, wall_z),
                (BIN_WALL, iy + 2 * BIN_WALL, iz),
            )
        )
        boxes.append(
            (
                f"bin_{color}_wall_left",
                (bl.bin_x, y - iy / 2 - BIN_WALL / 2, wall_z),
                (ix, BIN_WALL, iz),
            )
        )
        boxes.append(
            (
                f"bin_{color}_wall_right",
                (bl.bin_x, y + iy / 2 + BIN_WALL / 2, wall_z),
                (ix, BIN_WALL, iz),
            )
        )
    return boxes
