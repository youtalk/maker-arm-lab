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
# Carried over from the OpenYAM LegoSort layout (openyam-lab M1 Task 10) and CONFIRMED
# UNCHANGED on this arm. Task D4 (`scripts/release_envelope_check.py`) ran on 2026-09-06:
# a driven pick-and-place into every bin at the nominal layout and at all four bin-jitter
# corners, gated on EXTEND convergence and on the brick landing in its bin, 40/40. No layout
# number had to move; the one constant that did was `core.oracle.RELEASE_TILT_RAD`, which is
# not a layout number (see its comment for the 117-160 mm reach shortfall that forced it).
#
# READ WHAT THAT DOES AND DOES NOT COVER, because the distinction decides who owns which
# number. D4 picks from ONE FIXED POSITION, (0.20, 0.0), deliberately: holding the pick leg
# constant is what makes the bins the only thing varying between corners. So D4 is the
# authority on the BIN and SHELF positions and on the whole release leg, and it is NOT an
# independent re-measurement of the scatter band -- SCATTER_MAX_RADIUS below still rests on
# Task D3's sweep alone.
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
# band area against the 2.86 % the old 0.315 cost, both by uniform sampling over the box.
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
#
# Task D4 (`scripts/release_envelope_check.py`) ran on 2026-09-06 and DID NOT re-derive this
# number, so do not read it as twice-confirmed. D4 is a release-envelope probe: it picks from
# one fixed position by design (see SCATTER_X above), so it exercises this radius at exactly
# one point. It did honour the warning this comment used to carry forward -- every gate it
# reads is a driven servo's residual, never a solver's report -- but a bound sampled at one
# azimuth is not a sweep. D3's own caveat therefore still stands unreduced: 0.280 is the
# largest radius that tracked clean at every azimuth MEASURED, which is not the largest that
# would track clean at every azimuth. Whoever re-measures it must drive real descends across
# the band, not query the solver.
SCATTER_MAX_RADIUS = 0.280

# BIN-CLEARANCE bound: the largest brick x at which the arm can descend to the desk with
# the tool axis on world -z WITHOUT its forearm fouling the bins' back walls.
#
# THIS IS NOT A REACH MEASUREMENT, and reading it as one is the mistake to avoid. It is
# derived, not measured: `top_down_x_limit(bl)` returns `bin_front_x(bl) - X_LIMIT_MARGIN`,
# so it is a function of where the BINS are, and this constant is just its value for
# DEFAULT_BIN_LAYOUT (0.44 - 0.045 - 0.005 - 0.10). Move the bins and it moves;
# test_default_bin_layout_reproduces_current_constants pins the identity and
# test_scatter_zone_tracks_jittered_bins pins the tracking (bin_x 0.43 -> 0.28, 0.46 -> 0.31).
#
# The arm's actual reach bound is SCATTER_MAX_RADIUS above, which is MEASURED and, at 0.280,
# is tighter than this at every azimuth. So as a filter this is currently inert -- every
# sampled brick is already inside 0.280, hence inside 0.29. It is kept for two reasons:
# it answers a different physical question (bin fouling, not tracking), and it is NOT
# uniformly slacker -- `top_down_x_limit` drops below 0.280 for any bin_x <= 0.43, at which
# point it binds again and the oracle's screen (core.oracle, which calls the per-layout
# function rather than this constant) starts doing work.
#
# Do not "correct" this to 0.280 to match the reach bound: that would break its own
# derivation from bin geometry and the tests that pin it, and would conflate two bounds that
# move independently.
#
# Carried over from the OpenYAM LegoSort layout (openyam-lab M1 Task 10) and still NOT
# re-measured on this arm. Task D4 (`scripts/release_envelope_check.py`) confirmed the bin
# positions this is derived FROM (2026-09-06, 40/40 driven trials across the jitter corners),
# which pins the value it evaluates to -- but D4 never descends over the scatter band, so the
# forearm-fouling question this constant answers has not itself been re-measured here. It
# remains inert at the default layout anyway, being slacker than SCATTER_MAX_RADIUS.
#
# THIS IS WHY SCATTER_X's UPPER BOUND IS 0.29. Do not widen one without the
# other -- test_scatter_zone_is_pickable ties them together.
TOP_DOWN_X_LIMIT = 0.29

# Shelf pedestal on the desk; bins sit on its top face.
#
# VALIDATED ON THIS ARM, unchanged, by Task D4 (`scripts/release_envelope_check.py`,
# 2026-09-06): with the bins here and on their top face, the oracle reaches and releases into
# every bin at every corner of the run-level jitter. The shelf height is load-bearing for
# that, not cosmetic -- it sets the bin rim at `_BIN_FLOOR_TOP_Z + BIN_INTERIOR[2]` = 0.195,
# which is the floor `core.oracle.RELEASE_Z` has to clear, and this arm's reach ceiling at
# the release attitude is not far above it. Raising the shelf raises that floor without
# raising the ceiling.
SHELF_SIZE = (0.10, 0.54, 0.15)
SHELF_POS = (0.44, 0.0, 0.075)

BIN_INTERIOR = (0.09, 0.09, 0.04)
BIN_WALL = 0.005
_BIN_Y = (-0.21, -0.07, 0.07, 0.21)
_BIN_X = 0.44
_BIN_FLOOR_TOP_Z = 0.155  # shelf top 0.15 + floor slab 0.005

X_LIMIT_MARGIN = 0.10
# Margin between the bins' robot-facing wall and TOP_DOWN_X_LIMIT. Carried over
# from the OpenYAM layout along with TOP_DOWN_X_LIMIT (see above), and inherited
# unchanged: Task D4 confirmed the bin positions but never descends over the
# scatter band, so it could not re-measure this margin. The limit moves with the
# bins.

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
