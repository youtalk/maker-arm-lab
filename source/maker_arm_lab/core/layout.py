"""World layout for the LegoSort scene. Torch/stdlib only (CPU-testable).

Convention: arm base at the world origin, desk top surface at z = 0.
All lengths in meters. Real 2x4 LEGO brick: 31.8 x 15.8 x 9.6 mm.
"""

import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from maker_arm_lab.core.quat import quat_from_rpy_deg, quat_mul

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

# The heap drop zone. Owner decision 2026-09-18: no dish and no box, and the parts spread
# over the whole six-brick scatter zone, SCATTER_X by SCATTER_Y clipped to the driven reach
# band, so the desk to the arm's left and right is used. That zone is what Task D3 drove and
# what the overhead camera frames (`overhead_view_region`). Before the spread the heap was
# dumped on a disc of radius 0.088 m at (0.19, 0.0), 243 cm2 against the zone's ~470 cm2, and
# 50 parts there blocked each other's landing spots (~/data/maker-arm-ms6/i/rA). Nothing
# holds the parts on the desk: a part that slides out of the band is out of reach and stays
# where it is (`core.heap_oracle`).
HEAP_CENTER = (0.21, 0.0)
"""The zone's centre: where the push heading's bias points (`HeapOracle._push_heading`)."""

# MS7a design, the tray: the H1 tray returns (MS6 design H1, git c15f8f8^). The interior is
# 0.13 x 0.20 m at x 0.13 to 0.26 and y -0.10 to 0.10, inside the measured reach, with four
# static walls TRAY_WALL thick and TRAY_WALL_HEIGHT high. The MS7 desk boxes keep their
# footprints outside it. The tray itself is switched on per env; without it the heap drops on
# the bare desk as in MS6.
TRAY_CENTER = (0.195, 0.0)
TRAY_INTERIOR = (0.13, 0.20)
TRAY_WALL_HEIGHT = 0.02
TRAY_WALL = 0.005


TRAY_DROP_INSET = 0.005
"""Inset (m) of the tray drop over a part's whole half diagonal, so no part starts over a wall
at any orientation. # INITIAL: the MS7a tray ladder reads the spill over the walls."""


def tray_bounds() -> tuple[tuple[float, float], tuple[float, float]]:
    """((x_min, x_max), (y_min, y_max)) of the tray interior."""
    cx, cy = TRAY_CENTER
    hx, hy = TRAY_INTERIOR[0] / 2, TRAY_INTERIOR[1] / 2
    return ((cx - hx, cx + hx), (cy - hy, cy + hy))


def tray_static_boxes() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """(name, center, size) for the four tray walls, as static colliders on the desk.

    The near and far walls (at -x and +x) run the full outer length, so they close the corners.
    """
    (x0, x1), (y0, y1) = tray_bounds()
    z = TRAY_WALL_HEIGHT / 2
    w = TRAY_WALL
    long_y = TRAY_INTERIOR[1] + 2 * w
    return [
        ("tray_wall_near", (x0 - w / 2, TRAY_CENTER[1], z), (w, long_y, TRAY_WALL_HEIGHT)),
        ("tray_wall_far", (x1 + w / 2, TRAY_CENTER[1], z), (w, long_y, TRAY_WALL_HEIGHT)),
        (
            "tray_wall_left",
            (TRAY_CENTER[0], y0 - w / 2, z),
            (TRAY_INTERIOR[0], w, TRAY_WALL_HEIGHT),
        ),
        (
            "tray_wall_right",
            (TRAY_CENTER[0], y1 + w / 2, z),
            (TRAY_INTERIOR[0], w, TRAY_WALL_HEIGHT),
        ),
    ]


TRAY_WALL_OUTWARD_DEG = {"near": 180.0, "far": 0.0, "left": -90.0, "right": 90.0}
"""The world heading each tray wall faces outward, degrees (``tray_lean_boxes``)."""


def tray_lean_boxes(
    lean_deg: float,
) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], tuple]]:
    """(name, center, size, quat) of the four tray walls leaning outward by ``lean_deg``.

    Each wall is a slab TRAY_WALL thick whose inner bottom edge lies on the interior edge on
    the desk, tilted outward about that edge so its rim stands TRAY_WALL_HEIGHT high, as the
    MS6 round dish's rim leaned (git c15f8f8). The interior footprint at the desk is the H1
    one; at the rim it is ``TRAY_WALL_HEIGHT * tan(lean)`` wider on every side. Each slab runs
    long enough past the corners to meet its neighbours at the rim. ``lean_deg`` 0 builds
    vertical slabs; the scene builds ``tray_static_boxes`` then instead.
    """
    tilt = math.radians(lean_deg)
    span = TRAY_WALL_HEIGHT / math.cos(tilt)
    out = 0.5 * span * math.sin(tilt) + 0.5 * TRAY_WALL * math.cos(tilt)
    z = 0.5 * span * math.cos(tilt) - 0.5 * TRAY_WALL * math.sin(tilt)
    grow = TRAY_WALL_HEIGHT * math.tan(tilt) + TRAY_WALL
    (x0, x1), (y0, y1) = tray_bounds()
    cx, cy = TRAY_CENTER
    edges = {"near": (x0, cy), "far": (x1, cy), "left": (cx, y0), "right": (cx, y1)}
    sides = {"near": y1 - y0, "far": y1 - y0, "left": x1 - x0, "right": x1 - x0}
    boxes = []
    for name, heading in TRAY_WALL_OUTWARD_DEG.items():
        ux, uy = math.cos(math.radians(heading)), math.sin(math.radians(heading))
        ex, ey = edges[name]
        quat = quat_mul(
            quat_from_rpy_deg((0.0, 0.0, heading)), quat_from_rpy_deg((0.0, lean_deg, 0.0))
        )
        size = (TRAY_WALL, sides[name] + 2.0 * grow, span)
        boxes.append((f"tray_wall_{name}", (ex + out * ux, ey + out * uy, z), size, quat))
    return boxes


def sample_tray_point(rng: random.Random, inset: float) -> tuple[float, float]:
    """A drop point uniform over the tray interior shrunk by ``inset`` on every edge.

    An inset past half the interior collapses that axis onto the tray's center line.
    """
    (x0, x1), (y0, y1) = tray_bounds()
    ix = min(inset, (x1 - x0) / 2)
    iy = min(inset, (y1 - y0) / 2)
    return rng.uniform(x0 + ix, x1 - ix), rng.uniform(y0 + iy, y1 - iy)


HEAP_RADIUS = 0.088
"""The radius over which that bias narrows to its floor: the old drop disc's."""
HEAP_DROP_INSET = 0.02
"""Extra inset (m) of the drop over a part's half diagonal, so a bounce stays in reach.

0.01 left 6 of 150 parts just past the reach radius before a single pick, at 0.280 to
0.291 m (rung b3 of the bare-desk ladder, ~/data/maker-arm-ms6/j)."""
SCATTER_MIN_RADIUS = 0.10
"""Nearest radius from the base a driven descend picked at: campaign d's reach band probe
(docs/ms6-lever-verification.md), `scripts/mj_probe_bottom_pick.py --mode reach_band`. Like
SCATTER_MAX_RADIUS it is a TRACKING bound read on a lone brick, not a solver report."""
REACH_Y_SLACK = 0.02
"""How far (m) past SCATTER_Y a part is still a target: the overhead camera frames the zone
(`overhead_view_region`) and a part further out to the side is not in the frame."""
REACH_MAX_AZIMUTH_RAD = math.atan2(SCATTER_Y[1], SCATTER_X[0]) + math.radians(5.0)
"""The widest azimuth from the base's +x a part is still a target: the zone's near corner,
54 degrees, plus a little. D3 and campaign d drove the zone's azimuths and no wider. Without
this cut the annulus admitted the half-plane behind the arm (branch review, 2026-09-18)."""
REACH_WEDGE_AZIMUTH_RAD = math.radians(25.0)
REACH_WEDGE_RADIUS = 0.295
"""The driven reach is a wedge, not a circle (SCATTER_MAX_RADIUS's own table, and campaign
d's reach band probe in docs/ms6-lever-verification.md). Near azimuth zero the descend stops
short between 0.280 and 0.290 m, so SCATTER_MAX_RADIUS stays the drop's bound and the
candidate bound there. At 25 degrees the D3 grid tracked clean at 0.295 m and at 30 degrees
at 0.308 m; campaign d stowed the part at 0.30 and 0.32 m at plus and minus 40 degrees. A
part that ends between 0.280 and 0.295 m at 25 degrees or more is therefore still a target.
The drop never places a part there: this only keeps a nudged part in play."""

# The heap drop (design section 6): part k lands at a height uniform in
# [HEAP_DROP_Z_MIN, HEAP_DROP_Z_MIN + HEAP_DROP_Z_PER_PART * K]. The parts are released in
# batches of HEAP_DROP_BATCH, HEAP_DROP_BATCH_STEPS of physics apart, each batch's points
# HEAP_DROP_GAP clear of one another (`sample_heap_batch`), so a later batch lands on a
# settled one and a pile forms instead of an explosion. Then the physics settles for
# HEAP_SETTLE_STEPS steps at 240 Hz (2 s) before the episode starts.
HEAP_DROP_Z_MIN = 0.03
HEAP_DROP_Z_PER_PART = 0.0012
HEAP_DROP_BATCH = 5
HEAP_DROP_BATCH_STEPS = 60
HEAP_DROP_GAP = 0.005
HEAP_DROP_TRIES = 500
HEAP_SETTLE_STEPS = 480


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


def in_reach(x: float, y: float, bl: BinLayout, margin: float = 0.0) -> bool:
    """Can a top-down descend pick at (x, y)? The driven reach band, short of the bins.

    The band is the annulus SCATTER_MIN_RADIUS to SCATTER_MAX_RADIUS, widened to
    REACH_WEDGE_RADIUS from REACH_WEDGE_AZIMUTH_RAD outward, within REACH_MAX_AZIMUTH_RAD of
    +x, cut at the top-down x limit and at the scatter zone's y span plus REACH_Y_SLACK.
    ``margin`` shrinks it on every edge (negative grows it), for a caller that wants
    hysteresis; the azimuth cut does not move with it.
    """
    radius = math.hypot(x, y)
    azimuth = abs(math.atan2(y, x))
    outer = REACH_WEDGE_RADIUS if azimuth >= REACH_WEDGE_AZIMUTH_RAD else SCATTER_MAX_RADIUS
    return (
        SCATTER_MIN_RADIUS + margin <= radius <= outer - margin
        and azimuth <= REACH_MAX_AZIMUTH_RAD
        and x <= top_down_x_limit(bl) - margin
        and abs(y) <= SCATTER_Y[1] + REACH_Y_SLACK - margin
    )


def sample_heap_point(
    rng: random.Random, bl: BinLayout, inset: float, zone=None
) -> tuple[float, float]:
    """A drop point uniform over the scatter zone shrunk by ``inset`` on every edge.

    The zone is ``scatter_x(bl)`` by SCATTER_Y with the far corners clipped at
    SCATTER_MAX_RADIUS, the same region the six-brick task scatters over. Rejection on the
    radius keeps the draw uniform per unit area instead of piling parts on the arc.
    ``zone``, ``((x0, x1), (y0, y1))``, replaces the scatter zone: the tray footprint when
    desk boxes stand beside it (MS7a design, desk boxes).
    """
    (x0, x1), (y0, y1) = zone or (scatter_x(bl), SCATTER_Y)
    while True:
        x, y = rng.uniform(x0 + inset, x1 - inset), rng.uniform(y0 + inset, y1 - inset)
        if math.hypot(x, y) <= SCATTER_MAX_RADIUS - inset:
            return x, y


def sample_heap_batch(
    rng: random.Random,
    bl: BinLayout,
    half_diagonals: Sequence[float],
    tries: int = HEAP_DROP_TRIES,
    zone=None,
    tray: bool = False,
) -> list[tuple[float, float]]:
    """Drop points for parts released into the air together, one per half diagonal.

    Two points are at least the two half diagonals plus HEAP_DROP_GAP apart, so no two parts
    of the batch start inside each other. Writing 50 parts into the air at once interpenetrated
    them and the solver flung them apart: 8 of 150 settled out of reach, up to 80 mm out
    (branch review, 2026-09-18). After ``tries`` rejections the last draw is taken as it is,
    the same escape ``sample_brick_poses`` uses, so a crowded batch cannot loop forever.

    The zone's edge is inset by HEAP_DROP_INSET plus a quarter of the half diagonal, not the
    whole: the edge is a reach bound on the part's CENTRE, not a wall, and the whole diagonal
    left a 4x8 brick a sliver 5 cm wide to land in, so two of them could not keep apart. The
    largest parts are placed first, so the small ones fill in around them. Measured over 300
    batches of HEAP_DROP_BATCH catalog parts: no pair overlaps. ``zone`` is
    ``sample_heap_point``'s.

    ``tray`` (MS7a design, the tray) draws inside the tray interior instead, inset by the
    whole half diagonal plus TRAY_DROP_INSET: there the edge is a wall, not a reach bound. It
    takes the place of ``zone``.
    """
    order = sorted(range(len(half_diagonals)), key=lambda k: -half_diagonals[k])
    points: dict[int, tuple[float, float]] = {}
    for k in order:
        hd = half_diagonals[k]
        for attempt in range(tries):
            if tray:
                x, y = sample_tray_point(rng, hd + TRAY_DROP_INSET)
            else:
                x, y = sample_heap_point(rng, bl, 0.25 * hd + HEAP_DROP_INSET, zone)
            clear = all(
                math.dist((x, y), q) >= hd + half_diagonals[j] + HEAP_DROP_GAP
                for j, q in points.items()
            )
            if clear or attempt == tries - 1:
                points[k] = (x, y)
                break
    return [points[k] for k in range(len(half_diagonals))]


def scatter_x(bl: BinLayout) -> tuple[float, float]:
    return (SCATTER_X[0], top_down_x_limit(bl))


def color_slot(bl: BinLayout, color_index: int) -> int:
    return bl.slot_colors.index(color_index)


def sample_bin_layout(
    rng: random.Random, slot_colors: tuple[int, int, int, int], spatial_scale: float = 1.0
) -> BinLayout:
    """Run-level draw: x in [-0.01, +0.02] around nominal, per-slot y +-0.02.

    ``spatial_scale`` multiplies the x range (the generalization ladder of the MS4 roadmap
    reads camera and bin_x shifts outside the training range). The y range is NOT scaled:
    the rejection sampling below keeps adjacent slot centers >= _MIN_SLOT_GAP apart (outer
    bin width is 0.10 m, so anything closer risks wall contact), and a wider y band would
    mostly be rejected.
    """
    bin_x = _BIN_X + rng.uniform(-0.01, 0.02) * spatial_scale
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


# MS6 rework section 6. Heap mode drops 50 parts into ONE bin, and the four-bin footprint
# overflows: 0.09 by 0.09 by 0.04 m is 324 cm3 against 159 cm3 of solid parts, which piles
# over the rim. This box grows along y only. bin_front_x reads BIN_INTERIOR[0] and
# top_down_x_limit sits 0.10 m in front of it, so a wider x would pull the pick limit under
# the drop disc's far edge.
HEAP_BIN_INTERIOR = (0.09, 0.30, 0.06)
HEAP_BIN_BACKSTOP = 0.18
"""Height (m) of the heap bin's far wall, the one at +x, above the bin floor.

RELEASE holds the tool 80 degrees over and opens the jaws in a near-vertical plane, so the
part slides off the lower plate along the tool axis, toward +x, and leaves it at up to about
0.9 m/s. From RELEASE_Z that carries it 5 to 15 cm past the release point. On rung b3 of the
bare-desk ladder 6 of about 124 carried parts ended on the desk at x 0.50 to 0.69, past the
0.06 m far wall (~/data/maker-arm-ms6/j). This wall rises to 0.335 m, above RELEASE_Z, so the
slide hits it and drops into the bin. The three other walls keep HEAP_BIN_INTERIOR[2], and
the grader's z band (STOW_HALF_EXTENTS, twice the interior height) does not move."""


def heap_bin_center(bl: BinLayout | None = None) -> tuple[float, float, float]:
    """Centre of the single heap bin's interior."""
    bl = bl or DEFAULT_BIN_LAYOUT
    return (bl.bin_x, 0.0, _BIN_FLOOR_TOP_Z + HEAP_BIN_INTERIOR[2] / 2)


def heap_bin_static_boxes(
    bl: BinLayout | None = None,
) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """(name, center, size) for the one heap bin: floor plus four walls.

    The front wall, at +x, is the HEAP_BIN_BACKSTOP; the other three are the interior height.
    """
    bl = bl or DEFAULT_BIN_LAYOUT
    ix, iy, iz = HEAP_BIN_INTERIOR
    floor_z = _BIN_FLOOR_TOP_Z - BIN_WALL / 2
    wall_z = _BIN_FLOOR_TOP_Z + iz / 2
    outer_x, outer_y = ix + 2 * BIN_WALL, iy + 2 * BIN_WALL
    return [
        ("heap_bin_floor", (bl.bin_x, 0.0, floor_z), (outer_x, outer_y, BIN_WALL)),
        (
            "heap_bin_wall_front",
            (bl.bin_x + ix / 2 + BIN_WALL / 2, 0.0, _BIN_FLOOR_TOP_Z + HEAP_BIN_BACKSTOP / 2),
            (BIN_WALL, outer_y, HEAP_BIN_BACKSTOP),
        ),
        (
            "heap_bin_wall_back",
            (bl.bin_x - ix / 2 - BIN_WALL / 2, 0.0, wall_z),
            (BIN_WALL, outer_y, iz),
        ),
        ("heap_bin_wall_left", (bl.bin_x, -iy / 2 - BIN_WALL / 2, wall_z), (ix, BIN_WALL, iz)),
        ("heap_bin_wall_right", (bl.bin_x, iy / 2 + BIN_WALL / 2, wall_z), (ix, BIN_WALL, iz)),
    ]


def sample_brick_poses(
    num_bricks: int,
    min_separation: float,
    pose_range: dict[str, tuple[float, float]],
    max_sample_tries: int = 5000,
    max_radius: float | None = None,
) -> list[list[float]]:
    """Rejection-sample (x, y, z, roll, pitch, yaw) per brick, min_separation apart in xyz.

    ``max_radius`` (``SCATTER_MAX_RADIUS``) clips the scatter box's unreachable
    corners. It is an INVARIANT, not a preference: unlike ``min_separation``, which the
    last-try escape below is allowed to violate rather than loop forever, a brick outside
    it is never emitted. Task D3 measured that the grasp task has no solution at all beyond
    that radius, at any jaw heading, so such a brick is a permanent failure floor in every
    dataset generated from the scene rather than a merely awkward draw. On the final try the
    sample is pulled back along its own radius instead of being accepted as-is.
    """
    range_list = [
        pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    pose_list: list[list[float]] = []

    for _ in range(num_bricks):
        for try_index in range(max_sample_tries):
            sample = [random.uniform(low, high) for low, high in range_list]
            last_try = try_index == max_sample_tries - 1

            if max_radius is not None and math.hypot(sample[0], sample[1]) > max_radius:
                if not last_try:
                    # Resample rather than clamp, so the distribution stays uniform over the
                    # admissible region instead of piling up on the boundary arc.
                    continue
                sample = list(sample)
                sample[0], sample[1] = clip_to_reach(sample[0], sample[1], max_radius)

            # Accept the first brick unconditionally, or on the last try (avoids an infinite loop).
            if len(pose_list) == 0 or last_try:
                pose_list.append(sample)
                break

            # Accept if sufficiently far from every already-placed brick.
            if all(math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list):
                pose_list.append(sample)
                break

    return pose_list


def sample_run_draw(rng, spatial_scale: float = 1.0):
    """One generation run's randomization draw -> LegoSortDatagenEnvCfg kwargs.

    Consumed by scripts/generate_batch.py (``LegoSortDatagenEnvCfg(**sample_run_draw(rng))``)
    and by ``maker_arm_lab.mj.scene``, so the run-level draw distribution has exactly one
    definition across both simulators. The permutation drawn here is a
    uniform shuffle for smoke purposes only -- the batch supervisor overrides
    ``slot_colors`` explicitly for without-replacement scheduling across a
    campaign. ``spatial_scale`` widens the camera pose and bin x jitter (1.0 is the
    training distribution); the MuJoCo env draws the same distribution per episode.
    """
    perm = list(range(4))
    rng.shuffle(perm)
    bl = sample_bin_layout(rng, tuple(perm), spatial_scale)
    return {
        "bin_x": bl.bin_x,
        "bin_ys": bl.bin_ys,
        "slot_colors": bl.slot_colors,
        "dome_intensity": rng.uniform(1000.0, 6000.0),
        "dome_color_temp": rng.uniform(3000.0, 8000.0),
        "sun_intensity": rng.uniform(0.0, 2000.0),
        "sun_rpy_deg": (rng.uniform(-60, 60), rng.uniform(-60, 60), rng.uniform(0, 360)),
        "floor_color": (rng.uniform(0.05, 0.6),) * 3,
        "wrist_cam_dpos": tuple(rng.uniform(-0.002, 0.002) * spatial_scale for _ in range(3)),
        "overhead_cam_dpos": tuple(rng.uniform(-0.01, 0.01) * spatial_scale for _ in range(3)),
        "overhead_cam_drot_deg": tuple(rng.uniform(-2.0, 2.0) * spatial_scale for _ in range(3)),
    }
