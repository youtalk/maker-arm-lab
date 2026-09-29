"""MS7a desk boxes: open boxes that stand on the desk beside the arm. Torch and stdlib only.

MS7a design, desk boxes: with the switch on, the shelf and its heap bin leave the scene, and
N open boxes (a floor and four walls) stand on the desk top (z = 0). They stand to the left
(+y) and right (-y) of the arm, never behind it and never in front of the tray. Each episode
draws every box's centre inside the side band and its yaw uniform in +-DESK_BOX_MAX_YAW,
which covers every orientation of a square box. Every footprint keeps DESK_BOX_CLEARANCE
from the pick corridor, the convex hull of the arm base and the tray's outer walls, so no box
lies under the forearm during a pick. Two boxes keep DESK_BOX_GAP apart.

A box pose is ``(x, y, yaw)``: the centre of its footprint on the desk and the heading of its
own x axis. ``mj.scene`` builds each box as a mocap body from ``box_geoms`` and ``mj.env``
moves it per episode.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from maker_arm_lab.core import layout
from maker_arm_lab.core.grasp import Wall, polygon_distance
from maker_arm_lab.core.parts import BASE_CLASSES, Part, part_class

DESK_BOX_INTERIOR_ONE = (0.11, 0.11, 0.06)  # INITIAL: rung 1, one box takes all 50 parts
"""Interior (x, y, z) of the one desk box of rung 1, m. The six-brick bin's 0.09 x 0.09 x 0.04
lost parts over its rim: the first one-box teacher smoke (2026-09-23, two held-out heaps of
50) stowed 93 of 100, and on heap 1 five failed cycles left their part piled at z 0.086 to
0.096 m, above the grader's band (twice the 0.04 m interior)."""
DESK_BOX_INTERIOR_MANY = layout.BIN_INTERIOR  # INITIAL: rung 2, four boxes of ~12 parts each
"""Interior of each desk box when there are several: the six-brick bin's 0.09 x 0.09 x 0.04.
Four boxes share a heap of 50, and the four-box smoke stowed 97 of 100 with no pile loss."""
DESK_BOX_WALL = layout.BIN_WALL
DESK_BOX_RGBA = (0.12, 0.12, 0.14, 1.0)
"""Near black. The desk color draw (``mj.env.DESK_COLOR_RANGE``) is brown to pink, and a
bright dome draw saturates it toward white, where a white box vanished in the overhead frame
(rendered 2026-09-23). A dark box reads against every draw, and it is none of the four part
colors."""
DESK_BOX_MAX_YAW = math.radians(45.0)

DESK_BOX_BAND_X = (0.05, 0.21)
DESK_BOX_BAND_Y = (0.18, 0.22)
"""The side band: the box centre's x and |y|, on both sides.

Measured first on a driven grid of box centres (a lone brick_2x4 picked at the tray centre
and released over one box, 4 trials a cell, x 0.02 to 0.24 by 0.01, |y| 0.18 to 0.25 by
0.01, 2026-09-23): every centre with |y| <= 0.20 landed from x 0.02 to 0.24. The reach then
narrows, to x 0.04-0.23 at |y| 0.22, 0.07-0.19 at 0.24 and 0.10-0.16 at 0.25, and past it
EXTEND has no plan at all (the tool leans toward +x, ``core.global_ik``). The band keeps 1 to
3 cm from every failed cell. Its lower |y| is the corridor's: a box yawed 45 degrees at |y|
0.18 already comes within DESK_BOX_CLEARANCE of the tray's side wall.

Certified by ``scripts/mj_release_envelope_check.py`` on Machine A, 2026-09-23: the band's
four corners and its centre, both sides, box yaw -45 and +45 degrees, part yaw 0, 45, 90 and
-45 degrees, continuous gripper, release at DESK_BOX_RELEASE_Z 0.10 m. brick_2x4, brick_1x2x5
and brick_2x7 each landed 80 of 80 with EXTEND converged: EXTEND residual at most 1.19 mm,
RETREAT's at most 0.78 mm, landing at most 21 mm from the box centre, no arm geom touching a
box, tightest joint 0.039 rad from its stop at the release.

Two boxes on one side fill it: 1.6 % of joint draws are clear (0.92 for one box), and the
clear ones lean to small yaws, 19 % over 30 degrees against 33 % for a uniform yaw."""
DESK_BOX_BAND_X_ONE = (0.08, 0.21)
"""The one 11 cm box of rung 1 starts its band at x 0.08, not 0.05. Its outer corner hangs
8.5 cm from its centre at a 45 degree yaw, 65 mm up at the rim, and at x 0.05 that corner
left the overhead frame by 6.1 px with the episode's camera draw at its extremes (+-0.01 m,
+-2 degrees per axis); at 0.08 it keeps 3.5 px (projected, 2026-09-23).

Certified with the 11 cm box by ``scripts/mj_release_envelope_check.py --desk_boxes 1`` the
same day, same grid of points and yaws as the band above: brick_2x4, brick_1x2x5 and
brick_2x7 each landed 80 of 80 with EXTEND converged (residual at most 1.19 mm, RETREAT's
at most 0.84 mm, landing at most 24 mm from the box centre, no arm geom touching the box,
tightest joint 0.099 rad from its stop). The 9 cm box, now the four-box box, re-read 80 of
80 on brick_2x4 with ``--desk_boxes 4`` (the other three boxes parked off the desk)."""
DESK_BOX_CLEARANCE = 0.015
"""m between a box footprint and the pick corridor."""
DESK_BOX_GAP = 0.02
"""m between two box footprints."""
ARM_BASE_RADIUS = 0.08  # INITIAL
"""Radius (m) of the arm base in the pick corridor. The collision hulls of ``base_link`` and
``link_002`` reach 0.074 m and 0.082 m from the base axis below 0.1 m (read off the MuJoCo
model's geom bounds at the home pose, 2026-09-23)."""
DESK_BOX_RELEASE_Z = 0.10  # INITIAL
"""The pick point's height at the release over any desk box, m. A held part hangs about half
its height below the pick point: the tallest catalog part (brick_1x2x5, 48 mm) clears the
9 cm box's 45 mm rim by about 3 cm and the 11 cm box's 65 mm rim by about 1 cm, then drops
straight down between the level jaws. It is not raised with the 11 cm box's rim: at 0.12 m
the driven reach over the band shrank (4 trials a cell, 2026-09-23) to x 0.08-0.22 at |y|
0.18, 0.09-0.22 at 0.20, 0.11-0.20 at 0.21 and 0.14-0.16 at 0.22, against x 0.02-0.24 up to
|y| 0.20 at 0.10 m, and a 45 degree box beside the tray needs |y| 0.205 or more."""
DESK_BOX_STOW_HEIGHT = 0.08
"""How far above a box floor the grader still counts a part in the box, m: twice the 9 cm
box's 0.04 m interior, so a pile over its rim counts (MS6 design H6). The 11 cm box keeps the
same 0.08 m, not twice its 0.06 m: that band would reach 0.125 m, above DESK_BOX_RELEASE_Z,
and a part still in the jaws over the box counted as stowed. The first 11 cm smoke ended
heap 2 as a success that way, its last part held at z 0.099 m (2026-09-23)."""
DESK_BOX_RETREAT_POINT = (0.15, 0.0, 0.12)  # INITIAL
"""Where RETREAT ends over desk boxes: over the near half of the tray, at the pick tilt.

The shelf's ``oracle.RETREAT_POINT`` (0.26, 0.0, 0.28) sits at the release tilt of 80 degrees,
and a near-level tool cannot hold 0.22 m over the band (``oracle.RAISE_CLEARANCE_Z``). A desk
box releases at the pick tilt, so the retreat stays at it too and low. Driven from releases
over the band (40 trials each): (0.12, 0.0, 0.15) planned 23 times, and (0.15, 0.0, 0.12),
(0.18, 0.0, 0.14) and (0.20, 0.0, 0.12) all 40, in 10 to 61 steps. The nearest of the three
to the arm keeps the most of the heap in the overhead frame."""
DRAW_TRIES = 5000
DESK_BOX_SLOTS = 8
"""MS7e design section 4.2: with the box draw on, the scene builds this many boxes."""
BOX_COUNT_MIN = 4
"""MS7e design section 4.2: the box draw's count is uniform from here to the rung's ceiling."""
BOX_MIN = 0.07
BOX_MAX = 0.10
BOX_MAX_ASPECT = 1.4
BOX_DEPTH = (0.03, 0.05)
"""The box draw's interior (MS7e design section 4.2): each footprint side in BOX_MIN to
BOX_MAX with at most BOX_MAX_ASPECT between the two, and the depth in BOX_DEPTH. The rim is the
depth plus DESK_BOX_WALL.

BOX_MIN is the size floor of MS7e ledger R4 a, its first step: ``scripts/
mj_release_envelope_check.py --desk_boxes 4 --interior 0.07,0.07,0.05`` on the MS7a band
landed brick_2x4, brick_1x2x5, brick_2x7, brick_1x8 and brick_2x8 each 80 of 80 with EXTEND
converged (EXTEND residual at most 1.20 mm, landing at most 15.4 mm from the box centre, no
arm geom touching the box, 2026-09-29)."""
SIZE_DRAWS = 200
"""How often one side of the box draw draws its sizes. Three boxes on a side of B1_BAND_X fit
only when their sides are small: 20 size draws placed five boxes in 554 of the 1000 ceiling
seeds on x 0.02..0.24 at BOX_MIN 0.07, and 200 place them in 999 (2026-09-29). A failed size
draw costs microseconds, so a draw stays under 3 ms."""
SIZE_ROUNDS = 10
"""A side that exhausts SIZE_DRAWS draws again from the same generator, up to this many rounds
of SIZE_DRAWS in all; only past that does the draw raise, a defect and not an episode outcome
(MS7e ledger D7). The count is never stepped down, so the count distribution stays uniform.
On the B2 band 4 of 1000 seeds exhaust the first round at eight boxes (the lean track)."""
YAW_CLIP_STEP = math.radians(1.0)
B1_BAND_X = (0.03, 0.25)
B1_BAND_Y = (0.18, 0.20)
"""MS7e rung B1's band: the box centre's x and |y| under the box draw.

The MS7a band (DESK_BOX_BAND_X) holds two boxes a side: its 0.16 m of centre x fits three
boxes of 8 cm outside and 2 cm gaps in no seed. This band takes the MS7a driven reach at |y|
0.20 or less (x 0.02 to 0.24), leaves its back cell and adds 1 cm in front.

Certified by ``scripts/mj_release_envelope_check.py --desk_boxes 4 --band_x=0.03,0.25
--band_y=0.18,0.20`` with 7 and 10 cm interiors 0.05 m deep on Machine A, 2026-09-29:
brick_2x4, brick_1x2x5, brick_2x7, brick_1x8 and brick_2x8 each landed 80 of 80 with EXTEND
converged at both sizes, EXTEND residual at most 1.56 mm, RETREAT's at most 0.84 mm, no
abandoned cycle, no arm geom touching a box, tightest joint 0.032 rad from its stop.

Its edges keep 1 to 2 cm from failed cells. At x 0.02, |y| 0.18, box yaw -45 degrees and part
yaw 90 degrees, RETREAT ended 119 mm from its goal and the rig abandoned, for four of the five
parts, and the tightest joint came within 0.010 rad at |y| 0.20. In front (brick_2x4, 16
trials a cell): x 0.26 landed every trial, x 0.27 none at |y| 0.20, x 0.28 none at |y| 0.19
and 0.20, and x 0.30 to 0.34 none."""
B1_CEILING = 6
"""The largest box count B1's draw places in at least 990 of 1000 seeds (20290000 to 20290999,
MS7e ledger R4 d): six in 1000 of 1000, seven in none (2026-09-29)."""
UNSEEN_DRAWS = {
    "size": {"sides": (BOX_MIN, 0.12), "aspect": 1.7, "beyond": (BOX_MAX, BOX_MAX_ASPECT)},
    "rim": {"depth": (0.05, 0.06), "back": -0.010},
}
"""The unseen-box ladder rows (MS7e design section 8.3), as ``draw_boxes`` arguments over the
rung's own. Size: every box has a side over BOX_MAX or an aspect over BOX_MAX_ASPECT, with
sides to 0.12 m and aspects to 1.7. Rim: interiors 0.05 to 0.06 m deep; the 0.065 m rim
projects further out of the frame, so the back edge moves to -0.010 (0.4 px inside at |y|
0.30, 2026-09-29)."""
UNSEEN_CEILINGS = {"size": 4, "rim": 6}
"""The largest count each unseen row's draw places in at least 990 of the 1000 ceiling seeds on
B1's band (2026-09-29): with sizes to 0.12 m five boxes place in 840, with 0.065 m rims six in
1000. The row draws counts up to this ("at the counts that fit")."""
FRAME_BACK_X = -0.015
"""The box draw keeps every footprint corner at x >= this, m: the overhead frame's back edge.

Projected at the rim of the deepest drawn box (0.055 m) with the camera draw at its extremes
(+-0.01 m, +-2 degrees per axis, 2026-09-29): a rim point at x -0.015 keeps 0.5 px inside the
frame at |y| 0.30 and 2.2 px at |y| 0.20, and the margin grows about 3 px per cm of x. The
frame's other edges keep 5 px or more over the whole band. The MS7a band's back edge at x 0.05
was sized for the 9 cm box: a 10 cm box yawed 45 degrees there leaves the frame by 2.1 px."""
B1_DRAW = {"band_x": B1_BAND_X, "band_y": B1_BAND_Y, "back": FRAME_BACK_X}
"""Rung B1's ``draw_boxes`` arguments: its band and the frame's back edge (MS7e ledger D6). A
rung passes its own; an unseen row's ``UNSEEN_DRAWS`` entry goes over them."""
BOX_FILTER_MARGIN_M = 0.005  # INITIAL: the MS7d isolation row R1 measures it
"""m: ``in_outer_footprint`` grows every box's outer footprint by this on each side (MS7d
design section 4.2, L1). The perceived candidate filter tested a detection's 3D centre against
the grader's region, whose lower face is the box floor, so a z error of a few mm under a part
at rest in its box put it back among the candidates: 92 cycles of MS7c candidate 3."""


def interior(n: int) -> tuple[float, float, float]:
    """The interior of every box when ``n`` MS7a desk boxes stand on the desk. The functions
    below take each box's own interior (``size``), which is this one with the box draw off."""
    return DESK_BOX_INTERIOR_ONE if n == 1 else DESK_BOX_INTERIOR_MANY


def interiors(n: int) -> tuple[tuple[float, float, float], ...]:
    """Every box's interior when ``n`` MS7a desk boxes stand on the desk."""
    return (interior(n),) * n


def outer_half(size: Sequence[float]) -> tuple[float, float]:
    """Half the outer footprint (x, y) of a box of interior ``size`` in its own frame, m."""
    return (size[0] / 2 + DESK_BOX_WALL, size[1] / 2 + DESK_BOX_WALL)


def rim_z(size: Sequence[float]) -> float:
    return DESK_BOX_WALL + size[2]


def stow_region(size: Sequence[float]) -> tuple[float, tuple[float, float, float]]:
    """(centre z, half extents in the box frame) of the grader's box of interior ``size``: the
    interior footprint, and z from the box floor up DESK_BOX_STOW_HEIGHT whatever the depth."""
    ix, iy = size[0], size[1]
    h = DESK_BOX_STOW_HEIGHT
    return DESK_BOX_WALL + h / 2, (ix / 2, iy / 2, h / 2)


def in_outer_footprint(xy, poses, sizes, margin: float = BOX_FILTER_MARGIN_M) -> bool:
    """Does the point ``xy`` lie in the outer footprint (interior plus wall) of any of these
    box ``poses``, one interior in ``sizes`` per pose, grown by ``margin`` on every side, in
    each box's own frame? Its height does not count. Pure."""
    for (x, y, yaw), size in zip(poses, sizes):
        hx, hy = outer_half(size)
        dx, dy = float(xy[0]) - float(x), float(xy[1]) - float(y)
        c, s = math.cos(yaw), math.sin(yaw)
        if abs(c * dx + s * dy) <= hx + margin and abs(c * dy - s * dx) <= hy + margin:
            return True
    return False


DESK_BOX_FLOOR_DEPTH = 0.010
"""How far (m) the floor's collider reaches under the desk top with ``deep_box_floor`` (MS7e
flat track, lever 1). Its top face stays at DESK_BOX_WALL, so a part at rest on it does not move.

A flat part (3.2 mm) released at DESK_BOX_RELEASE_Z hangs at the carry's 20 degree tool lean
and lands corner first at about 1.3 m/s, about 5.5 mm per 240 Hz step. Once its centre passes
the 5 mm floor's centre (1.6 + 2.5 mm of penetration) the box-box contact normal flips, and the
floor pins the part under it on the desk. Dropped still at that lean over a box (40 drops a
cell, 2026-09-29): plate_2x4 fell through 24, 10, 17 and 0 times from 0.10, 0.08, 0.06 and
0.04 m, plate_1x2 6, 21, 0 and 0, brick_2x4 never. With the collider this much deeper none of
320 plate drops fell through. Any depth past about 2 mm keeps the floor's centre under a part
resting on the desk; 10 mm is chosen, not swept. The desk stands under it and a mocap box
never collides with the world, so the extra depth touches nothing else."""


def box_geoms(
    size: Sequence[float], floor_depth: float = 0.0
) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """(name, centre, size) of the floor and the four walls of a box of interior ``size``, in
    the box's own frame. The floor reaches ``floor_depth`` under the desk top
    (``DESK_BOX_FLOOR_DEPTH``), its top unchanged."""
    ix, iy, iz = (float(v) for v in size)
    w = DESK_BOX_WALL
    wall_z = w + iz / 2
    return [
        ("floor", (0.0, 0.0, (w - floor_depth) / 2), (ix + 2 * w, iy + 2 * w, w + floor_depth)),
        ("wall_px", (ix / 2 + w / 2, 0.0, wall_z), (w, iy + 2 * w, iz)),
        ("wall_nx", (-ix / 2 - w / 2, 0.0, wall_z), (w, iy + 2 * w, iz)),
        ("wall_py", (0.0, iy / 2 + w / 2, wall_z), (ix, w, iz)),
        ("wall_ny", (0.0, -iy / 2 - w / 2, wall_z), (ix, w, iz)),
    ]


def box_walls(poses: Sequence[Sequence[float]], sizes) -> tuple[Wall, ...]:
    """The four walls of each box at these ``poses`` (one interior in ``sizes`` each) for the
    grasp model, as ``grasp.tray_walls`` gives the tray's (MS7d design section 4.2, L9). Each
    footprint is the wall's ``box_geoms`` rectangle turned by the box yaw, the top is the rim,
    the heading runs along the wall's length, ``outward`` points away from the box centre, and
    no wall leans. ``Wall.box`` marks them. Pure."""
    walls = []
    for (x, y, yaw), box in zip(poses, sizes):
        c, s = math.cos(yaw), math.sin(yaw)
        for name, (u, v, _), size in box_geoms(box):
            if name == "floor":
                continue
            hu, hv = size[0] / 2, size[1] / 2
            footprint = tuple(
                (x + c * (u + a) - s * (v + b), y + s * (u + a) + c * (v + b))
                for a, b in ((hu, hv), (-hu, hv), (-hu, -hv), (hu, -hv))
            )
            out = math.atan2(s * u + c * v, c * u - s * v)
            normal = (math.cos(out), math.sin(out))
            walls.append(Wall(footprint, out + math.pi / 2, rim_z(box), normal, box=True))
    return tuple(walls)


def corners(pose: Sequence[float], size: Sequence[float]) -> list[tuple[float, float]]:
    """The outer footprint's four corners in the world, counter-clockwise, of a box of
    interior ``size``."""
    x, y, yaw = (float(v) for v in pose)
    hx, hy = outer_half(size)
    c, s = math.cos(yaw), math.sin(yaw)
    return [
        (x + c * u - s * v, y + s * u + c * v)
        for u, v in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))
    ]


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def pick_corridor() -> list[tuple[float, float]]:
    """The convex hull of the arm base (a 16-gon of ARM_BASE_RADIUS) and the tray's outer
    walls, counter-clockwise (Andrew's monotone chain)."""
    (x0, x1), (y0, y1) = layout.tray_bounds()
    w = layout.TRAY_WALL
    points = [(x0 - w, y0 - w), (x1 + w, y0 - w), (x1 + w, y1 + w), (x0 - w, y1 + w)]
    points += [
        (ARM_BASE_RADIUS * math.cos(a), ARM_BASE_RADIUS * math.sin(a))
        for a in (2 * math.pi * k / 16 for k in range(16))
    ]
    points = sorted(set(points))

    def chain(seq):
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2 and _cross(out[-2], out[-1], p) <= 0.0:
                out.pop()
            out.append(p)
        return out[:-1]

    return chain(points) + chain(reversed(points))


def band_x(n: int) -> tuple[float, float]:
    """The band's x range when ``n`` desk boxes stand on the desk."""
    return DESK_BOX_BAND_X_ONE if n == 1 else DESK_BOX_BAND_X


def in_band(pose: Sequence[float], n: int) -> bool:
    """Is the centre inside the side band, on either side, and the yaw inside its range?"""
    x, y, yaw = (float(v) for v in pose)
    x0, x1 = band_x(n)
    return (
        x0 <= x <= x1
        and DESK_BOX_BAND_Y[0] <= abs(y) <= DESK_BOX_BAND_Y[1]
        and abs(yaw) <= DESK_BOX_MAX_YAW
    )


def clear(poses: Sequence[Sequence[float]], sizes: Sequence[Sequence[float]]) -> bool:
    """Every footprint of these boxes, one interior in ``sizes`` per pose, clears the pick
    corridor and every other footprint.

    Two boxes whose inscribed discs come within DESK_BOX_GAP fail, and two whose circumscribed
    discs keep it pass, before any polygon distance: the answer is the same, and a crowded
    side's rejection draw runs about ten times faster."""
    halves = [outer_half(size) for size in sizes]
    for a in range(len(poses)):
        for b in range(a + 1, len(poses)):
            d = math.hypot(poses[a][0] - poses[b][0], poses[a][1] - poses[b][1])
            if d < min(halves[a]) + min(halves[b]) + DESK_BOX_GAP:
                return False
            if d >= math.hypot(*halves[a]) + math.hypot(*halves[b]) + DESK_BOX_GAP:
                continue
            gap = polygon_distance(corners(poses[a], sizes[a]), corners(poses[b], sizes[b]))
            if gap < DESK_BOX_GAP:
                return False
    corridor = pick_corridor()
    return all(
        polygon_distance(corners(p, size), corridor) >= DESK_BOX_CLEARANCE
        for p, size in zip(poses, sizes)
    )


def sample_boxes(rng: random.Random, n: int) -> list[tuple[float, float, float]]:
    """``n`` box poses for one episode, from ``rng`` alone so a seed reproduces them.

    One box stands on a side drawn per episode. More boxes split over the two sides, the
    first side taking the odd one. Each side is drawn on its own by rejection: the corridor
    lies between the sides, so boxes on opposite sides never meet.
    """
    if n <= 0:
        return []
    signs = [rng.choice((1.0, -1.0))] if n == 1 else [(1.0, -1.0)[k % 2] for k in range(n)]
    poses: list[tuple[float, float, float] | None] = [None] * n
    for sign in (1.0, -1.0):
        slots = [k for k in range(n) if signs[k] == sign]
        if not slots:
            continue
        for _ in range(DRAW_TRIES):
            side = [
                (
                    rng.uniform(*band_x(n)),
                    sign * rng.uniform(*DESK_BOX_BAND_Y),
                    rng.uniform(-DESK_BOX_MAX_YAW, DESK_BOX_MAX_YAW),
                )
                for _ in slots
            ]
            if clear(side, [interior(n)] * len(side)):
                break
        else:
            raise RuntimeError(
                f"no clear draw of {len(slots)} desk boxes on one side in {DRAW_TRIES} tries"
            )
        for k, pose in zip(slots, side):
            poses[k] = pose
    return poses


def draw_interior(
    rng: random.Random,
    sides: tuple[float, float] = (BOX_MIN, BOX_MAX),
    aspect: float = BOX_MAX_ASPECT,
    depth: tuple[float, float] = BOX_DEPTH,
    beyond: tuple[float, float] | None = None,
) -> tuple[float, float, float]:
    """One box's interior (x, y, z): both footprint sides uniform in ``sides``, drawn again
    until neither exceeds ``aspect`` times the other, and the depth uniform in ``depth``.

    ``beyond`` (a side and an aspect) also draws again a footprint within both: the unseen
    size row takes only boxes with a side or an aspect that training never drew."""
    while True:
        ix, iy = rng.uniform(*sides), rng.uniform(*sides)
        ratio = max(ix, iy) / min(ix, iy)
        if ratio > aspect:
            continue
        if beyond is not None and max(ix, iy) <= beyond[0] and ratio <= beyond[1]:
            continue
        return ix, iy, rng.uniform(*depth)


def _place_in_order(rng, sizes, sign, band_x, band_y, back):
    """One side's boxes along x in draw order, or None if these sizes cannot fit the band.

    Each centre is drawn in the interval the boxes before it leave, keeping room for the boxes
    after it at zero yaw and its back corner at x ``back`` or more. Its yaw is drawn first and
    then turned toward zero, a degree at a time, until the box's x extent fits. The x extents
    keep DESK_BOX_GAP between neighbours, so only the corridor can still refuse the draw."""
    halves = [outer_half(size) for size in sizes]
    x0, x1 = band_x
    poses, right = [], -math.inf
    for j, (hx, hy) in enumerate(halves):
        later = halves[j + 1 :]
        room = sum(DESK_BOX_GAP + 2 * h[0] for h in later) - (later[-1][0] if later else 0.0)
        yaw = rng.uniform(-DESK_BOX_MAX_YAW, DESK_BOX_MAX_YAW)
        while True:
            extent = hx * abs(math.cos(yaw)) + hy * abs(math.sin(yaw))
            lo = max(x0, back + extent, right + DESK_BOX_GAP + extent)
            hi = x1 - room - extent if later else x1
            if lo <= hi or yaw == 0.0:
                break
            yaw = math.copysign(max(0.0, abs(yaw) - YAW_CLIP_STEP), yaw)
        if lo > hi:
            return None
        x = rng.uniform(lo, hi)
        poses.append((x, sign * rng.uniform(*band_y), yaw))
        right = x + extent
    return poses


def _draw_side(rng, sizes, sign, band_x, band_y, back):
    """One side's poses for these interiors in DRAW_TRIES placements, or None."""
    for _ in range(DRAW_TRIES):
        side = _place_in_order(rng, sizes, sign, band_x, band_y, back)
        if side is None or clear(side, sizes):
            return side
    return None


def draw_boxes(
    rng: random.Random,
    n: int,
    band_x: tuple[float, float] = B1_BAND_X,
    band_y: tuple[float, float] = B1_BAND_Y,
    sides: tuple[float, float] = (BOX_MIN, BOX_MAX),
    aspect: float = BOX_MAX_ASPECT,
    depth: tuple[float, float] = BOX_DEPTH,
    beyond: tuple[float, float] | None = None,
    back: float = FRAME_BACK_X,
    size_draws: int = SIZE_DRAWS,
    rounds: int = SIZE_ROUNDS,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]], int]:
    """The MS7e box draw (design section 4.2): ``n`` box poses, their interiors and the extra
    rounds of ``size_draws`` its sides took, from ``rng`` alone so a seed reproduces them.

    The boxes alternate sides, and the side of the odd box is drawn. Each side draws its
    interiors (``draw_interior``), then places them along x in order (``_place_in_order``),
    every footprint corner at x ``back`` or more (the frame, FRAME_BACK_X). A side whose sizes
    cannot fit, or that fails DRAW_TRIES placements on the corridor, draws its interiors again,
    up to ``rounds`` rounds of ``size_draws`` (SIZE_ROUNDS), and raises past them.

    Not by rejection as ``sample_boxes`` places: on every band tried, rejection placed three
    boxes on one side in 0 of 1000 seeds (5000 tries times 20 size draws, 2026-09-29), where the
    in-order placement places them in 1000 of 1000 on B1_BAND_X.
    """
    first = rng.choice((1.0, -1.0))
    signs = [first if k % 2 == 0 else -first for k in range(n)]
    poses: list = [None] * n
    sizes: list = [None] * n
    extra = 0
    for sign in (first, -first):
        slots = [k for k in range(n) if signs[k] == sign]
        if not slots:
            continue
        for attempt in range(rounds * size_draws):
            drawn = [draw_interior(rng, sides, aspect, depth, beyond) for _ in slots]
            side = _draw_side(rng, drawn, sign, band_x, band_y, back)
            if side is not None:
                extra += attempt // size_draws
                break
        else:
            raise RuntimeError(
                f"no clear draw of {len(slots)} desk boxes on one side in {rounds} rounds of "
                f"{size_draws} size draws"
            )
        for k, pose, size in zip(slots, side, drawn):
            poses[k], sizes[k] = pose, size
    return poses, sizes, extra


def class_map(rng: random.Random, n: int, classes: Sequence[str] = BASE_CLASSES) -> tuple[int, ...]:
    """The episode's class-to-box map: the box of each of the active ``classes``, in order.

    MS7a design, classes and the class map: with four boxes it is a random permutation of the
    four classes over the four boxes, drawn from ``rng`` alone, so no shortcut from a part's
    looks to a box position survives across episodes. One box takes every class and draws
    nothing. Any other count folds a permutation onto its boxes: the eight MS7e classes
    (``parts.PART_CLASSES``) shuffle over the boxes and fold ``% n`` (MS7e design section 5).
    """
    if n == 1:
        return (0,) * len(classes)
    boxes = list(range(len(classes)))
    rng.shuffle(boxes)
    return tuple(b % n for b in boxes)


def part_destinations(
    parts: Sequence[Part], class_to_box: Sequence[int], classes: Sequence[str] = BASE_CLASSES
) -> list[int]:
    """The box each part goes to, one entry per part: its class's box, ``class_to_box`` read in
    the order of the active ``classes``."""
    return [int(class_to_box[list(classes).index(part_class(p))]) for p in parts]
