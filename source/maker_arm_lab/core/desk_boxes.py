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
from maker_arm_lab.core.grasp import polygon_distance

DESK_BOX_INTERIOR = layout.BIN_INTERIOR  # INITIAL: the teacher ladder decides, 9 or 11 cm
"""Interior (x, y, z) of a desk box, m. It starts at the six-brick bin's 0.09 x 0.09 x 0.04;
the teacher ladder may raise it to 0.11 x 0.11 x 0.06 if it loses parts on the drop."""
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

Two boxes on one side fill it: 1.6 % of joint draws are clear (0.92 for one box), and the
clear ones lean to small yaws, 19 % over 30 degrees against 33 % for a uniform yaw."""
DESK_BOX_CLEARANCE = 0.015
"""m between a box footprint and the pick corridor."""
DESK_BOX_GAP = 0.02
"""m between two box footprints."""
ARM_BASE_RADIUS = 0.08  # INITIAL
"""Radius (m) of the arm base in the pick corridor. The collision hulls of ``base_link`` and
``link_002`` reach 0.074 m and 0.082 m from the base axis below 0.1 m (read off the MuJoCo
model's geom bounds at the home pose, 2026-09-23)."""
DESK_BOX_RELEASE_CLEARANCE = 0.055  # INITIAL
"""m from the rim up to the pick point at the release. A held part hangs about half its
height below the pick point, and the tallest catalog part (brick_1x2x5) is 48 mm tall, so it
clears the rim by about 3 cm before the jaws open and drops straight down, level jaws."""
DESK_BOX_RETREAT_POINT = (0.15, 0.0, 0.12)  # INITIAL
"""Where RETREAT ends over desk boxes: over the near half of the tray, at the pick tilt.

The shelf's ``oracle.RETREAT_POINT`` (0.26, 0.0, 0.28) sits at the release tilt of 80 degrees,
and a near-level tool cannot hold 0.22 m over the band (``oracle.RAISE_CLEARANCE_Z``). A desk
box releases at the pick tilt, so the retreat stays at it too and low. Driven from releases
over the band (40 trials each): (0.12, 0.0, 0.15) planned 23 times, and (0.15, 0.0, 0.12),
(0.18, 0.0, 0.14) and (0.20, 0.0, 0.12) all 40, in 10 to 61 steps. The nearest of the three
to the arm keeps the most of the heap in the overhead frame."""
DRAW_TRIES = 5000


def outer_half() -> tuple[float, float]:
    """Half the outer footprint (x, y) of a box in its own frame, m."""
    return (DESK_BOX_INTERIOR[0] / 2 + DESK_BOX_WALL, DESK_BOX_INTERIOR[1] / 2 + DESK_BOX_WALL)


def rim_z() -> float:
    return DESK_BOX_WALL + DESK_BOX_INTERIOR[2]


def release_z() -> float:
    """The pick point's height at the release over a desk box."""
    return rim_z() + DESK_BOX_RELEASE_CLEARANCE


def stow_region() -> tuple[float, tuple[float, float, float]]:
    """(centre z, half extents in the box frame) of the grader's box.

    The interior footprint, and z from the box floor up to twice the interior height, so a pile
    over the rim still counts (MS6 design H6).
    """
    ix, iy, iz = DESK_BOX_INTERIOR
    return DESK_BOX_WALL + iz, (ix / 2, iy / 2, iz)


def box_geoms() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """(name, centre, size) of the floor and the four walls, in the box's own frame."""
    ix, iy, iz = DESK_BOX_INTERIOR
    w = DESK_BOX_WALL
    wall_z = w + iz / 2
    return [
        ("floor", (0.0, 0.0, w / 2), (ix + 2 * w, iy + 2 * w, w)),
        ("wall_px", (ix / 2 + w / 2, 0.0, wall_z), (w, iy + 2 * w, iz)),
        ("wall_nx", (-ix / 2 - w / 2, 0.0, wall_z), (w, iy + 2 * w, iz)),
        ("wall_py", (0.0, iy / 2 + w / 2, wall_z), (ix, w, iz)),
        ("wall_ny", (0.0, -iy / 2 - w / 2, wall_z), (ix, w, iz)),
    ]


def corners(pose: Sequence[float]) -> list[tuple[float, float]]:
    """The outer footprint's four corners in the world, counter-clockwise."""
    x, y, yaw = (float(v) for v in pose)
    hx, hy = outer_half()
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


def in_band(pose: Sequence[float]) -> bool:
    """Is the centre inside the side band, on either side, and the yaw inside its range?"""
    x, y, yaw = (float(v) for v in pose)
    return (
        DESK_BOX_BAND_X[0] <= x <= DESK_BOX_BAND_X[1]
        and DESK_BOX_BAND_Y[0] <= abs(y) <= DESK_BOX_BAND_Y[1]
        and abs(yaw) <= DESK_BOX_MAX_YAW
    )


def clear(poses: Sequence[Sequence[float]]) -> bool:
    """Every footprint clears the pick corridor and every other footprint."""
    corridor = pick_corridor()
    prints = [corners(p) for p in poses]
    if any(polygon_distance(c, corridor) < DESK_BOX_CLEARANCE for c in prints):
        return False
    return all(
        polygon_distance(prints[a], prints[b]) >= DESK_BOX_GAP
        for a in range(len(prints))
        for b in range(a + 1, len(prints))
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
                    rng.uniform(*DESK_BOX_BAND_X),
                    sign * rng.uniform(*DESK_BOX_BAND_Y),
                    rng.uniform(-DESK_BOX_MAX_YAW, DESK_BOX_MAX_YAW),
                )
                for _ in slots
            ]
            if clear(side):
                break
        else:
            raise RuntimeError(
                f"no clear draw of {len(slots)} desk boxes on one side in {DRAW_TRIES} tries"
            )
        for k, pose in zip(slots, side):
            poses[k] = pose
    return poses


def part_destinations(num_parts: int, num_boxes: int) -> list[int]:
    """The box each part goes to, one entry per part.

    One box takes every part. With more boxes, part b goes to box b modulo their number: the
    heap's part order is already drawn at random, so this spreads the parts without a draw.
    ``feat/ms7-classes`` replaces it with the episode's class-to-box map.
    """
    return [b % num_boxes for b in range(num_parts)]
