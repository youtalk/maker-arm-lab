"""The MS6 part catalog: LEGO-like bricks, plates, tiles and slopes as exact primitives.

Stdlib only. ``mj.scene`` turns a ``Part`` into MuJoCo geoms, ``core.grasp`` reads its size,
and ``core.batch_loop`` draws a run's part set from the split. The generator writes geometry
from three numbers: the stud pitch, the footprint clearance and the two heights. A part's
``size[0]`` is its long side along the body x axis, which is the convention the six-brick
task already uses (``layout.BRICK_SIZE`` is a 2x4 brick with its long side on x).

Mass (design H2): bounding volume times the density of ABS times a hollow fraction, floored.
The hollow fraction is chosen so a 2x4 brick lands on ``layout.BRICK_MASS``, the value every
contact constant of the MuJoCo backend was tuned at. A solid-volume rule gives 5.0 g and moves
the pinch tuning. When the real parts arrive, they go on a scale (MA track).

Flat parts (plates and tiles, 3.2 mm) left the catalog on 2026-09-18 by owner decision: the
LIFT squeezes a 3.2 mm part out of the jaws (docs/ms6-matched-jaw-opening.md section 4), and
the 31.8 mm grasps slide off one side (section 5). They stay constructible so a shard recorded
with them still replays through ``by_name``. Tall bricks (two, three and five bricks high)
took their place. Round bricks (cylinders) came and went the same day (``RETIRED_FAMILIES``).

``closing_axes`` names the body axes the jaws may close along. A box offers all three and
``core.grasp`` keeps the ones near the jaw plane; a slope offers only y (``SLOPE_AXES``).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

STUD_PITCH = 0.008
FOOTPRINT_CLEARANCE = 0.0002  # a 1x1 is 7.8 mm across
BRICK_HEIGHT = 0.0096
PLATE_HEIGHT = 0.0032
STUD_RADIUS = 0.0024
STUD_HEIGHT = 0.0017
ABS_DENSITY_KG_M3 = 1040.0
HOLLOW_FRACTION = 0.6  # 2x4 brick -> 3.0 g = layout.BRICK_MASS
MASS_FLOOR_KG = 0.0005  # solver health on 1x1 plates (0.12 g solid)
# 7 mm under the 52.4 mm per-jaw travel; the full gap is 104.8 mm (robots.frames.JAW_FULL_GAP)
MAX_GRASP_WIDTH = 0.045
HELDOUT_FRACTION = 0.2
FAMILIES = ("brick", "slope")
FLAT_FAMILIES = ("plate", "tile")  # constructible for replays, not in the catalog
RETIRED_FAMILIES = ("round",)
"""The round bricks left the catalog on 2026-09-18 after the bare-desk ladder
(~/data/maker-arm-ms6/j, rung b3): a cylinder on its side rolls when a jaw touches it, so
round_2x2 lifted 21 of 43 times and round_2x2x2 5 of 15, two rolled off the desk during the
settle, two rolled under the arm, and four of the five late arrivals still rolling in the
bin at RETREAT were round. Studs are visual only, so nothing stops the roll. They stay
constructible for replays; collision studs would be the way back."""
# (nx across, ny along): 1xN, 2xN and the three wide plates and bricks LEGO makes.
_BOX_FOOTPRINTS = tuple(
    [(1, n) for n in range(1, 9)] + [(2, n) for n in range(2, 9)] + [(4, 4), (4, 6), (4, 8)]
)
# (nx, ny, nz) of the tall bricks LEGO makes: 1x1x3, 1x2x2, 1x2x3, 1x2x5 and 2x2x3.
_TALL_BRICKS = ((1, 1, 3), (1, 2, 2), (1, 2, 3), (1, 2, 5), (2, 2, 3))
# ny is the run along the slope; ny = 1 is a pure wedge whose front edge has no height, and
# the two wedges lifted 26 of 47 times on the bare-desk read (~/data/maker-arm-ms6/i/rA).
# They stay constructible for replays, with every round brick (RETIRED_FAMILIES).
_SLOPE_FOOTPRINTS = tuple((nx, ny) for nx in (1, 2) for ny in (2, 3, 4))
_RETIRED = (
    ("round", 1, 1, 1),
    ("round", 2, 2, 1),
    ("round", 2, 2, 2),
    ("round", 4, 4, 1),
    ("slope", 1, 1, 1),
    ("slope", 2, 1, 1),
)
BOX_AXES = (0, 1, 2)
SLOPE_AXES = (1, 2)
"""A slope closes along body y, across its two flat side faces, or along body z, between its
flat bottom and its flat top. Never along x: that puts one jaw on the slope face and
squeezes the part out, which is how the wedges lost 21 of 47 lifts on the rA read at the
narrower-width tie. z matters for the 1xN slopes, 7.8 mm wide under a 9.6 mm height: they
come to rest on a side face, y points up, and 11 of 150 parts on the b3 read had no closing
axis at all. `core.grasp` keeps only the axes near the jaw plane, so an upright slope still
offers y alone."""


@dataclass(frozen=True)
class Part:
    name: str
    family: str
    nx: int
    ny: int
    size: tuple[float, float, float]  # long side (x), short side (y), height (z), m
    studs: bool
    mass: float  # kg
    hull: tuple[tuple[float, float, float], ...] | None = None  # slope vertices, part frame
    closing_axes: tuple[int, ...] = BOX_AXES  # body axes the jaws may close along

    @property
    def half_diagonal(self) -> float:
        return 0.5 * math.sqrt(sum(s * s for s in self.size))


def footprint(nx: int, ny: int) -> tuple[float, float]:
    """(long, short) footprint of an nx by ny part; ny runs along the body x axis."""
    return (STUD_PITCH * ny - FOOTPRINT_CLEARANCE, STUD_PITCH * nx - FOOTPRINT_CLEARANCE)


def part_mass(size: Sequence[float], fill: float = 1.0) -> float:
    """``fill`` is the shape's share of its bounding box: 1 for a box, pi/4 for a cylinder."""
    volume = size[0] * size[1] * size[2] * fill
    return max(MASS_FLOOR_KG, volume * ABS_DENSITY_KG_M3 * HOLLOW_FRACTION)


def slope_hull(size: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    """A 45-degree slope as a convex hull, centered on its bounding box.

    The back edge at +x is full height. The top flat runs ``size[0] - height`` back from it,
    so the face descends one height over one height. A part shorter than its height along x
    is a pure wedge with a steeper face (the 1x1 slope).
    """
    sx, sy, h = size
    base = [(-sx / 2, -sy / 2, -h / 2), (sx / 2, -sy / 2, -h / 2)]
    base += [(sx / 2, sy / 2, -h / 2), (-sx / 2, sy / 2, -h / 2)]
    back = [(sx / 2, -sy / 2, h / 2), (sx / 2, sy / 2, h / 2)]
    flat = sx - h
    if flat <= 0.0:
        return tuple(base + back)
    front = [(sx / 2 - flat, -sy / 2, h / 2), (sx / 2 - flat, sy / 2, h / 2)]
    return tuple(base + back + front)


def make_part(family: str, nx: int, ny: int, nz: int = 1) -> Part:
    """``nz`` is the height in bricks; a round part is ``nx`` studs across and ``nx == ny``."""
    known = FAMILIES + FLAT_FAMILIES + RETIRED_FAMILIES
    if family not in known:
        raise ValueError(f"family must be one of {known}, got {family!r}")
    long_side, short_side = footprint(nx, ny)
    height = (PLATE_HEIGHT if family in FLAT_FAMILIES else BRICK_HEIGHT) * nz
    size = (long_side, short_side, height)
    return Part(
        name=f"{family}_{nx}x{ny}" + (f"x{nz}" if nz > 1 else ""),
        family=family,
        nx=nx,
        ny=ny,
        size=size,
        studs=family in ("brick", "plate", "round"),
        mass=part_mass(size, math.pi / 4 if family == "round" else 1.0),
        hull=slope_hull(size) if family == "slope" else None,
        closing_axes=SLOPE_AXES if family == "slope" else BOX_AXES,
    )


def stud_positions(part: Part) -> list[tuple[float, float]]:
    """(x, y) of every stud on the top face, part frame. Tiles and slopes have none."""
    if not part.studs:
        return []
    return [
        (STUD_PITCH * (j - (part.ny - 1) / 2), STUD_PITCH * (i - (part.nx - 1) / 2))
        for j in range(part.ny)
        for i in range(part.nx)
    ]


def graspable(part: Part) -> bool:
    """The jaws close across the shorter footprint side; it must fit the opening."""
    return min(part.size[0], part.size[1]) <= MAX_GRASP_WIDTH


def catalog() -> list[Part]:
    """Every generator part that passes the graspability filter, in a fixed order."""
    out = [make_part("brick", nx, ny) for nx, ny in _BOX_FOOTPRINTS]
    out += [make_part("brick", nx, ny, nz) for nx, ny, nz in _TALL_BRICKS]
    out += [make_part("slope", nx, ny) for nx, ny in _SLOPE_FOOTPRINTS]
    return [p for p in out if graspable(p)]


def _replayable() -> list[Part]:
    """The catalog plus the flat and retired parts older shards were recorded with."""
    flat = [make_part(f, nx, ny) for f in FLAT_FAMILIES for nx, ny in _BOX_FOOTPRINTS]
    retired = [make_part(*spec) for spec in _RETIRED]
    return catalog() + [p for p in flat if graspable(p)] + retired


def split(parts: Sequence[Part], seed: int) -> tuple[list[Part], list[Part]]:
    """(train, heldout): a seeded shuffle per family holds out ceil(20 %) of that family.

    Both lists keep the catalog order, so a run's ``parts`` metadata reads the same way
    whatever the seed. Stratified so every family has a held-out ladder row (design 5).
    """
    rng = random.Random(seed)
    order = {p.name: k for k, p in enumerate(parts)}
    train: list[Part] = []
    heldout: list[Part] = []
    for family in FAMILIES:
        members = [p for p in parts if p.family == family]
        rng.shuffle(members)
        n_out = math.ceil(HELDOUT_FRACTION * len(members))
        heldout += members[:n_out]
        train += members[n_out:]
    key = lambda p: order[p.name]
    return sorted(train, key=key), sorted(heldout, key=key)


def sample_parts(rng: random.Random, parts: Sequence[Part], k: int) -> list[Part]:
    """A run's part set: ``k`` draws with replacement (design H4)."""
    return [rng.choice(list(parts)) for _ in range(k)]


def by_name(names: Sequence[str]) -> list[Part]:
    """Parts from their names, for a replay of a shard's ``parts`` attribute."""
    table = {p.name: p for p in _replayable()}
    return [table[n] for n in names]
