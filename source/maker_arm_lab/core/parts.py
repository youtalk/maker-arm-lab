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

MS7f (MS7f design section 6.1) sorts by six categories (``PART_CATEGORIES``,
``part_category``) in place of the eight classes when its ``--categories`` switch is on.

MS7e (MS7e design sections 5 and 6) brings three families back one at a time: round, LDraw
and flat (plates and tiles, no 4-wide part). ``catalog(families)`` adds a family only when a
``FAMILY_KEYS`` key names it, so with no key the catalog, the split and the four classes are
MS7d's. A key whose behaviour is not in the commit is not in ``READY_FAMILIES``, and every
script refuses it with its fail token.

Neither flat mechanism of 2026-09-18 survives the continuous gripper, which holds the close at
``grasp.close_gap`` instead of running the jaws shut: the MS7e lone-part table (2026-09-29,
12 trials per flat part) lifted 120 of 120. What it lost, 34 of 120, was the release: the
part dropped through the 5 mm desk box floor. The flat track's ``--deep_box_floor``
(``desk_boxes.DESK_BOX_FLOOR_DEPTH``) stows 120 of 120.

``closing_axes`` names the body axes the jaws may close along. A box offers all three and
``core.grasp`` keeps the ones near the jaw plane; a slope offers only y (``SLOPE_AXES``).

An LDraw part (``ldraw_part``) is read from ``assets/parts/<id>.json``, which
``scripts/ldraw_parts.py`` wrote from the LDraw library: its size is the collision box, its
collider is up to four convex hulls (``Part.hulls``) and its studs are visual
(``Part.stud_frames``). Its mass keeps the H2 rule, and each hull carries the share of it that
its volume holds (MS7e design section 6.1).

MS7f (MS7f design section 7) adds two more families the same generator builds: joint
(``JOINT_IDS``) and minifig (``MINIFIG_IDS``). Their parts are named ``<family>_<id>``, come after tile in ``FAMILIES`` so no
catalog index moves, split on their own generator and class as ``special`` while the eight
MS7e classes sort.
"""

from __future__ import annotations

import functools
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
FAMILIES = ("brick", "slope", "round", "ldraw", "plate", "tile", "joint", "minifig")
"""Catalog order. The MS7e families come after brick and slope, so ``split`` draws the base
split unchanged (MS7e design section 5), and the MS7f families after tile."""
FAMILY_KEYS = {
    "round": ("round",),
    "ldraw": ("ldraw",),
    "flat": ("plate", "tile"),
    "joint": ("joint",),
    "minifig": ("minifig",),
}
"""The keys ``--families`` accepts, and the families each brings into the catalog."""
READY_FAMILIES: frozenset[str] = frozenset({"round", "ldraw", "flat", "joint", "minifig"})
"""The ``FAMILY_KEYS`` keys whose behaviour is in this commit. Each family track adds its key
when it lands; ``--families`` with any other key is a usage error (MS7e design section 10.2)."""
LDRAW_IDS = ("3700", "3701", "3894", "4490", "3659", "87087", "11211", "3660")
"""The pre-registered LDraw parts, in catalog order (MS7e design section 6.5). Part names are
``ldraw_<id>``; ``ldraw_part`` loads each from ``assets/parts/<id>.json``."""
JOINT_IDS = ("2780", "6558", "43093", "3937", "3938", "14417", "14418", "4081b")
"""MS7f joint family: three Technic pins (friction, long, axle pin), the two halves of a 1x2
hinge brick, a 1x2 plate with a side ball, one with a socket and a 1x1 plate with a clip."""
MINIFIG_IDS = ("3626b", "973", "3815c01", "3901", "3624", "3838", "4349")
"""MS7f minifig family, each part one rigid body: a head, a torso without arms, the hips and
legs as one assembly, male hair, a police hat, air tanks and a loudhailer. The cowboy hat 3629
left on 2026-09-30: upright it offers only its brim's edge, which the jaws ride over, and on
the minifig heap read (run 20303500 + i, layout 20303550 + i) its 21 cycles took 1009 steps
each against 346 for the parts of the same seeds without the new families."""
LDRAW_FAMILY_IDS = {"ldraw": LDRAW_IDS, "joint": JOINT_IDS, "minifig": MINIFIG_IDS}
"""Every family read from ``LDRAW_DIR``, with its part ids in catalog order."""
CLASSIFIER_EXTRA = (
    *("brick_1x1x2", "brick_1x1x5", "brick_1x4x3", "brick_2x2x2", "brick_2x4x2"),
    *("brick_1x10", "brick_2x10"),
    *("ldraw_3037", "ldraw_3298", "ldraw_3665a", "ldraw_3702", "ldraw_32000", "ldraw_6541"),
    *("ldraw_4070", "ldraw_30414"),
    *("plate_1x3", "plate_1x8", "plate_2x3", "plate_2x8"),
    *("tile_1x3", "tile_1x6", "tile_1x8", "tile_2x3"),
    *("round_1x1x2", "round_1x1x3", "round_2x2x3", "round_4x4x2"),
    *("joint_32054", "joint_32002", "joint_6628", "joint_63868", "joint_60470a"),
    *("joint_4276b", "joint_44301a", "joint_2430", "joint_14419"),
    *("minifig_6132", "minifig_3625", "minifig_3959", "minifig_973c01", "minifig_30162"),
    *("minifig_86035", "minifig_30135", "minifig_4530", "minifig_6025"),
)
"""Kinds only the category classifier's render set sees (MS7f, the controller's deviation of
2026-09-30): in no catalog, no split and no ``catalog_index``, so the catalog's held-out kinds
stay unseen by the policy and the classifier. ``by_name`` builds them (``extra_part``), and
each family's ``FAMILY_CATEGORY`` gives its category. They add category cues the ~50 train
kinds lack: other bricks, slopes, Technic bricks, arches and side-stud bricks, other plate and
tile footprints, taller round bricks, other pins, clip, hinge and ball plates, and other
helmets, hair and accessories. No primitive extra repeats a catalog kind's family and size,
and none has a category a shape leaves open (a round plate would be round and flat). The domed
helmets 3833 and 3844 stayed out: dropped on the desk, they rocked on their domes for 1 s after
a 3 s settle (up to 21 mm), on 3 and 2 of 5 drops. So did every kind the heap teacher stowed
under 9 of 12 times alone (``scripts/mj_part_table.py``, 2026-09-30): the arches 3455 (6) and
3307 (5), which offer no closing axis on their side, the fire helmet 3834 (8), the radio 3962b
(8) and the pickaxe 3841 (0). The camera 4360 did not build (a degenerate CoACD hull)."""
LDRAW_DIR = Path(__file__).resolve().parents[3] / "assets" / "parts"
FLAT_FAMILIES = ("plate", "tile")  # in the catalog only under ``--families flat``
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
_FAMILY_SPECS = {
    "brick": tuple((nx, ny, 1) for nx, ny in _BOX_FOOTPRINTS) + _TALL_BRICKS,
    "slope": tuple((nx, ny, 1) for nx, ny in _SLOPE_FOOTPRINTS),
    "round": ((1, 1, 1), (2, 2, 1), (2, 2, 2), (4, 4, 1)),
    "plate": ((1, 2, 1), (1, 4, 1), (1, 6, 1), (2, 2, 1), (2, 4, 1), (2, 6, 1)),
    "tile": ((1, 2, 1), (1, 4, 1), (2, 2, 1), (2, 4, 1)),
}
"""(nx, ny, nz) of each primitive family's parts, in catalog order (MS7e design sections 6.4
and 6.6): round is the four retired round bricks, and no 4-wide flat part enters."""
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
    hulls: tuple[tuple[float, tuple[tuple[float, float, float], ...]], ...] = ()
    """An LDraw part's convex hulls, part frame: (share of ``mass``, vertices) each."""
    stud_frames: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = ()
    """An LDraw part's visual studs: (base on the face, outward unit axis) each."""

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
    known = tuple(_FAMILY_SPECS)
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


@functools.cache
def ldraw_part(part_id: str, family: str = "ldraw") -> Part:
    """The ``family`` part ``part_id`` from its JSON file (module docstring). ``nx`` and ``ny``
    are its footprint in studs; its studs are ``stud_frames``, so ``studs`` is False."""
    rec = json.loads((LDRAW_DIR / f"{part_id}.json").read_text())
    size = tuple(rec["size"])
    total = sum(rec["hull_volumes"])
    return Part(
        name=f"{family}_{part_id}",
        family=family,
        nx=round(size[1] / STUD_PITCH),
        ny=round(size[0] / STUD_PITCH),
        size=size,
        studs=False,
        mass=part_mass(size),
        closing_axes=tuple(rec["closing_axes"]),
        hulls=tuple(
            (v / total, tuple(tuple(p) for p in h))
            for v, h in zip(rec["hull_volumes"], rec["hulls"])
        ),
        stud_frames=tuple((tuple(s["base"]), tuple(s["axis"])) for s in rec["studs"]),
    )


PART_CLASSES = ("slope", "tall", "narrow", "wide", "round", "special", "plate", "tile")
"""The eight sorting classes of MS7e (MS7e design section 5), in the order the class-to-box map
lists them. The first four are MS7a's (``BASE_CLASSES``), each with a held-out part."""
BASE_CLASSES = PART_CLASSES[:4]
"""The four MS7a classes: the active class list whenever every MS7e switch is off, so the class
map and the perception's class draws are MS7d's."""


def part_class(part: Part) -> str:
    """The sorting class of ``part``: its family for a round part, a plate and a tile,
    ``special`` for a part from ``LDRAW_DIR``, ``slope`` for a slope, ``tall`` for a brick
    taller than one brick height, ``narrow`` for a one-height brick one stud wide and ``wide``
    for a one-height brick two or more studs wide. The base catalog holds 6, 5, 8 and 10.

    The MS7f families join ``special``: a ninth class would reshuffle every MS7e class map."""
    if part.family in ("round", "plate", "tile"):
        return part.family
    if part.family in LDRAW_FAMILY_IDS:
        return "special"
    if part.family == "slope":
        return "slope"
    if part.size[2] > BRICK_HEIGHT + 1e-9:
        return "tall"
    return "narrow" if part.nx == 1 else "wide"


PART_CATEGORIES = ("brick", "special", "flat", "round", "joint", "minifig")
"""The six sort categories of MS7f (MS7f design section 6.1), in the order the category map
lists them. They replace the eight classes under ``--categories``."""
FAMILY_CATEGORY = {
    "brick": "brick",
    "slope": "special",
    "ldraw": "special",
    "plate": "flat",
    "tile": "flat",
    "round": "round",
    "joint": "joint",
    "minifig": "minifig",
}
"""The category of each family: a brick of any height or width is a brick, a slope and an LDraw
part are special, a plate and a tile are flat. ``joint`` and ``minifig`` are the families of
MS7f design section 7, which the parts track adds to ``FAMILIES``."""


def part_category(part: Part) -> str:
    """The sort category of ``part`` (``PART_CATEGORIES``), read off its family."""
    return FAMILY_CATEGORY[part.family]


def active_categories(families: Sequence[str] = ()) -> tuple[str, ...]:
    """The categories of a run whose catalog holds the ``FAMILY_KEYS`` keys ``families``
    (``catalog``): those of its families, in ``PART_CATEGORIES`` order. Brick and slope are in
    every catalog, so brick and special always are."""
    present = {"brick", "slope"} | {f for key in families for f in FAMILY_KEYS[key]}
    return tuple(c for c in PART_CATEGORIES if c in {FAMILY_CATEGORY[f] for f in present})


def stud_positions(part: Part) -> list[tuple[float, float]]:
    """(x, y) of every stud on the top face, part frame. Tiles and slopes have none.

    A round part keeps the studs whose centre lies on its disc, so a round 4x4 has 12, as
    LEGO's does. The four corner studs of the full grid stood 3.5 mm past its rim, and as
    collision studs (MS7e design section 6.4) they held the part off the desk like cogs."""
    if not part.studs:
        return []
    grid = [
        (STUD_PITCH * (j - (part.ny - 1) / 2), STUD_PITCH * (i - (part.nx - 1) / 2))
        for j in range(part.ny)
        for i in range(part.nx)
    ]
    if part.family == "round":
        return [(x, y) for x, y in grid if math.hypot(x, y) < part.size[0] / 2]
    return grid


def graspable(part: Part) -> bool:
    """The jaws close across the shorter footprint side; it must fit the opening."""
    return min(part.size[0], part.size[1]) <= MAX_GRASP_WIDTH


def _family_parts(family: str) -> list[Part]:
    """Every graspable part of one family, in catalog order."""
    made = (
        [ldraw_part(i, family) for i in LDRAW_FAMILY_IDS[family]]
        if family in LDRAW_FAMILY_IDS
        else [make_part(family, *spec) for spec in _FAMILY_SPECS[family]]
    )
    return [p for p in made if graspable(p)]


def catalog(families: Sequence[str] = ()) -> list[Part]:
    """Every generator part that passes the graspability filter, in a fixed order: brick and
    slope (the MS7d catalog, 29 parts), then the families the ``FAMILY_KEYS`` keys in
    ``families`` bring, in ``FAMILIES`` order."""
    unknown = sorted(set(families) - set(FAMILY_KEYS))
    if unknown:
        raise ValueError(f"unknown family keys {unknown}; keys: {tuple(FAMILY_KEYS)}")
    wanted = {"brick", "slope"} | {f for key in families for f in FAMILY_KEYS[key]}
    return [p for f in FAMILIES if f in wanted for p in _family_parts(f)]


def catalog_index(name: str) -> int:
    """The part's index in the full MS7e order: the MS7d catalog (0 to 28), then round, LDraw,
    plate and tile, then the MS7f families. A part from ``LDRAW_DIR`` needs no JSON for it. The
    lone-part table seeds a part's trials at 20289000 plus this index (MS7e design section
    8.4)."""
    names = []
    for f in FAMILIES:
        names += (
            [f"{f}_{i}" for i in LDRAW_FAMILY_IDS[f]]
            if f in LDRAW_FAMILY_IDS
            else [p.name for p in _family_parts(f)]
        )
    return names.index(name)


def extra_part(name: str) -> Part:
    """A ``CLASSIFIER_EXTRA`` kind from its name: an LDraw-built family's JSON part, or a
    primitive from its ``<family>_<nx>x<ny>[x<nz>]`` name."""
    family, spec = name.split("_", 1)
    if family in LDRAW_FAMILY_IDS:
        return ldraw_part(spec, family)
    return make_part(family, *(int(n) for n in spec.split("x")))


def _replayable() -> list[Part]:
    """The catalog plus the flat, retired and LDraw-built parts other shards were recorded
    with, and the classifier's extras."""
    flat = [make_part(f, nx, ny) for f in FLAT_FAMILIES for nx, ny in _BOX_FOOTPRINTS]
    retired = [make_part(*spec) for spec in _RETIRED]
    ldraw = [p for f in LDRAW_FAMILY_IDS for p in _family_parts(f)]
    extra = [extra_part(n) for n in CLASSIFIER_EXTRA]
    return catalog() + [p for p in flat if graspable(p)] + retired + ldraw + extra


def split(parts: Sequence[Part], seed: int) -> tuple[list[Part], list[Part]]:
    """(train, heldout): a seeded shuffle per family holds out ceil(20 %) of that family.

    Both lists keep the catalog order, so a run's ``parts`` metadata reads the same way
    whatever the seed. Stratified so every family has a held-out ladder row (design 5).

    Brick and slope shuffle on one generator, the MS6 split. Each MS7e family shuffles on its
    own generator, seeded from ``seed`` and its name, so its held-out parts do not depend on
    which other families join a run: a family adopted between two waves moves no part of a
    later family across the split (MS7e design section 5).
    """
    rng = random.Random(seed)
    order = {p.name: k for k, p in enumerate(parts)}
    train: list[Part] = []
    heldout: list[Part] = []
    for family in FAMILIES:
        members = [p for p in parts if p.family == family]
        own = rng if family in ("brick", "slope") else random.Random(f"{seed}:{family}")
        own.shuffle(members)
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
