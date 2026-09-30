"""The MS6 part catalog: exact LEGO-like primitives, torch-free (design section 5)."""

import math
import random
from collections import Counter

import pytest

from maker_arm_lab.core import layout, parts


def test_catalog_has_29_parts_in_a_fixed_order_and_no_flat_or_round_part():
    names = [p.name for p in parts.catalog()]
    assert len(names) == 29 and len(set(names)) == 29
    assert names[0] == "brick_1x1" and names[-1] == "slope_2x4"
    assert "brick_1x2x5" in names and "slope_1x2" in names
    assert not [n for n in names if n.startswith(parts.FLAT_FAMILIES + parts.RETIRED_FAMILIES)]
    assert parts.FAMILIES[:7] == ("brick", "slope", "round", "ldraw", "plate", "tile", "joint")
    assert parts.FAMILIES[7:] == ("minifig",)
    assert [p.name for p in parts.catalog(())] == names


def test_the_retired_parts_left_the_catalog_and_still_replay():
    """The 1x1 round lifted 15 of 44 and the two pure wedges 26 of 47 on the rA read; the
    2x2 rounds rolled (21 of 43 lifts, two off the desk) on the b3 read."""
    names = {p.name for p in parts.catalog()}
    assert not names & {"round_1x1", "round_2x2", "round_2x2x2", "round_4x4"}
    assert not names & {"slope_1x1", "slope_2x1"}
    assert parts.by_name(["round_1x1", "round_2x2x2", "slope_2x1"]) == [
        parts.make_part("round", 1, 1),
        parts.make_part("round", 2, 2, 2),
        parts.make_part("slope", 2, 1),
    ]
    with pytest.raises(ValueError):
        parts.make_part("cone", 1, 1)


def test_a_slope_never_closes_along_its_run():
    """y across the side faces, z between the flat bottom and the flat top, never x."""
    assert parts.make_part("slope", 2, 4).closing_axes == parts.SLOPE_AXES == (1, 2)
    assert parts.make_part("brick", 2, 4).closing_axes == parts.BOX_AXES == (0, 1, 2)
    assert parts.make_part("round", 2, 2).closing_axes == parts.BOX_AXES


def test_tall_and_round_bricks_have_the_lego_sizes():
    tall = parts.make_part("brick", 1, 2, 3)
    assert tall.name == "brick_1x2x3" and tall.family == "brick"
    assert tall.size == pytest.approx((0.0158, 0.0078, 3 * parts.BRICK_HEIGHT))
    round_ = parts.make_part("round", 2, 2)
    assert round_.size == pytest.approx((0.0158, 0.0158, parts.BRICK_HEIGHT))
    assert round_.studs and round_.hull is None
    box = parts.make_part("brick", 2, 2)
    assert round_.mass == pytest.approx(box.mass * math.pi / 4)


def test_the_2x4_brick_is_todays_brick():
    p = parts.make_part("brick", 2, 4)
    assert p.size == pytest.approx(layout.BRICK_SIZE)
    assert p.mass == pytest.approx(layout.BRICK_MASS, rel=0.02)
    assert p.studs and p.hull is None
    assert p.half_diagonal == pytest.approx(0.5 * math.hypot(*layout.BRICK_SIZE))


def test_masses_follow_the_hollow_abs_rule_with_a_floor():
    plate = parts.make_part("plate", 1, 1)
    assert plate.mass == parts.MASS_FLOOR_KG
    big = parts.make_part("brick", 4, 8)
    volume = big.size[0] * big.size[1] * big.size[2]
    assert big.mass == pytest.approx(volume * parts.ABS_DENSITY_KG_M3 * parts.HOLLOW_FRACTION)


def test_slope_hull_is_a_wedge_or_a_wedge_with_a_flat_top():
    wedge = parts.make_part("slope", 1, 1)
    flat = parts.make_part("slope", 1, 2)
    assert len(wedge.hull) == 6 and len(flat.hull) == 8
    for part in (wedge, flat):
        sx, _sy, h = part.size
        assert all(abs(v[2]) == pytest.approx(h / 2) for v in part.hull)
        assert max(v[0] for v in part.hull if v[2] > 0) == pytest.approx(sx / 2)
    top_front = min(v[0] for v in flat.hull if v[2] > 0)
    assert top_front == pytest.approx(flat.size[0] / 2 - (flat.size[0] - parts.BRICK_HEIGHT))


def test_graspability_filter_drops_a_six_by_six():
    assert not parts.graspable(parts.make_part("brick", 6, 6))
    assert all(parts.graspable(p) for p in parts.catalog())


def test_split_is_stratified_deterministic_and_disjoint():
    cat = parts.catalog()
    train, held = parts.split(cat, 20260915)
    assert len(held) == 7 and len(train) == 22
    counts = {f: sum(1 for p in held if p.family == f) for f in ("brick", "slope")}
    assert counts == {"brick": 5, "slope": 2}
    assert not {p.name for p in train} & {p.name for p in held}
    again = parts.split(cat, 20260915)
    assert [p.name for p in again[1]] == [p.name for p in held]
    assert {p.name for p in parts.split(cat, 1)[1]} != {p.name for p in held}
    order = {p.name: k for k, p in enumerate(cat)}
    assert [order[p.name] for p in train] == sorted(order[p.name] for p in train)


def test_stud_positions_lie_inside_the_footprint():
    brick = parts.make_part("brick", 2, 4)
    studs = parts.stud_positions(brick)
    assert len(studs) == 8
    assert all(abs(x) <= 0.012 and abs(y) <= 0.004 for x, y in studs)
    assert parts.stud_positions(parts.make_part("tile", 2, 4)) == []
    assert parts.stud_positions(parts.make_part("slope", 2, 4)) == []


def test_a_round_part_keeps_the_studs_on_its_disc():
    """A round 4x4 has 12 studs, as LEGO's: the grid's corners stood past its rim."""
    counts = {p.name: len(parts.stud_positions(p)) for p in parts._replayable()}
    assert [counts[f"round_{n}"] for n in ("1x1", "2x2", "2x2x2", "4x4")] == [1, 4, 4, 12]
    for p in parts._replayable():
        if p.family != "round" and p.studs:
            assert counts[p.name] == p.nx * p.ny, p.name


def test_sample_parts_draws_with_replacement():
    pool = parts.catalog()[:3]
    drawn = parts.sample_parts(random.Random(0), pool, 10)
    assert len(drawn) == 10 and all(p in pool for p in drawn)


def test_by_name_round_trips_the_catalog_and_still_replays_a_flat_part():
    cat = parts.catalog()
    assert parts.by_name([p.name for p in cat]) == cat
    assert parts.by_name(["plate_2x4"]) == [parts.make_part("plate", 2, 4)]
    with pytest.raises(KeyError):
        parts.by_name(["brick_9x9"])


def test_every_catalog_part_has_exactly_one_class():
    classes = Counter(parts.part_class(p) for p in parts.catalog())
    assert classes == {"slope": 6, "tall": 5, "narrow": 8, "wide": 10}
    assert set(classes) == set(parts.BASE_CLASSES) == set(parts.PART_CLASSES[:4])
    assert parts.part_class(parts.make_part("brick", 1, 2, 5)) == "tall"
    assert parts.part_class(parts.make_part("brick", 1, 8)) == "narrow"
    assert parts.part_class(parts.make_part("brick", 2, 2)) == "wide"
    assert parts.part_class(parts.make_part("slope", 2, 4)) == "slope"


def test_every_class_holds_a_held_out_part():
    _, heldout = parts.split(parts.catalog(), 20260915)
    by_class = {
        c: sorted(p.name for p in heldout if parts.part_class(p) == c) for c in parts.BASE_CLASSES
    }
    assert by_class == {
        "slope": ["slope_1x2", "slope_1x3"],
        "tall": ["brick_1x2x2", "brick_1x2x5"],
        "narrow": ["brick_1x6"],
        "wide": ["brick_2x3", "brick_2x7"],
    }
