"""The MS6 part catalog: exact LEGO-like primitives, torch-free (design section 5)."""

import math
import random

import pytest

from maker_arm_lab.core import layout, parts


def test_catalog_has_62_parts_in_a_fixed_order():
    names = [p.name for p in parts.catalog()]
    assert len(names) == 62 and len(set(names)) == 62
    assert names[0] == "brick_1x1" and names[-1] == "slope_2x4"
    assert names.count("plate_2x4") == 1 and "tile_4x8" in names


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
    assert len(held) == 14 and len(train) == 48
    counts = {f: sum(1 for p in held if p.family == f) for f in parts.FAMILIES}
    assert counts == {"brick": 4, "plate": 4, "tile": 4, "slope": 2}
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


def test_sample_parts_draws_with_replacement():
    pool = parts.catalog()[:3]
    drawn = parts.sample_parts(random.Random(0), pool, 10)
    assert len(drawn) == 10 and all(p in pool for p in drawn)


def test_by_name_round_trips_the_catalog():
    cat = parts.catalog()
    assert parts.by_name([p.name for p in cat]) == cat
    with pytest.raises(KeyError):
        parts.by_name(["brick_9x9"])
