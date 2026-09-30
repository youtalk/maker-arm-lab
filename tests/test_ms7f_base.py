"""The MS7f base branch (MS7f design sections 4, 6.1, 6.2, 8.1 and 12.2): the six categories,
the category map, the grader's category boxes, the payload's throughput and held-out rows, the
S0 cycle intervals and the switch plumbing. With every MS7f switch off, every draw, map and
payload field equals MS7e's: the pinned values were computed from 29a36ef before the change."""

import random
from collections import Counter

import pytest
import torch

from maker_arm_lab.core import checker, desk_boxes, parts, perception

FAMILIES = ("round", "ldraw", "flat")


def test_every_catalog_kind_has_one_category():
    everything = parts.catalog(tuple(parts.FAMILY_KEYS))
    categories = Counter(parts.part_category(p) for p in everything)
    assert categories == {"brick": 23, "special": 14, "flat": 10, "round": 4}
    assert set(parts.FAMILY_CATEGORY) >= set(parts.FAMILIES)
    assert set(parts.FAMILY_CATEGORY.values()) == set(parts.PART_CATEGORIES)
    tall = parts.make_part("brick", 1, 2, 5)
    assert parts.part_category(tall) == "brick" and parts.part_class(tall) == "tall"


def test_the_active_categories_are_the_catalogs_in_table_order():
    assert parts.active_categories() == ("brick", "special")
    assert parts.active_categories(("flat", "round")) == ("brick", "special", "flat", "round")
    assert parts.active_categories(("ldraw",)) == ("brick", "special")
    for keys in (("round",), ("round", "ldraw", "flat")):
        present = {parts.part_category(p) for p in parts.catalog(keys)}
        assert set(parts.active_categories(keys)) == present


CATS = ("brick", "special", "flat", "round")


SIZES8 = [(0.08, 0.08, 0.04)] * 8


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 9])
def test_the_category_map_reproduces_and_maps_every_category(n):
    volumes = [4e-5, 1e-5, 3e-5, 2e-5]
    sizes = [(0.1, 0.1, 0.05)] * n
    got = desk_boxes.category_map(random.Random(3), sizes, CATS, volumes)
    assert got == desk_boxes.category_map(random.Random(3), sizes, CATS, volumes)
    assert len(got) == 4 and all(own for own in got)
    used = [b for own in got for b in own]
    assert set(used) == set(range(n)) if n >= 4 else set(used) <= set(range(n))
    if n >= 4:
        assert sorted(used) == list(range(n))  # every box once
    else:
        assert all(len(own) == 1 for own in got)  # R11's fold: several categories to a box


def test_the_second_boxes_go_in_volume_order_largest_first():
    volumes = [2e-5, 1e-5, 4e-5, 3e-5]  # flat, round, brick, special by volume
    for n, counts in ((5, [1, 1, 2, 1]), (6, [1, 1, 2, 2]), (7, [2, 1, 2, 2]), (8, [2, 2, 2, 2])):
        got = desk_boxes.category_map(random.Random(n), [(0.1, 0.1, 0.05)] * n, CATS, volumes)
        assert [len(own) for own in got] == counts, (n, got)
    # a tie keeps the category order; round again past one second box each
    got = desk_boxes.category_map(random.Random(1), [(0.1, 0.1, 0.05)] * 7, CATS[:2], [1, 1])
    assert [len(own) for own in got] == [4, 3]


def test_the_category_map_fits_the_boxes_to_the_expected_volume():
    """R11's floor: a big category's boxes are the big ones when a draw fits."""
    sizes = [(0.07, 0.07, 0.03)] * 4 + [(0.1, 0.1, 0.05)]
    volumes = [2.0e-4, 1e-5, 1e-5, 1e-5]  # the brick load fits only a 0.1 m box and a second
    got = desk_boxes.category_map(random.Random(5), sizes, CATS, volumes)
    assert 4 in got[0] and len(got[0]) == 2
    fold = desk_boxes.category_map(random.Random(5), sizes[:3], CATS, volumes)
    assert [len(own) for own in fold] == [1, 1, 1, 1]


def test_the_lighter_box_takes_the_least_volume_and_a_tie_the_lower_index():
    assert desk_boxes.lighter_box((5, 2), [0.0] * 8) == 2
    assert desk_boxes.lighter_box((2, 5), [0, 0, 3e-6, 0, 0, 1e-6, 0, 0]) == 5
    assert desk_boxes.lighter_box((3,), [9.0] * 4) == 3


def test_a_part_at_rest_in_any_box_of_its_category_grades_correct():
    pos = torch.zeros(1, 2, 3)
    boxes = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    half = torch.tensor([0.05, 0.05, 0.05])
    pos[0, 1, 0] = 0.5
    allowed = torch.tensor([[[True, False, True], [True, False, True]]])
    ok = checker.parts_at_rest_in_box_sets(pos, torch.zeros(1, 2, 3), boxes, half, allowed)
    assert ok.tolist() == [[True, False]]
    moving = torch.full((1, 2, 3), 0.1)
    assert not checker.parts_at_rest_in_box_sets(pos, moving, boxes, half, allowed).any()


def test_the_stand_in_errs_among_the_active_categories_on_the_same_draws():
    """Knob rows stay paired (design section 16): a classification draws twice whatever the
    class list, so the classes and the categories fire on the same calls."""
    cats = parts.active_categories(FAMILIES)
    a = perception.Perceiver(7, 2, {"class": 20.0}, parts.PART_CLASSES)
    b = perception.Perceiver(7, 2, {"class": 20.0}, cats)
    fired = []
    for true_cls, true_cat in [("wide", "brick"), ("slope", "special"), ("tile", "flat")] * 20:
        fired.append((a.classify(true_cls) != true_cls, b.classify(true_cat) != true_cat))
    assert all(x == y for x, y in fired) and any(x for x, _ in fired)
    assert a.rng.getstate() == b.rng.getstate()
    wrong = {b.classify("round") for _ in range(400)} - {"round"}
    assert wrong == set(cats) - {"round"}
    v = perception.ClassViews(7, 2, {"class": 1000.0, "rho": 0.0}, 1, cats)
    assert {v.view(0, "flat") for _ in range(200)} == set(cats) - {"flat"}
