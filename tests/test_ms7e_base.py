"""The MS7e base branch (MS7e design sections 5, 6.1 and 10.2): the family catalog, the split,
the eight classes, the class map, the family heap draw and the switches. With every MS7e
switch off, each draw equals MS7d's: the pinned values were computed from main c728747 before
the change."""

import argparse
import importlib.util
import itertools
import math
import random
from collections import Counter
from pathlib import Path

import pytest

from maker_arm_lab.core import batch_loop, desk_boxes, parts, perception

BATCH = Path(__file__).resolve().parents[1] / "scripts" / "mj_generate_batch.py"

MS7D_CATALOG = [
    *(f"brick_1x{n}" for n in range(1, 9)),
    *(f"brick_2x{n}" for n in range(2, 9)),
    "brick_4x4",
    "brick_4x6",
    "brick_4x8",
    "brick_1x1x3",
    "brick_1x2x2",
    "brick_1x2x3",
    "brick_1x2x5",
    "brick_2x2x3",
    *(f"slope_{nx}x{ny}" for nx in (1, 2) for ny in (2, 3, 4)),
]
MS7D_HELDOUT = [
    "brick_1x6",
    "brick_2x3",
    "brick_2x7",
    "brick_1x2x2",
    "brick_1x2x5",
    "slope_1x2",
    "slope_1x3",
]
# heap_parts on main: K 50, the held-out split, layout seed 20285000 (the first wave 1 run).
MS7D_HEAP = [
    "brick_1x2x5",
    "slope_1x3",
    "brick_2x3",
    "brick_1x6",
    "brick_1x6",
    "brick_2x3",
    "slope_1x3",
    "brick_1x2x2",
    "brick_2x3",
    "brick_2x3",
    "brick_2x3",
    "brick_2x3",
    "brick_2x3",
    "slope_1x3",
    "slope_1x3",
    "brick_2x7",
    "slope_1x3",
    "brick_2x7",
    "brick_2x3",
    "slope_1x2",
    "slope_1x3",
    "brick_1x2x5",
    "brick_2x7",
    "brick_1x2x5",
    "slope_1x2",
    "brick_1x6",
    "slope_1x2",
    "slope_1x3",
    "slope_1x3",
    "brick_1x2x2",
    "slope_1x3",
    "brick_2x3",
    "brick_1x6",
    "brick_1x2x5",
    "slope_1x3",
    "brick_1x2x2",
    "slope_1x2",
    "brick_1x2x2",
    "brick_1x2x2",
    "brick_1x2x2",
    "brick_2x7",
    "brick_2x3",
    "brick_1x6",
    "brick_1x2x5",
    "slope_1x2",
    "slope_1x3",
    "brick_2x3",
    "brick_1x2x5",
    "brick_1x2x5",
    "slope_1x2",
]


def _args(**kw):
    base = {"heap_k": 50, "parts_seed": 20260915, "parts_split": "heldout"}
    return argparse.Namespace(**{**base, "layout_seed": 20285000, **kw})


def test_with_every_switch_off_the_catalog_split_heap_and_classes_are_ms7ds():
    assert [p.name for p in parts.catalog()] == MS7D_CATALOG
    assert [p.name for p in parts.split(parts.catalog(), 20260915)[1]] == MS7D_HELDOUT
    assert [p.name for p in batch_loop.heap_parts(_args())] == MS7D_HEAP
    assert batch_loop.heap_classes(_args()) == parts.BASE_CLASSES == parts.PART_CLASSES[:4]
    assert batch_loop.ms7e_switches(argparse.Namespace()) == {
        "families": [],
        "box_draw": 0,
        "release_lean": False,
        "carry_via": False,
        "box_unseen": None,
        "heap_family": None,
        "passenger_check": False,
        "wall_band_cont": False,
        "thin_snap": False,
    }


@pytest.mark.parametrize(
    "seed, maps, after",
    [
        (
            0,
            [(2, 0, 1, 3), (0, 1, 3, 2), (0, 2, 1, 0), (0, 0, 0, 0), (1, 0, 1, 0)],
            0.5046868558173903,
        ),
        (
            1,
            [(3, 0, 2, 1), (3, 1, 0, 2), (0, 2, 1, 0), (0, 0, 0, 0), (0, 1, 0, 1)],
            0.02834747652200631,
        ),
        (
            20290000,
            [(0, 3, 1, 2), (3, 0, 1, 2), (0, 1, 0, 2), (0, 0, 0, 0), (1, 0, 0, 1)],
            0.21487768479953984,
        ),
    ],
)
def test_the_class_map_draws_as_ms7d_with_the_base_classes(seed, maps, after):
    rng = random.Random(seed)
    assert [desk_boxes.class_map(rng, n) for n in (4, 4, 3, 1, 2)] == maps
    assert rng.random() == after


def test_the_perceiver_draws_as_ms7d_with_the_base_classes():
    p = perception.Perceiver(7, 2, {"class": 20.0})
    got = [p.classify(c) for c in ("slope", "tall", "narrow", "wide") * 4]
    assert got == [
        "slope",
        "tall",
        "wide",
        "wide",
        "slope",
        "slope",
        "narrow",
        "wide",
        "wide",
        "slope",
        "wide",
        "wide",
        "wide",
        "slope",
        "tall",
        "wide",
    ]
    assert p.rng.random() == 0.6935598674413352
    v = perception.ClassViews(7, 2, {"class": 20.0, "rho": 3.0}, 6)
    views = [
        v.view(b, c) for b, c in enumerate(("slope", "tall", "narrow", "wide", "slope", "tall"))
    ]
    assert views == ["slope", "tall", "narrow", "narrow", "wide", "tall"]
    assert v.rng.random() == 0.9162593913495918


def test_every_family_combination_keeps_the_seven_base_heldout_parts():
    for n in range(len(parts.FAMILY_KEYS) + 1):
        for keys in itertools.combinations(parts.FAMILY_KEYS, n):
            _, held = parts.split(parts.catalog(keys), 20260915)
            assert [p.name for p in held if p.family in ("brick", "slope")] == MS7D_HELDOUT, keys


def test_each_new_family_holds_out_a_fifth_rounded_up_whatever_else_joins():
    everything = parts.catalog(tuple(parts.FAMILY_KEYS))
    sizes = Counter(p.family for p in everything)
    assert sizes == {"brick": 23, "slope": 6, "round": 4, "ldraw": 8, "plate": 6, "tile": 4}
    _, held = parts.split(everything, 20260915)
    counts = Counter(p.family for p in held)
    for family in ("round", "ldraw", "plate", "tile"):
        assert counts[family] == math.ceil(0.2 * sizes[family])
    assert [counts[f] for f in ("round", "ldraw", "plate", "tile")] == [1, 2, 2, 1]
    # Each new family shuffles on its own generator: a family adopted between two waves moves
    # no part of another across the split.
    for keys in (("flat",), ("round", "flat"), ("ldraw", "flat")):
        _, alone = parts.split(parts.catalog(keys), 20260915)
        flat = [p.name for p in alone if p.family in parts.FLAT_FAMILIES]
        assert flat == [p.name for p in held if p.family in parts.FLAT_FAMILIES], keys


def test_the_eight_class_table_classes_every_part():
    everything = parts.catalog(tuple(parts.FAMILY_KEYS))
    classes = Counter(parts.part_class(p) for p in everything)
    assert classes == {
        "slope": 6,
        "tall": 5,
        "narrow": 8,
        "wide": 10,
        "round": 4,
        "special": 8,
        "plate": 6,
        "tile": 4,
    }
    assert set(classes) == set(parts.PART_CLASSES) and len(parts.PART_CLASSES) == 8
    assert parts.part_class(parts.make_part("round", 2, 2, 2)) == "round"  # tall, but round


def test_the_catalog_appends_the_families_in_order_and_the_index_pins_the_full_order():
    added = [p.name for p in parts.catalog(("flat", "round"))[29:]]
    assert added == [
        "round_1x1",
        "round_2x2",
        "round_2x2x2",
        "round_4x4",
        *(f"plate_{s}" for s in ("1x2", "1x4", "1x6", "2x2", "2x4", "2x6")),
        *(f"tile_{s}" for s in ("1x2", "1x4", "2x2", "2x4")),
    ]
    assert not [p for p in parts.catalog(("flat",)) if p.family in parts.FLAT_FAMILIES and p.nx > 2]
    assert parts.by_name(added) == parts.catalog(("round", "flat"))[29:]
    pins = {"brick_1x1": 0, "slope_2x4": 28, "round_1x1": 29, "round_4x4": 32}
    pins |= {"ldraw_3700": 33, "ldraw_3660": 40, "plate_1x2": 41, "tile_1x2": 47, "tile_2x4": 50}
    assert {name: parts.catalog_index(name) for name in pins} == pins
    for k, name in enumerate(MS7D_CATALOG):
        assert parts.catalog_index(name) == k
    with pytest.raises(ValueError, match="unknown family"):
        parts.catalog(("cone",))
    assert [p.name for p in parts.catalog(("ldraw",))[29:]] == [
        f"ldraw_{i}" for i in parts.LDRAW_IDS
    ]
    with pytest.raises(ValueError):
        parts.make_part("ldraw", 1, 2)


def test_the_eight_classes_shuffle_over_the_boxes_and_fold():
    rng = random.Random(4)
    for n in (4, 5, 6, 7, 8):
        m = desk_boxes.class_map(rng, n, parts.PART_CLASSES)
        assert len(m) == 8 and sorted(Counter(m).values()) == sorted(
            Counter(b % n for b in range(8)).values()
        )
    state = rng.getstate()
    assert desk_boxes.class_map(rng, 1, parts.PART_CLASSES) == (0,) * 8
    assert rng.getstate() == state
    heap = [parts.make_part("round", 1, 1), parts.make_part("tile", 2, 4)]
    assert desk_boxes.part_destinations(heap, range(8), parts.PART_CLASSES) == [4, 7]
    with pytest.raises(ValueError):  # a round part has no box among the base classes
        desk_boxes.part_destinations(heap, range(4))


def test_with_the_eight_classes_a_wrong_class_is_any_of_the_seven_others():
    p = perception.Perceiver(7, 2, {"class": 1000.0}, parts.PART_CLASSES)
    wrong = {p.classify("round") for _ in range(400)}
    assert wrong == set(parts.PART_CLASSES) - {"round"}
    v = perception.ClassViews(7, 2, {"class": 1000.0, "rho": 0.0}, 1, parts.PART_CLASSES)
    assert {v.view(0, "tile") for _ in range(400)} == set(parts.PART_CLASSES) - {"tile"}


def test_the_heap_family_draws_half_the_heap_from_the_family_and_shuffles_it():
    for k in (50, 51):
        args = _args(heap_k=k, parts_split="all", families=("round",), heap_family="round")
        heap = batch_loop.heap_parts(args)
        own = [p.family == "round" for p in heap]
        assert len(heap) == k and sum(own) == k // 2
        assert own != sorted(own, reverse=True)  # not all at the bottom of the heap
        assert {p.family for p in heap} == {"brick", "slope", "round"}
    flat = batch_loop.heap_parts(_args(families=("round", "flat"), heap_family="flat"))
    assert sum(p.family in parts.FLAT_FAMILIES for p in flat) == 25
    assert {p.name for p in flat if p.family in parts.FLAT_FAMILIES} <= {
        "plate_1x4",
        "plate_2x2",
        "tile_1x4",
    }  # the held-out flat parts
    assert "round" in {p.family for p in flat}  # the rest holds the other families too


def test_any_family_or_the_box_draw_turns_on_the_eight_classes():
    assert batch_loop.heap_classes(_args(families=("round",))) == parts.PART_CLASSES
    assert batch_loop.heap_classes(_args(box_draw=6)) == parts.PART_CLASSES
    assert batch_loop.heap_classes(_args(release_lean=True)) == parts.BASE_CLASSES


def _load_batch():
    spec = importlib.util.spec_from_file_location("mj_generate_batch", BATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEAP = ["--out_dir", "/tmp/x", "--heap_k", "3"]
UNBUILT = [
    ["--families", "round"],
    ["--families", "ldraw"],
    ["--families", "flat"],
    ["--families", "cone"],
    ["--box_draw", "6"],
    ["--release_lean"],
    ["--box_unseen", "size"],
    ["--box_unseen", "wide"],
    ["--heap_family", "round"],
]


@pytest.mark.parametrize("argv", UNBUILT)
def test_the_recorder_refuses_an_unbuilt_ms7e_switch_with_its_fail_token(argv, capsys, monkeypatch):
    """The base commit's state, whatever the tracks have built since."""
    monkeypatch.setattr(parts, "READY_FAMILIES", frozenset())
    monkeypatch.setattr(batch_loop, "UNBUILT_SWITCHES", ("box_draw", "release_lean", "box_unseen"))
    batch = _load_batch()
    with pytest.raises(SystemExit) as exit_info:
        batch.main(HEAP + argv)
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("BATCH_FAIL usage: "), out


def test_the_run_meta_records_the_ms7e_switches(monkeypatch):
    batch = _load_batch()
    args = batch.parse_args(HEAP)
    draw, _, noise = batch.build_run(args)
    meta = batch.run_meta(
        args, draw, noise, None, num_bricks=3, export_mode="EXPORT_ALL", backend="mujoco", task="t"
    )
    assert meta["ms7e"] == batch_loop.ms7e_switches(argparse.Namespace())
    monkeypatch.setattr(parts, "READY_FAMILIES", frozenset({"round", "flat"}))
    args = batch.parse_args(HEAP + ["--families", "flat,round", "--heap_family", "flat"])
    assert args.families == ("round", "flat")
    meta = batch.run_meta(
        args, draw, noise, None, num_bricks=3, export_mode="EXPORT_ALL", backend="mujoco", task="t"
    )
    assert meta["ms7e"]["families"] == ["round", "flat"] and meta["ms7e"]["heap_family"] == "flat"
