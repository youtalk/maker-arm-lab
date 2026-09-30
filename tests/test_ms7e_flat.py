"""MS7e flat track (design sections 6.6 and 10.2): the flat parts enter only under
``--families flat``, no 4-wide flat part enters, and lever 1 (``--deep_box_floor``) changes
the desk box floors and nothing else."""

import argparse
import random

import numpy as np
import pytest

from maker_arm_lab.core import batch_loop, parts
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj import scene

FLAT = [
    *(f"plate_{s}" for s in ("1x2", "1x4", "1x6", "2x2", "2x4", "2x6")),
    *(f"tile_{s}" for s in ("1x2", "1x4", "2x2", "2x4")),
]


def _heap_args(*extra):
    parser = argparse.ArgumentParser()
    batch_loop.add_batch_arguments(parser)
    batch_loop.add_ms7e_arguments(parser)
    args = parser.parse_args(["--out_dir", "/tmp/x", "--heap_k", "50", *extra])
    batch_loop.check_ms7e(parser, args)
    return args


def test_the_flat_parts_enter_only_under_their_key_and_none_is_4_wide():
    assert "flat" in parts.READY_FAMILIES
    assert not [p for p in parts.catalog() if p.family in parts.FLAT_FAMILIES]
    flat = [p for p in parts.catalog(("flat",)) if p.family in parts.FLAT_FAMILIES]
    assert [p.name for p in flat] == FLAT
    assert max(max(p.nx, p.ny) for p in flat) <= 6 and min(p.nx for p in flat) == 1
    assert all(min(p.nx, p.ny) < 4 for p in flat)
    assert not [p for p in batch_loop.heap_parts(_heap_args()) if p.family in parts.FLAT_FAMILIES]
    heap = batch_loop.heap_parts(
        _heap_args("--families", "flat", "--heap_family", "flat", "--parts_split", "all")
    )
    assert sum(p.family in parts.FLAT_FAMILIES for p in heap) == 25


def test_the_deep_floor_changes_the_desk_box_floors_and_nothing_else():
    """Lever 1 moves no plan and no other geom: every geom but the four box floors is the
    same, and each floor keeps its top face."""
    draw = sample_run_draw(random.Random(5))
    heap = parts.catalog(("flat",))[-4:] + parts.catalog()[:4]
    kwargs = {
        "cameras_on": False,
        "parts": heap,
        "num_desk_boxes": 4,
        "tray": True,
        "tray_lean_deg": 30.0,
    }
    plain, _ = scene.build_model(draw, **kwargs)
    deep, _ = scene.build_model(draw, **kwargs, deep_box_floor=True)
    assert plain.ngeom == deep.ngeom
    changed = {
        plain.geom(g).name
        for g in range(plain.ngeom)
        if not (
            np.array_equal(plain.geom_size[g], deep.geom_size[g])
            and np.array_equal(plain.geom_pos[g], deep.geom_pos[g])
        )
    }
    assert changed == {f"desk_box_{j}_floor" for j in range(4)}
    for name in changed:
        top = [
            m.geom_pos[m.geom(name).id][2] + m.geom_size[m.geom(name).id][2] for m in (plain, deep)
        ]
        assert np.isclose(*top)


def test_the_deep_floor_needs_desk_boxes():
    with pytest.raises(SystemExit):
        _heap_args("--deep_box_floor")
    assert _heap_args("--deep_box_floor", "--desk_boxes", "4").deep_box_floor
