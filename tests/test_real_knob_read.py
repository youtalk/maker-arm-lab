import math

import pytest

from maker_arm_lab.core import parts as parts_mod
from maker_arm_lab.real.knob_read import (
    FrameResult,
    fold_angle,
    heading_period,
    knob_table,
    match_parts,
    verdict,
)
from maker_arm_lab.real.layouts import Layout, TruthBox, TruthPart

BOXES = [TruthBox(0.1, 0.2, 0.0), TruthBox(0.1, -0.2, 0.0), TruthBox(0.2, 0.2, 0.0),
         TruthBox(0.2, -0.2, 0.0)]  # fmt: skip


def _part(name, x, y, yaw):
    p = parts_mod.by_name([name])[0]
    return TruthPart(name, parts_mod.part_class(p), x, y, 0.005, yaw)


def _layout(kind="sparse"):
    parts = [_part("brick_2x4", 0.30, 0.0, 0.3), _part("brick_2x2", 0.35, 0.05, 0.1)]
    return Layout("l", kind, parts, BOXES)


def _det(p, dx=0.0, dy=0.0):
    s = parts_mod.by_name([p.name])[0].size
    return {
        "centre": [p.x + dx, p.y + dy, p.z],
        "quat": [math.cos(p.yaw / 2), 0, 0, math.sin(p.yaw / 2)],
        "size": [max(s[:2]), min(s[:2]), s[2]],
        "cls": p.cls,
        "score": 1.0,
    }


def _poses():
    return [[b.x, b.y, b.yaw] for b in BOXES]


def _frame(lay, dx=0.0):
    return FrameResult(lay, [_det(p, dx) for p in lay.parts], _poses())


def test_perfect_detections_pass_with_zero_errors():
    table = knob_table([_frame(_layout())])
    assert verdict(table) == (True, [])
    assert table["miss"] == 0.0 and table["class"] == 0.0
    assert table["heading_deg"] == pytest.approx(0.0, abs=1e-6)


def test_a_seven_mm_bias_in_x_fails_its_row():
    ok, rows = verdict(knob_table([_frame(_layout(), dx=0.007)]))
    assert not ok and rows == ["pos_bias_mm.x"]


def test_two_touching_parts_merged_into_one_detection_count_one_miss():
    a, b = _layout().parts
    merged = _det(a, dx=0.003)
    pairs, missed, extra = match_parts([merged], [a, b])
    assert len(pairs) == 1 and len(missed) == 1 and extra == []


def test_a_frame_with_no_detection_reads_a_miss_of_one():
    table = knob_table([FrameResult(_layout(), [], [])])
    assert table["miss"] == 1.0 and math.isnan(table["heading_deg"])


def test_a_square_part_folds_its_heading_to_a_quarter_turn():
    err = fold_angle(math.radians(85), heading_period("brick_2x2"))
    assert err == pytest.approx(math.radians(-5))
    assert heading_period("brick_2x4") == pytest.approx(math.pi)


def test_heap_frames_stay_out_of_the_table():
    heap = Layout("h", "heap", [_part("brick_2x2", math.nan, math.nan, math.nan)] * 3, BOXES)
    table = knob_table([FrameResult(heap, [_det(_part("brick_2x2", 0, 0, 0))] * 2, _poses())])
    assert table["counts"]["heap"] == {"detected": 2, "poured": 3}
    assert table["counts"]["parts"] == 0 and math.isnan(table["miss"])


def test_a_wrong_class_and_a_box_offset_show_in_their_rows():
    lay = _layout()
    f = _frame(lay)
    f.detections[0]["cls"] = "other"
    f.boxes[0][0] += 0.02
    table = knob_table([f])
    assert table["class"] == 0.5
    assert table["box_mm"] == pytest.approx(10.0)
    assert "box_mm" not in verdict(table)[1]
