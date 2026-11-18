import math

import pytest

from maker_arm_lab.core import parts as parts_mod
from maker_arm_lab.real.knob_read import (
    FrameResult,
    fold_angle,
    heading_period,
    knob_table,
    match_boxes,
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
    assert "heading_deg" in verdict(table)[1]


def test_a_square_part_folds_its_heading_to_a_quarter_turn():
    err = fold_angle(math.radians(85), heading_period("brick_2x2"))
    assert err == pytest.approx(math.radians(-5))
    assert heading_period("brick_2x4") == pytest.approx(math.pi)


def test_heap_frames_stay_out_of_the_table():
    heap = Layout("h", "heap", [_part("brick_2x2", math.nan, math.nan, math.nan)] * 3, BOXES)
    table = knob_table([FrameResult(heap, [_det(_part("brick_2x2", 0, 0, 0))] * 2, _poses())])
    assert table["counts"]["heap"] == {"detected": 2, "poured": 3}
    assert table["counts"]["parts"] == 0 and math.isnan(table["miss"])
    assert not verdict(table)[0]


def test_a_wrong_class_and_a_box_offset_show_in_their_rows():
    lay = _layout()
    f = _frame(lay)
    f.detections[0]["cls"] = "other"
    f.boxes[0][0] += 0.02
    table = knob_table([f])
    assert table["class"] == 0.5
    assert table["box_mm"] == pytest.approx(10.0)
    assert "box_mm" not in verdict(table)[1]


def test_a_read_that_measured_nothing_fails_every_row():
    ok, rows = verdict(knob_table([]))
    assert not ok and "miss" in rows and "box_miss" in rows and "pos_bias_mm.x" in rows


def test_a_frame_with_no_boxes_fails_its_box_rows():
    f = _frame(_layout())
    f.boxes = []
    ok, rows = verdict(knob_table([f]))
    assert not ok and {"box_mm", "box_deg", "box_miss"} <= set(rows)


def test_three_of_four_boxes_found_reads_a_quarter_box_miss():
    f = _frame(_layout())
    f.boxes = f.boxes[:3]
    table = knob_table([f])
    assert table["box_miss"] == 0.25 and table["box_mm"] == pytest.approx(0.0)
    assert table["counts"]["box_miss"] == 1


def test_a_fifth_pose_counts_one_extra_box():
    f = _frame(_layout())
    f.boxes.append([0.5, 0.5, 0.0])
    pairs, missed, extra = match_boxes(f.boxes, BOXES)
    assert len(pairs) == 4 and missed == [] and len(extra) == 1
    assert knob_table([f])["counts"]["box_extra"] == 1


def test_a_square_box_found_a_quarter_turn_off_reads_no_heading_error():
    f = _frame(_layout())
    f.boxes[0][2] += math.pi / 2
    assert knob_table([f])["box_deg"] == pytest.approx(0.0, abs=1e-6)
    assert knob_table([f], box_period=math.pi)["box_deg"] > 40
