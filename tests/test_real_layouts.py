import itertools
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers import load_script
from maker_arm_lab.core import desk_boxes
from maker_arm_lab.real.camera_model import Pose
from maker_arm_lab.real.layouts import (
    CLUSTER_GAP_M,
    IPAD_AIR_11,
    SPARSE_MIN_FOOTPRINT_GAP_M,
    SPARSE_MIN_GAP_M,
    TRAY_EDGE_MARGIN_M,
    Layout,
    TruthPart,
    draw_layout,
    footprint_gap,
    footprint_inside_tray,
    nominal_screen_pose,
    place_layout,
    template_png,
)
from maker_arm_lab.real.rig import RigConfig, label_boxes

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["brick_1x1", "brick_2x4", "brick_4x8", "brick_1x1x3", "slope_1x2", "slope_2x4"]


def test_a_sparse_layout_keeps_its_parts_apart_and_inside_the_tray():
    rig = RigConfig()
    lay = draw_layout(random.Random(1), "sparse", NAMES, rig, "s0")
    assert len(lay.parts) == len(NAMES)
    for a, b in itertools.combinations(lay.parts, 2):
        assert math.dist((a.x, a.y), (b.x, b.y)) >= SPARSE_MIN_GAP_M
        assert footprint_gap(a, b) >= SPARSE_MIN_FOOTPRINT_GAP_M - 1e-12
    for p in lay.parts:
        assert footprint_inside_tray(p, rig)
        assert p.z > rig.ipad_z


def test_a_cluster_layout_has_touching_neighbours():
    rig = RigConfig()
    lay = draw_layout(random.Random(2), "cluster", NAMES, rig, "c0")
    gaps = [footprint_gap(a, b) for a, b in itertools.combinations(lay.parts, 2)]
    assert any(g <= CLUSTER_GAP_M + 1e-9 for g in gaps)
    assert min(gaps) > 0.0
    assert all(footprint_inside_tray(p, rig) for p in lay.parts)


def test_the_boxes_stand_in_label_order():
    lay = draw_layout(random.Random(3), "sparse", NAMES, RigConfig(), "s1")
    want = label_boxes(desk_boxes.sample_boxes(random.Random(3), 4))
    assert len(lay.boxes) == 4
    assert [(b.x, b.y, b.yaw) for b in lay.boxes] == want


def test_a_layout_round_trips_through_json(tmp_path):
    rig = RigConfig()
    for kind in ("sparse", "cluster", "heap"):
        lay = draw_layout(random.Random(5), kind, NAMES, rig, kind)
        path = tmp_path / f"{kind}.json"
        path.write_text(lay.to_json())
        back = Layout.from_json(path)
        assert back.name == lay.name and back.kind == kind and back.boxes == lay.boxes
        assert len(back.parts) == len(NAMES)
    assert math.isnan(back.parts[0].x)  # a heap keeps NaN


def test_the_class_comes_from_the_catalog():
    lay = draw_layout(random.Random(4), "sparse", ["brick_1x4"], RigConfig(), "s2")
    assert lay.parts[0].cls == "narrow"


def _one_part():
    return Layout("t", "sparse", [TruthPart("brick_2x4", "wide", 0.195, 0.0, 0.0109, 0.3)], [])


def test_the_template_is_one_to_one_on_the_screen():
    assert IPAD_AIR_11.mm_per_px == pytest.approx(25.4 / 264.0)
    img = template_png(_one_part(), RigConfig())
    assert img.size == (2360, 1640)
    assert IPAD_AIR_11.to_px((0.0, 0.0)) == pytest.approx((1180.0, 820.0))
    # the tray's 0.20 m side lies on the screen's long axis: its edges are 1039.4 px from centre
    assert any(img.getpixel((x, 820)) == (255, 255, 255) for x in range(138, 144))
    assert any(img.getpixel((x, 820)) == (255, 255, 255) for x in range(2216, 2222))


def test_the_tray_interior_fits_the_screen():
    rig = RigConfig()
    pose = nominal_screen_pose(rig)
    for dx, dy in itertools.product((-1, 1), repeat=2):
        corner = np.array(
            [
                rig.tray_centre[0] + dx * rig.tray_interior[0] / 2,
                rig.tray_centre[1] + dy * rig.tray_interior[1] / 2,
                rig.ipad_z,
            ]
        )
        s = pose.to_camera(corner)
        u, v = IPAD_AIR_11.to_px(s[:2])
        assert 0 < u < 2360 and 0 < v < 1640
    # screen x runs along base +y, screen y along base -x
    assert pose.to_camera([0.195, 0.05, rig.ipad_z])[0] == pytest.approx(0.05)
    assert pose.to_camera([0.185, 0.0, rig.ipad_z])[1] == pytest.approx(0.01)


def test_place_layout_moves_parts_by_the_measured_offset():
    rig = RigConfig()
    nominal = nominal_screen_pose(rig)
    th = math.radians(1.0)
    rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    # shift 2 mm along the screen x axis, rotate 1 degree about the screen z axis
    measured = Pose(nominal.rot @ rz, nominal.pos + nominal.rot[:, 0] * 0.002)
    lay = draw_layout(random.Random(6), "sparse", NAMES, rig, "s")
    moved = place_layout(lay, nominal, measured)
    assert moved.frame == "measured" and moved.boxes == lay.boxes
    for a, b in zip(lay.parts, moved.parts):
        s = nominal.to_camera([a.x, a.y, a.z])
        want = measured.rot @ s + measured.pos
        assert (b.x, b.y, b.z) == pytest.approx(tuple(want))
        assert (b.yaw - a.yaw + math.pi) % (2 * math.pi) - math.pi == pytest.approx(th)


def test_an_unknown_part_name_prints_the_fail_token(tmp_path):
    rig = tmp_path / "rig.json"
    rig.write_text(RigConfig().to_json())
    parts = tmp_path / "parts.json"
    parts.write_text(json.dumps([{"name": "no_such_part", "count": 1}]))
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "real_layouts.py"),
            "--rig",
            str(rig),
            "--parts",
            str(parts),
            "--seed",
            "1",
            "--out_dir",
            str(tmp_path / "o"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("LAYOUTS_OK", "LAYOUTS_FAIL"))]
    assert len(lines) == 1 and lines[0].startswith("LAYOUTS_FAIL")


def test_the_script_writes_layouts_and_templates(tmp_path):
    rig = tmp_path / "rig.json"
    rig.write_text(RigConfig().to_json())
    parts = tmp_path / "parts.json"
    parts.write_text(
        json.dumps(
            [
                {"name": "brick_2x4", "count": 3},
                {"name": "slope_1x2", "count": 2},
                {"name": None, "label": "odd gear"},
            ]
        )
    )
    out = tmp_path / "o"
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "real_layouts.py"),
            "--rig",
            str(rig),
            "--parts",
            str(parts),
            "--seed",
            "1",
            "--sparse",
            "1",
            "--cluster",
            "1",
            "--heap",
            "1",
            "--out_dir",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("LAYOUTS_OK", "LAYOUTS_FAIL"))]
    assert len(lines) == 1 and lines[0].startswith("LAYOUTS_OK"), r.stdout + r.stderr
    assert json.loads(lines[0].split(" ", 1)[1])["unmatched"] == ["odd gear"]
    names = sorted(p.name for p in out.iterdir())
    try:
        import cv2  # noqa: F401
    except ImportError:
        board = None
    else:
        board = "board.png"
    assert json.loads(lines[0].split(" ", 1)[1])["board"] == (board and str(out / board))
    assert names == [
        *([board] if board else []),
        "cluster0.json",
        "cluster0.png",
        "empty.json",
        "empty.png",
        "heap0.json",
        "sparse0.json",
        "sparse0.png",
    ]
    assert len(Layout.from_json(out / "heap0.json").parts) == 5
    empty = Layout.from_json(out / "empty.json")  # the empty screen reads as a sparse layout
    assert (empty.name, empty.kind, empty.parts, empty.boxes) == ("empty", "sparse", [], [])


def test_every_footprint_keeps_the_edge_margin_on_the_owners_tray():
    rig = RigConfig(tray_interior=(0.14, 0.18))
    for seed in range(8):
        for kind in ("sparse", "cluster"):
            lay = draw_layout(random.Random(seed), kind, NAMES, rig, kind)
            for p in lay.parts:
                d = np.abs(_corners_of(p) - rig.tray_centre)
                assert (d <= np.asarray(rig.tray_interior) / 2 - TRAY_EDGE_MARGIN_M + 1e-12).all()
            img = template_png(lay, rig)
            assert img.size == (2360, 1640)
    s = nominal_screen_pose(rig)
    for dx, dy in itertools.product((-1, 1), repeat=2):
        c = [rig.tray_centre[0] + dx * 0.07, rig.tray_centre[1] + dy * 0.09, rig.ipad_z]
        u, v = IPAD_AIR_11.to_px(s.to_camera(c)[:2])
        assert 0 < u < 2360 and 0 < v < 1640


def _corners_of(p):
    from maker_arm_lab.real.layouts import _part_corners

    return _part_corners(p)


def test_a_rig_json_without_the_new_fields_loads_with_defaults(tmp_path):
    path = tmp_path / "rig.json"
    data = json.loads(RigConfig().to_json())
    for k in ("tray_centre", "tray_interior", "tray_floor_z", "box_interior", "ipad_z"):
        del data[k]
    path.write_text(json.dumps(data))
    rig = RigConfig.from_json(path)
    assert rig.tray_interior == (0.13, 0.20) and rig.ipad_z == 0.0061
    assert rig.tray_centre == (0.195, 0.0) and rig.tray_floor_z == 0.0
    assert rig.box_interior == RigConfig().box_interior


def _layouts_main(tmp_path, capsys, entries, rig=None):
    """``real_layouts.main`` in process: (exit code, the one token line)."""
    script = load_script("real_layouts")
    (tmp_path / "rig.json").write_text((rig or RigConfig()).to_json())
    (tmp_path / "parts.json").write_text(json.dumps(entries))
    code = script.main(
        ["--rig", str(tmp_path / "rig.json"), "--parts", str(tmp_path / "parts.json"),
         "--seed", "1", "--sparse", "1", "--cluster", "1", "--heap", "1",
         "--out_dir", str(tmp_path / "o")]
    )  # fmt: skip
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("LAYOUTS_")]
    assert len(lines) == 1
    return code, lines[0]


def test_a_part_outside_the_base_classes_fails_naming_each_one(tmp_path, capsys):
    entries = [
        {"name": "brick_2x4", "count": 3},
        {"name": "plate_2x4", "count": 1},
        {"name": "tile_1x2", "count": 2},
        {"name": "ldraw_3700", "count": 1},
    ]
    code, line = _layouts_main(tmp_path, capsys, entries)
    assert code == 1 and line.startswith("LAYOUTS_FAIL ")
    for name in ("plate_2x4", "tile_1x2", "ldraw_3700"):
        assert name in line
    assert "brick_2x4" not in line


@pytest.mark.parametrize("entries", [[], [{"name": None, "label": "odd gear"}]])
def test_a_part_list_with_no_catalog_part_fails(tmp_path, capsys, entries):
    code, line = _layouts_main(tmp_path, capsys, entries)
    assert code == 1 and line.startswith("LAYOUTS_FAIL ") and "no catalog part" in line


def test_a_tray_interior_off_the_screen_fails(tmp_path, capsys):
    rig = RigConfig(tray_interior=(0.18, 0.14))  # x and y swapped: 0.18 m > the 0.158 m screen
    code, line = _layouts_main(tmp_path, capsys, [{"name": "brick_2x4", "count": 5}], rig)
    assert code == 1 and line.startswith("LAYOUTS_FAIL ") and "screen" in line
    assert not (tmp_path / "o").exists()


def test_ctrl_c_prints_the_fail_token(tmp_path, capsys, monkeypatch):
    import maker_arm_lab.real.layouts as layouts_mod

    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(layouts_mod, "draw_layout", interrupt)
    code, line = _layouts_main(tmp_path, capsys, [{"name": "brick_2x4", "count": 5}])
    assert code == 1 and line == "LAYOUTS_FAIL " + json.dumps({"reason": "interrupted"})
