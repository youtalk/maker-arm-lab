"""MS5 design section 7.1: the rig config, the box labels a rig can perceive, the sheet."""

import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

from maker_arm_lab.core import desk_boxes
from maker_arm_lab.core.parts import BASE_CLASSES
from maker_arm_lab.real import rig

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "real_episode_sheet.py"
POSES = [(0.18, 0.20, 0.1), (0.07, -0.19, -0.3), (0.08, 0.21, 0.4), (0.19, -0.20, 0.0)]


def test_boxes_are_labelled_by_side_then_by_x():
    assert rig.label_boxes(POSES) == [POSES[2], POSES[1], POSES[0], POSES[3]]


def test_labels_ignore_the_order_the_boxes_arrive_in():
    assert rig.label_boxes(POSES[::-1]) == rig.label_boxes(POSES)


def test_anything_but_two_boxes_a_side_is_refused():
    with pytest.raises(ValueError):
        rig.label_boxes(POSES[:3])
    with pytest.raises(ValueError):
        rig.label_boxes([(0.1, 0.2, 0.0), (0.2, 0.2, 0.0), (0.15, 0.19, 0.0), (0.1, -0.2, 0.0)])


def test_a_box_yaw_folds_into_a_quarter_turn():
    assert rig.fold_box_yaw(math.radians(80.0)) == pytest.approx(math.radians(-10.0))
    assert rig.fold_box_yaw(math.radians(-50.0)) == pytest.approx(math.radians(40.0))
    assert rig.fold_box_yaw(0.2) == pytest.approx(0.2)


def test_the_sheet_carries_each_class_to_the_box_the_draw_gave_it():
    sheet = rig.episode_sheet(20265000, 3)
    assert sheet == rig.episode_sheet(20265000, 3)
    assert sheet["classes"] == list(BASE_CLASSES)
    assert sorted(sheet["class_to_box"]) == [0, 1, 2, 3]
    assert [list(p) for p in rig.label_boxes(sheet["box_poses"])] == sheet["box_poses"]
    rng = random.Random("ms5-sheet:20265000:3")
    drawn = desk_boxes.sample_boxes(rng, 4)
    classes = desk_boxes.class_map(rng, 4)
    for c, box in enumerate(classes):
        assert sheet["box_poses"][sheet["class_to_box"][c]] == pytest.approx(list(drawn[box]))


def test_the_rig_config_round_trips_through_json_and_refuses_unknown_keys(tmp_path):
    config = rig.RigConfig(can_interface="can1", rt=(70, 4))
    path = tmp_path / "rig.json"
    path.write_text(config.to_json())
    assert rig.RigConfig.from_json(path) == config
    path.write_text(json.dumps({"camera_height": 0.7}))
    with pytest.raises(ValueError):
        rig.RigConfig.from_json(path)


def test_the_sheet_script_writes_the_sheet_and_prints_the_ok_token(tmp_path):
    out = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--seed",
            "20265000",
            "--episode",
            "3",
            "--out",
            str(tmp_path / "s.json"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    last = out.stdout.strip().splitlines()[-1]
    assert out.returncode == 0 and last.startswith("SHEET_OK ")
    assert json.loads((tmp_path / "s.json").read_text()) == rig.episode_sheet(20265000, 3)
    assert "box 0" in out.stdout


def test_a_sheet_usage_error_prints_the_fail_token():
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False)
    last = out.stdout.strip().splitlines()[-1]
    assert out.returncode == 2 and last.startswith("SHEET_FAIL ")


def test_the_reduced_profile_scales_the_caps_and_applies_the_calibration():
    base = rig.load_profile()
    config = rig.RigConfig(
        caps={"tau_max": [0.5] * 7, "kp": [0.25] * 7, "kd": [1.0] * 7},
        calibration={"direction": [-1.0] + [1.0] * 6, "offset": [0.1] * 7},
    )
    p = rig.reduced_profile(config)
    for j, b in zip(p["joints"], base["joints"]):
        assert j["tau_max"] == pytest.approx(0.5 * b["tau_max"])
        assert j["kp"] == pytest.approx(0.25 * b["kp"]) and j["kd"] == b["kd"]
        assert j["offset"] == 0.1
    assert p["joints"][0]["direction"] == -1.0 and p["joints"][1]["direction"] == 1.0
    assert rig.reduced_profile(rig.RigConfig()) == base


@pytest.mark.parametrize(
    ("caps", "match"),
    [
        ({"kp": [1.5] + [1.0] * 6}, r"caps\.kp\[0\]"),
        ({"kd": [0.0] * 7}, r"caps\.kd\[0\]"),
        ({"tau_max": [0.5] * 6}, "7"),
        ({"q_hi": [0.5] * 7}, "q_hi"),
    ],
)
def test_a_cap_outside_the_unit_interval_is_refused(caps, match):
    with pytest.raises(ValueError, match=match):
        rig.reduced_profile(rig.RigConfig(caps=caps))


def test_the_rig_json_round_trips_caps_and_calibration(tmp_path):
    config = rig.RigConfig(caps={"kp": [0.5] * 7}, calibration={"offset": [0.0] * 7})
    path = tmp_path / "rig.json"
    path.write_text(config.to_json())
    assert rig.RigConfig.from_json(path) == config
