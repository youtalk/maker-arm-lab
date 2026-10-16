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
        calibration={"direction": [1.0] * 7, "offset": [0.0] * 7},
    )
    p = rig.reduced_profile(config)
    for j, b in zip(p["joints"], base["joints"]):
        assert j["tau_max"] == pytest.approx(0.5 * b["tau_max"])
        assert j["kp"] == pytest.approx(0.25 * b["kp"]) and j["kd"] == b["kd"]
        assert j["offset"] == 0.0 and j["direction"] == 1.0
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


@pytest.mark.parametrize(
    "calibration",
    [{"direction": [-1.0] + [1.0] * 6}, {"offset": [0.0] * 6 + [0.1]}],
)
def test_a_non_identity_calibration_is_refused_while_the_limits_are_motor_frame(calibration):
    with pytest.raises(ValueError, match="motor frame"):
        rig.reduced_profile(rig.RigConfig(calibration=calibration))


def test_the_rig_json_round_trips_caps_and_calibration(tmp_path):
    config = rig.RigConfig(caps={"kp": [0.5] * 7}, calibration={"offset": [0.0] * 7})
    path = tmp_path / "rig.json"
    path.write_text(config.to_json())
    assert rig.RigConfig.from_json(path) == config


def _caps_rig(kp):
    return rig.RigConfig(caps={"tau_max": [0.5] * 7, "kp": kp, "kd": [0.5] * 7})


def test_the_default_error_clamp_is_the_sdks():
    import maker_arm_rs

    from maker_arm_lab.core import dynamics_model as dm
    from maker_arm_lab.core import profile as core_profile

    profile = core_profile.load_profile()
    mid = [0.5 * (j["q_lo"] + j["q_hi"]) for j in profile["joints"]]
    tracking = maker_arm_rs.Tracking(dm.dynamics(dm.urdf_bodies()), profile=profile)
    tracking.push(0.0, 0.005, [mid])
    q = list(mid)
    q[1] += 0.3
    for k in range(400):
        out = tracking.update(k * 0.005, q, [0.0] * 7, [0.0] * 7)
    assert q[1] - out[1][0] == pytest.approx(rig.E_MAX_DEFAULT_RAD)


def test_reduced_kp_widens_the_error_clamp_so_the_spring_ceiling_holds():
    kp = [0.25, 0.25, 0.5, 1.0, 0.5, 0.25, 0.25]
    params = rig.session_tracking_params(_caps_rig(kp), {"tau_th": [2.0] * 6})
    assert params["tau_th"] == [2.0] * 6
    assert params["e_max"] == pytest.approx([0.03 / s for s in kp[:6]])
    profile = rig.reduced_profile(_caps_rig(kp))
    full = rig.load_profile()
    for j in range(6):
        ceiling = profile["joints"][j]["kp"] * params["e_max"][j]
        assert ceiling == pytest.approx(full["joints"][j]["kp"] * rig.E_MAX_DEFAULT_RAD)


def test_no_caps_set_the_default_error_clamp_explicitly():
    params = rig.session_tracking_params(rig.RigConfig(), {})
    assert params["e_max"] == [rig.E_MAX_DEFAULT_RAD] * 6


def test_an_explicit_error_clamp_wins_only_up_to_the_scaled_one():
    caps = _caps_rig([0.25] * 7)
    tight = [0.05, 0.12, 0.03, 0.1, 0.1, 0.1]
    assert rig.session_tracking_params(caps, {"e_max": tight})["e_max"] == tight
    with pytest.raises(ValueError, match=r"e_max\[1\]"):
        rig.session_tracking_params(caps, {"e_max": [0.05, 0.13, 0.03, 0.1, 0.1, 0.1]})
    with pytest.raises(ValueError, match="e_max"):
        rig.session_tracking_params(caps, {"e_max": 0.05})


def test_the_error_clamp_rule_reads_the_kp_scale_of_a_given_profile():
    assert rig.capped_e_max([0.5] * 7) == pytest.approx([0.06] * 6)
    assert rig.capped_e_max(None) == [rig.E_MAX_DEFAULT_RAD] * 6


def test_the_gripper_map_round_trips_through_the_rig_json(tmp_path):
    config = rig.RigConfig(gripper_map=(0.03, 0.065))
    path = tmp_path / "rig.json"
    path.write_text(config.to_json())
    assert rig.RigConfig.from_json(path).gripper_map == (0.03, 0.065)
    assert rig.RigConfig().gripper_map is None


@pytest.mark.parametrize(
    "value", [[0.0, 0.05], [float("nan"), 0.05], [0.03, float("inf")], [0.03], "ab", [True, 0.0]]
)
def test_a_gripper_map_that_cannot_be_used_is_refused_on_load(tmp_path, value):
    path = tmp_path / "rig.json"
    path.write_text(json.dumps({"gripper_map": value}).replace("NaN", "NaN"))
    with pytest.raises(ValueError, match="gripper_map"):
        rig.RigConfig.from_json(path)
