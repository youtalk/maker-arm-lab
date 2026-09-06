import pytest

from maker_arm_lab.assets import urdf_prep
from maker_arm_lab.core import profile as prof


def test_sim_profile_uses_urdf_limits_for_arm_joints_only():
    p = prof.sim_profile()
    limits = urdf_prep.arm_joint_limits()
    for j, (lo, hi) in zip(p["joints"][:6], limits):
        assert (j["q_lo"], j["q_hi"]) == (lo, hi)
    g = p["joints"][prof.GRIPPER_INDEX]
    assert (g["q_lo"], g["q_hi"]) == (-2.092, -0.039)


def test_sim_profile_refuses_calibrated_direction_or_offset():
    p = prof.load_profile()
    p["joints"][1]["offset"] = 0.3
    with pytest.raises(ValueError, match="offset"):
        prof._check_uncalibrated(p)


def test_arm_gains_follow_the_profile_and_motor_ratings():
    gains = prof.arm_gains(prof.load_profile())
    assert list(gains) == list(urdf_prep.ARM_JOINT_NAMES)
    assert gains["link_002_joint"] == {"kp": 60.0, "kd": 4.0, "tau_max": 4.0, "effort_limit": 14.0}
    assert gains["link_003_joint"] == {"kp": 150.0, "kd": 4.5, "tau_max": 6.0, "effort_limit": 17.0}
    assert gains["link_004_joint"]["effort_limit"] == 17.0
    assert gains["link_007_joint"] == {"kp": 30.0, "kd": 2.0, "tau_max": 4.0, "effort_limit": 14.0}


def test_gripper_angle_is_linear_between_profile_endpoints():
    p = prof.load_profile()
    assert prof.gripper_angle(p, 0.0) == -2.092
    assert prof.gripper_angle(p, 1.0) == -0.039
    assert prof.gripper_angle(p, 0.5) == pytest.approx((-2.092 - 0.039) / 2)


def test_clamp_joint_targets_flags_and_clamps_out_of_range_envs():
    p = prof.sim_profile()
    limits = urdf_prep.arm_joint_limits()
    inside = [(lo + hi) / 2 for lo, hi in limits]
    outside = list(inside)
    outside[2] = limits[2][1] + 1.0
    out, flags = prof.clamp_joint_targets([inside, outside], [True, False], p)
    assert flags == [False, True]
    assert out[0] == pytest.approx(inside)
    assert out[1][2] == pytest.approx(limits[2][1])
    assert len(out[1]) == 6
