"""MS5 design section 4.3: the arm backend over maker_arm_rs, on the SDK's own simulator."""

import math
import time

import maker_arm_rs
import numpy as np
import pytest

from maker_arm_lab.assets.urdf_prep import ARM_JOINT_NAMES
from maker_arm_lab.core import dynamics_model as dm
from maker_arm_lab.core.global_ik import GlobalIK
from maker_arm_lab.core.profile import gripper_angle, sim_profile
from maker_arm_lab.real import arm as real_arm
from maker_arm_lab.robots.frames import HOME_JOINT_POS

HOME = [HOME_JOINT_POS[n] for n in ARM_JOINT_NAMES]


def test_jaw_fraction_inverts_the_provisional_transmission_and_clips():
    profile = sim_profile()
    for f in (0.0, 0.25, 1.0):
        assert real_arm.jaw_fraction(profile, gripper_angle(profile, f)) == pytest.approx(f)
    g = profile["joints"][6]
    assert real_arm.jaw_fraction(profile, g["q_hi"] + 0.5) == 1.0
    assert real_arm.jaw_fraction(profile, g["q_lo"] - 0.5) == 0.0


def test_the_pick_jacobian_predicts_a_small_motion():
    ik = GlobalIK()
    jac = real_arm.pick_jacobian(ik, HOME)
    dq = np.array([0.01, -0.02, 0.015, 0.0, 0.01, -0.01]) * 0.01
    moved = real_arm.pick_point(ik, np.array(HOME) + dq) - real_arm.pick_point(ik, HOME)
    assert jac.shape == (3, 6) and jac @ dq == pytest.approx(moved, abs=1e-7)


def test_the_reference_acceleration_is_the_second_difference():
    dt = 1.0 / 30.0
    targets = [[0.5 * 2.0 * (k * dt) ** 2] * 6 for k in range(3)]
    assert real_arm.reference_acceleration(targets, dt) == pytest.approx((2.0,) * 6)
    assert real_arm.reference_acceleration(targets[:2], dt) == (0.0,) * 6


def test_a_static_arm_whose_motors_carry_gravity_has_no_contact_residual():
    dynamics = dm.dynamics(dm.urdf_bodies())
    tau = dynamics.gravity(HOME)
    residual = real_arm.contact_residual(dynamics, HOME, (0.0,) * 6, (0.0,) * 6, tau)
    assert residual == pytest.approx((0.0,) * 6, abs=1e-9)


def test_the_sdk_arm_starts_pushes_holds_and_stops_on_the_sdk_simulator():
    sdk = real_arm.SdkArm.connect("sim", dm.dynamics(dm.urdf_bodies()))
    try:
        first = sdk.start()
        assert first.state == "enabled" and first.alive and len(first.q) == 6
        target = list(first.q)
        target[0] += 0.02
        sdk.push(sdk.now() + 0.05, 1.0 / 30.0, [*target, first.gripper])
        time.sleep(0.5)
        moved = sdk.read()
        assert abs(moved.q[0] - target[0]) < abs(first.q[0] - target[0])
        telemetry = sdk.telemetry()
        assert telemetry["tick_t"] - first.t > 0.3
        assert telemetry["max_interval"] > 0.0 and math.isfinite(telemetry["max_interval"])
        sdk.hold()
        time.sleep(0.3)
        assert sdk.now() is None
        with pytest.raises(RuntimeError):
            sdk.push(0.0, 1.0 / 30.0, [*target, first.gripper])
        assert sdk.read().state == "enabled"
    finally:
        sdk.stop()
    assert sdk.read() is None or sdk.read().state != "enabled"


def test_the_sdk_arm_connects_with_a_reduced_profile_and_sends_its_gains():
    profile = maker_arm_rs.profile()
    for j in profile["joints"]:
        j["kp"] *= 0.5
    sdk = real_arm.SdkArm.connect("sim", dm.dynamics(dm.urdf_bodies()), profile=profile)
    try:
        sdk.start()
        time.sleep(0.2)
        ticks, trk = sdk.drain()
        assert len(ticks["t"]) > 10 and len(trk["tick_t"]) > 10
        cap = np.array([j["kp"] for j in profile["joints"]])
        assert (np.asarray(ticks["cmd_kp"]) <= cap + 1e-9).all()
    finally:
        sdk.stop()


def test_a_raised_profile_is_refused_before_the_sdk_arm_connects():
    profile = maker_arm_rs.profile()
    profile["joints"][2]["tau_max"] *= 2.0
    with pytest.raises(ValueError, match=r"joints\[2\]\.tau_max"):
        real_arm.SdkArm.connect("can-does-not-exist", dm.dynamics(dm.urdf_bodies()),
                                profile=profile)  # fmt: skip
