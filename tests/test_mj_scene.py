"""The MuJoCo scene against the constants it is built from (MS3 parity checks, section 3).

Real MuJoCo on CPU: a model compiles in tens of milliseconds, so these run in the ordinary
suite. The FK identity is the check that matters most -- the global IK plans every leg the
oracle walks, and if MuJoCo's chain disagreed with it by even a millimetre the jaws would
close beside every brick.
"""

import random

import mujoco
import numpy as np
import pytest
import torch

from maker_arm_lab.core import cameras, layout
from maker_arm_lab.core.global_ik import GlobalIK
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj import scene
from maker_arm_lab.robots import frames


@pytest.fixture(scope="module")
def built():
    draw = sample_run_draw(random.Random(3))
    draw["slot_colors"] = (0, 3, 1, 2)
    model, handles = scene.build_model(draw, num_bricks=6, cameras_on=True)
    return draw, model, handles


def test_model_carries_the_whole_scene(built):
    _, model, h = built
    assert len(h.joint_ids) == 8 and len(h.arm_actuators) == 6 and len(h.gripper_actuators) == 2
    assert len(h.brick_bodies) == 6 and len(h.brick_geoms) == 6
    assert h.cameras == ("overhead", "wrist")
    bin_geoms = [
        model.geom(g).name for g in range(model.ngeom) if model.geom(g).name.startswith("bin_")
    ]
    assert len(bin_geoms) == 20  # 4 bins x (floor + 4 walls)
    assert model.body("grasp_center").id == h.tool_body
    assert all(len(f) > 0 for f in h.finger_geoms)


def test_joint_order_and_limits_are_the_urdf_ones(built):
    _, model, h = built
    names = [model.joint(j).name for j in h.joint_ids]
    assert names == list(scene.JOINT_NAMES)
    ik = GlobalIK()
    arm_ranges = np.array([model.jnt_range[j] for j in h.joint_ids[:6]])
    assert np.allclose(arm_ranges[:, 0], ik.lower.numpy()) and np.allclose(
        arm_ranges[:, 1], ik.upper.numpy()
    )
    for j in h.joint_ids[6:]:
        assert np.allclose(model.jnt_range[j], (frames.GRIPPER_CLOSED_POS, frames.GRIPPER_OPEN_POS))


def test_actuators_carry_the_profile_gains(built):
    _, model, h = built
    from maker_arm_lab.core import profile as _profile

    gains = _profile.arm_gains(_profile.load_profile())
    for act, name in zip(h.arm_actuators, frames.ARM_JOINT_NAMES):
        g = gains[name]
        assert model.actuator_gainprm[act, 0] == pytest.approx(g["kp"])
        assert model.actuator_biasprm[act, 1] == pytest.approx(-g["kp"])
        assert model.actuator_biasprm[act, 2] == pytest.approx(-g["kd"])
        assert model.actuator_forcerange[act].tolist() == pytest.approx(
            [-g["effort_limit"], g["effort_limit"]]
        )
    for act in h.gripper_actuators:
        assert model.actuator_gainprm[act, 0] == pytest.approx(frames.GRIPPER_STIFFNESS)
        assert model.actuator_forcerange[act, 1] == pytest.approx(frames.GRIPPER_EFFORT_LIMIT_N)
    for j in h.joint_ids[:6]:
        assert model.dof_armature[model.jnt_dofadr[j]] == pytest.approx(frames.ARM_ARMATURE)
    assert all(model.body_gravcomp[model.body(b).id] == 1.0 for b in scene.ARM_BODIES)
    assert model.body_gravcomp[h.brick_bodies[0]] == 0.0


def test_forward_kinematics_matches_the_global_ik(built):
    """grasp_center under MuJoCo's chain == GlobalIK.fk, at HOME and at random joint vectors."""
    _, model, h = built
    data = mujoco.MjData(model)
    ik = GlobalIK()
    rng = np.random.default_rng(0)
    home = [frames.HOME_JOINT_POS[n] for n in frames.ARM_JOINT_NAMES]
    cases = [home] + [
        [rng.uniform(lo, hi) for lo, hi in zip(ik.lower.tolist(), ik.upper.tolist())]
        for _ in range(20)
    ]
    for q in cases:
        for adr, v in zip(h.joint_qpos[:6], q):
            data.qpos[adr] = v
        mujoco.mj_forward(model, data)
        pos, rot = ik.fk(torch.tensor(q, dtype=torch.float64))
        assert np.abs(data.xpos[h.tool_body] - pos.numpy()).max() < 1e-6
        assert np.abs(data.xmat[h.tool_body].reshape(3, 3) - rot.numpy()).max() < 1e-6
    # The pinned home pose reproduces the measured grasp_center (robots.frames).
    for adr, v in zip(h.joint_qpos[:6], home):
        data.qpos[adr] = v
    mujoco.mj_forward(model, data)
    assert data.xpos[h.tool_body].tolist() == pytest.approx([0.2749, 0.0371, 0.1646], abs=5e-4)


def test_home_pose_holds_and_bricks_rest(built):
    _, model, h = built
    data = mujoco.MjData(model)
    home = [frames.HOME_JOINT_POS[n] for n in frames.ARM_JOINT_NAMES]
    for adr, v in zip(h.joint_qpos[:6], home):
        data.qpos[adr] = v
    for adr in h.joint_qpos[6:]:
        data.qpos[adr] = frames.GRIPPER_OPEN_POS
    for i, adr in enumerate(h.brick_qpos):
        data.qpos[adr : adr + 3] = (0.17 + 0.02 * i, -0.12 + 0.05 * i, layout.SPAWN_Z)
    mujoco.mj_forward(model, data)
    data.ctrl[list(h.arm_actuators)] = home
    data.ctrl[list(h.gripper_actuators)] = frames.GRIPPER_OPEN_POS
    mujoco.mj_step(model, data, nstep=120)  # one second
    held = np.array([data.qpos[a] for a in h.joint_qpos[:6]])
    assert np.abs(held - home).max() < 0.005
    for body in h.brick_bodies:
        assert data.xpos[body][2] == pytest.approx(layout.BRICK_SIZE[2] / 2, abs=0.001)


def test_cameras_are_placed_from_the_shared_constants(built):
    draw, model, _h = built
    over = model.camera("overhead")
    expected = [p + d for p, d in zip(cameras.OVERHEAD_CAM_POS, draw["overhead_cam_dpos"])]
    assert over.pos.tolist() == pytest.approx(expected)
    assert over.fovy[0] == pytest.approx(cameras.OVERHEAD_FOV_DEG)
    wrist = model.camera("wrist")
    assert model.body(wrist.bodyid[0]).name == frames.WRIST_LINK
    expected = [p + d for p, d in zip(frames.WRIST_CAM_POS_IN_WRIST, draw["wrist_cam_dpos"])]
    assert wrist.pos.tolist() == pytest.approx(expected)
    assert wrist.fovy[0] == pytest.approx(cameras.WRIST_FOV_DEG)


def test_overhead_camera_looks_straight_down_with_bins_at_the_top(built):
    """The ROS->MuJoCo conversion: the camera's -z (view) is world -z and its +y (image up)
    is world +x, the composition the Isaac renders confirmed (core.cameras)."""
    _, model, _h = built
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = model.camera("overhead").id
    rot = data.cam_xmat[cam].reshape(3, 3)  # columns are the camera axes in world
    assert np.allclose(rot[:, 2], [0.0, 0.0, 1.0], atol=0.05)  # -z looks down
    assert np.allclose(rot[:, 1], [1.0, 0.0, 0.0], atol=0.05)  # image up = world +x


def test_kelvin_rgb_is_warm_below_and_cool_above_daylight():
    warm, day, cool = scene.kelvin_rgb(3000.0), scene.kelvin_rgb(6600.0), scene.kelvin_rgb(9000.0)
    assert warm[0] > warm[2] and cool[2] > cool[0]
    assert all(0.9 <= c <= 1.0 for c in day)
