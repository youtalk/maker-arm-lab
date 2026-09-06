import math

import pytest
import torch

from maker_arm_lab.assets import urdf_prep
from maker_arm_lab.robots import frames


def _rot(q):
    w, x, y, z = q
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_canonical_tool_frame_is_a_proper_rotation():
    c = torch.tensor(frames.CANON_IN_BODY)
    assert torch.allclose(c @ c.T, torch.eye(3), atol=1e-9)
    assert torch.det(c) == pytest.approx(1.0)
    assert tuple(c[0].tolist()) == frames.JAW_AXIS_BODY
    assert tuple(c[2].tolist()) == frames.TOOL_AXIS_BODY


def test_quat_from_axes_identity_and_wrist_camera():
    assert frames.quat_wxyz_from_axes((1, 0, 0), (0, 1, 0), (0, 0, 1)) == pytest.approx(
        (1.0, 0.0, 0.0, 0.0)
    )
    r = _rot(frames.WRIST_CAM_ROT_IN_WRIST)
    forward = r @ torch.tensor([0.0, 0.0, 1.0])
    # Looks along the tool axis (-z of link_007), pitched 30 deg toward -x -- back toward
    # the tool axis, since the mount is offset on +x to clear the jaws. The SIGN is the
    # half of this that matters: a +x pitch is what aimed the previous placeholder back
    # across the wrist motor.
    assert forward.tolist() == pytest.approx(
        [-math.sin(math.radians(30)), 0.0, -math.cos(math.radians(30))], abs=1e-6
    )
    right = r @ torch.tensor([1.0, 0.0, 0.0])
    assert right.tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)


def test_wrist_camera_clears_the_motor_and_aims_at_the_grasp_point():
    """Pin the hardware constraint the wrist-camera pose is derived from.

    On the physical arm the camera is mounted in front of the wrist motor, distal to it on
    the jaw side, so the motor cannot appear in the image. In link_007 coordinates, measured
    from the vendored collision STLs, that means the lens must sit past the motor's front
    face at z = -0.1128 and outside the jaws' x extreme of +0.0205 -- the jaws travel only
    in y, so an x offset clears them at every opening -- while still aiming at grasp_center.
    """
    motor_front_z = -0.1128
    jaw_x_max = 0.0205
    grasp_z = -0.169638704618

    x, y, z = frames.WRIST_CAM_POS_IN_WRIST
    assert z < motor_front_z, "the lens must sit in front of the wrist motor's front face"
    assert x > jaw_x_max, "the lens must sit outside the volume the jaws sweep through"
    assert y == 0.0

    forward = _rot(frames.WRIST_CAM_ROT_IN_WRIST) @ torch.tensor([0.0, 0.0, 1.0])
    assert forward[2] < 0.0
    hit = torch.tensor([x, y, z]) + forward * ((grasp_z - z) / forward[2])
    # The optical axis passes through the grasp point, so the grasp point stays centred in
    # frame at every range -- what the off-axis mount's pitch is there to buy back.
    assert hit[:2].tolist() == pytest.approx([0.0, 0.0], abs=1e-3)


def test_gripper_convention_matches_the_urdf():
    assert frames.GRIPPER_CLOSED_POS == 0.0
    assert frames.GRIPPER_OPEN_POS == pytest.approx(urdf_prep.gripper_travel())


def test_home_pose_names_every_arm_joint_inside_its_limits():
    limits = dict(zip(urdf_prep.ARM_JOINT_NAMES, urdf_prep.arm_joint_limits()))
    assert set(frames.HOME_JOINT_POS) == set(urdf_prep.ARM_JOINT_NAMES) | set(
        urdf_prep.GRIPPER_JOINT_NAMES
    )
    for name, (lo, hi) in limits.items():
        assert lo + 0.05 <= frames.HOME_JOINT_POS[name] <= hi - 0.05, name
    for name in urdf_prep.GRIPPER_JOINT_NAMES:
        assert frames.HOME_JOINT_POS[name] == frames.GRIPPER_OPEN_POS


def test_measured_sign_relations_are_pinned_to_their_measured_values():
    """Pin the SHIPPED sign constants, not just the fold's handling of arbitrary signs.

    ``core.oracle.fold_jaw_heading`` takes ``roll_sign`` / ``base_sign`` as arguments and
    ``test_oracle.py`` exercises it with both polarities, which is the right way to test the
    fold. It cannot catch a wrong value in ``frames.py``, though, because it never reads
    these constants -- and Task D3 measured both of them to be the OPPOSITE of the OpenYAM
    values the repo shipped with, a flip that the whole CPU suite passed straight through.

    These are measured physical facts about the arm, so pinning the literals is the point.
    Re-measure with ``scripts/scripted_grasp_check.py --signs`` before changing either; the
    provenance comment in ``frames.py`` records the run that set them.
    """
    assert frames.WRIST_ROLL_SIGN == 1.0
    assert frames.BASE_YAW_SIGN == -1.0


def test_sign_relations_are_unit_magnitude_directions():
    """Each is a d(jaw_yaw)/d(joint) direction, so +-1 exactly -- never 0, never a gain."""
    for name in ("WRIST_ROLL_SIGN", "BASE_YAW_SIGN"):
        value = getattr(frames, name)
        assert isinstance(value, float), name
        assert abs(value) == 1.0, name
