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
    # Looks along the tool axis (-z of link_007), pitched 20 deg toward +x.
    assert forward.tolist() == pytest.approx(
        [math.sin(math.radians(20)), 0.0, -math.cos(math.radians(20))], abs=1e-6
    )
    right = r @ torch.tensor([1.0, 0.0, 0.0])
    assert right.tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)


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
