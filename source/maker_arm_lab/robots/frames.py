"""Maker Arm frame conventions and measured constants (isaaclab-free).

Everything an oracle, a check script, or the real-arm interface needs to know about WHERE
things are on this arm lives here, so `oracle_runtime.py` and `scripted_grasp_check.py`
carry no body names of their own. Values marked ``# INITIAL:`` are starting points that a
named GPU task measures and replaces; the task deletes the marker.
"""

import math

from maker_arm_lab.assets.urdf_prep import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES  # noqa: F401

ARM_JOINT_REGEX = "link_00[2-7]_joint"
TOOL_BODY = "grasp_center"  # the URDF's fixed frame between the jaws; the IK/servo target
WRIST_LINK = "link_007"  # parent of the gripper joints and of the wrist camera
WRIST_ROLL_INDEX = 5  # index of link_007_joint within ARM_JOINT_NAMES

# The canonical tool frame the oracle reasons in: x = jaw closing axis, z = tool axis
# pointing from the wrist toward the jaws. In grasp_center coordinates the jaws slide along
# +y (gripper_left_joint axis) and the jaws lie along -z (grasp_center sits at z = -0.1696 in
# link_007, i.e. the tool points along -z). Rows are the canonical x, y, z axes expressed in
# body coordinates; the matrix is a proper rotation (det +1), checked in tests.
JAW_AXIS_BODY = (0.0, 1.0, 0.0)
TOOL_AXIS_BODY = (0.0, 0.0, -1.0)
CANON_IN_BODY = (JAW_AXIS_BODY, (1.0, 0.0, 0.0), TOOL_AXIS_BODY)

# URDF-native gripper convention: 0 = jaws touching, 0.0524125 = fully open (per jaw).
GRIPPER_CLOSED_POS = 0.0
GRIPPER_OPEN_POS = 0.0524125

# Kinematic sign relations the jaw-heading fold (core.oracle.fold_jaw_heading) depends on:
# d(jaw_yaw)/d(wrist roll) and d(jaw_yaw)/d(base yaw), each +-1.
# INITIAL: the OpenYAM relation; measured by scripts/scripted_grasp_check.py --signs (Task D3).
WRIST_ROLL_SIGN = -1.0
BASE_YAW_SIGN = 1.0

# Home pose, URDF joint coordinates. Tool axis within 30 deg of world -z, grasp_center over
# the scatter zone at the achievable height of roughly 0.16-0.19 m above the desk -- the
# spec's original ~0.25 m target is geometrically unreachable at this tool-axis and
# joint-margin constraint; see scripts/find_home_pose.py's module docstring for the sweep
# that established the ceiling and the reasoning for lowering the target instead of either
# constraint.
# Pinned by scripts/find_home_pose.py on 2026-09-05: grasp_center at (0.2745, 0.0375, 0.1628),
# tool axis 23.57 deg off -z, min joint margin 0.30 rad.
HOME_JOINT_POS: dict[str, float] = {
    "link_002_joint": 0.0,
    "link_003_joint": -1.79,
    "link_004_joint": 2.05,
    "link_005_joint": -1.3,
    "link_006_joint": -0.22,
    "link_007_joint": 0.0,
    "gripper_left_joint": GRIPPER_OPEN_POS,
    "gripper_right_joint": GRIPPER_OPEN_POS,
}

# Where a desk-resting brick sits between the jaws, in grasp_center coordinates (m), and the
# extra descend depth (m) that lets the jaw tips reach it.
# INITIAL: at grasp_center itself, no extra depth; measured by scripted_grasp_check (Task D3).
PICK_OFFSET_MEASURED: tuple[float, float, float] = (0.0, 0.0, 0.0)
DESCEND_CLEARANCE_MEASURED: float = 0.0


def quat_wxyz_from_axes(right, down, forward) -> tuple[float, float, float, float]:
    """Unit quaternion (w, x, y, z) of the rotation whose columns are the three unit vectors.

    Used for a ROS-convention camera (x right, y down, z forward) expressed in its parent
    body. Shepperd's method; inputs must be orthonormal and right-handed.
    """
    m = [
        [right[0], down[0], forward[0]],
        [right[1], down[1], forward[1]],
        [right[2], down[2], forward[2]],
    ]
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x, y, z = (
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        )
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w, x, y, z = (
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        )
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w, x, y, z = (
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        )
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w, x, y, z = (
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        )
    return (w, x, y, z)


# Wrist camera placeholder (spec section 3.3): 50 mm off the tool axis on -x, 60 mm toward
# the jaws, looking along the tool axis pitched 20 deg toward +x so the jaws stay in frame.
# The URDF has no camera frame and the SDK publishes no optics, so this pose is UNVERIFIED
# and is an arrival re-measurement item (spec section 7). Image "right" is the jaw axis (+y).
WRIST_CAM_UNVERIFIED = True
WRIST_CAM_POS_IN_WRIST: tuple[float, float, float] = (-0.05, 0.0, -0.06)
_PITCH = math.radians(20.0)
_FORWARD = (math.sin(_PITCH), 0.0, -math.cos(_PITCH))
_RIGHT = (0.0, 1.0, 0.0)
_DOWN = (math.cos(_PITCH), 0.0, math.sin(_PITCH))  # forward x right
WRIST_CAM_ROT_IN_WRIST = quat_wxyz_from_axes(_RIGHT, _DOWN, _FORWARD)
