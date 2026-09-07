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


def test_pick_offset_z_reproduces_the_formula_its_comment_names():
    """Tie PICK_OFFSET_MEASURED's z to the shipped code that produced it.

    Task D3 exists to pin this constant and GRASP_TILT_DEG, and the sign relations got a
    literal-pinning test on the explicit principle that measured physical facts about the
    arm are worth pinning. These two had none.

    This one is stronger than a literal, because the z offset is exactly reproducible from
    shipped code: `descend_pick_offset` balances desk clearance against tip margin at a given
    tilt, and the pinned value is its output at MIN_DESCEND_TILT for this brick, rounded to
    0.1 mm. Asserting the round-trip catches a drift in EITHER the constant or the formula,
    and it would have caught the comment defect the D3 review found (a paragraph describing
    an intermediate -0.030 m offset with a 4.8 mm tip margin, twelve lines above a pinned
    -0.0321 whose margin is 2.1 mm).
    """
    from maker_arm_lab.core.global_ik import MIN_DESCEND_TILT, descend_pick_offset
    from maker_arm_lab.core.layout import BRICK_SIZE

    solved = descend_pick_offset(MIN_DESCEND_TILT, BRICK_SIZE[2] / 2.0)
    assert solved is not None
    offset_z, tip_margin = solved
    assert frames.PICK_OFFSET_MEASURED[2] == pytest.approx(offset_z, abs=5e-5)
    assert round(offset_z, 4) == frames.PICK_OFFSET_MEASURED[2]
    # The margin the shipped offset actually buys, which is NOT the 4.8 mm an intermediate
    # offset in the same comment block buys. The factor of two decides whether the jaw
    # plates foul the desk.
    assert tip_margin == pytest.approx(0.00213, abs=5e-5)


def test_pick_offset_lateral_and_descend_clearance_are_measured_zeros():
    """Both are 0.0 as a RESULT, not as an unset default, so pin them.

    A zero that means "measured, and it came out zero" is indistinguishable at a glance from
    a zero that means "nobody set this". frames.py's comments record which these are -- x is
    where the jaws actually grip (centring the brick on the plate face was tried on GPU and
    made things strictly worse), and the descend clearance is degenerate with the pick offset
    on the global-IK path, which commands the descend to a position rather than driving it
    into the desk.
    """
    assert frames.PICK_OFFSET_MEASURED[0] == 0.0
    assert frames.PICK_OFFSET_MEASURED[1] == 0.0
    assert frames.DESCEND_CLEARANCE_MEASURED == 0.0


def test_grasp_tilt_deg_is_pinned_to_its_measured_worst_case():
    """20.0 deg, the smallest round value clearing the measured 18.42 deg worst case.

    Pinned as a literal for the same reason as the sign relations: nothing else in the CPU
    suite reads it, so a wrong value passes everything. The 18.42 figure is the corrected
    one -- an earlier sweep over the wrong band (|y| <= 0.05) produced 15.0, and a stale
    13.01 from that sweep survived in oracle_runtime.py until the D3 review found it.
    """
    assert frames.GRASP_TILT_DEG == 20.0
    assert frames.GRASP_TILT_DEG > 18.42
    assert frames.GRASP_TILT_RAD == pytest.approx(math.radians(20.0))
