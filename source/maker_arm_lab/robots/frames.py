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
# Measured by scripted_grasp_check.py --signs on 2026-09-05: d_yaw_per_roll=0.9478,
# d_yaw_per_base=-0.9999962, at the home pose (tool axis 23.85 deg off world -z).
# BOTH flip relative to the OpenYAM relation this started from (roll -1, base +1). The base
# one is exact and independently derivable: link_002_joint's URDF axis is (0, 0, -1) in a
# frame rotated rpy=(-pi/2, 0, 0) from base_link, which assets.urdf_prep.MOUNT_RPY then
# carries onto world -z, so d(jaw_yaw)/d(joint1) = -1 identically, at any pose. The roll
# magnitude is below 1 because the wrist roll axis is off vertical by the home tool angle.
WRIST_ROLL_SIGN = 1.0
BASE_YAW_SIGN = -1.0

# Tool-axis tilt off world -z that the oracle GRASPS at, in degrees. NOT a comfort number:
# this arm cannot hold a level tool over the scatter band at hover height, so demanding one
# spends joint travel it does not have. The OpenYAM port asked for exactly 0.000 deg, which
# on this arm drives link_005_joint (only 0.30 rad from its stop at home, the tightest of the
# six) into that stop within 60 hover steps, after which the arm is jammed and the servo is
# driving a dead mechanism -- measured, Task D3, /tmp/d3r_hover_diag2.log.
#
# Measured by /tmp/d3_band_sweep.py on 2026-09-06 -- a numpy FK of the vendored URDF plus
# urdf_prep's Z-up mount, reproducing the live Isaac grasp_center and tool angle at
# HOME_JOINT_POS to 4 decimals (0.2749, 0.0371, 0.1646 / 23.85 deg) before being trusted.
# Sweeping link_003..link_006 on a 2 deg grid (69,984,000 shapes) and binning by reach
# radius over core.layout's ACTUAL scatter band -- SCATTER_X (0.13, 0.29) and SCATTER_Y
# (-0.18, 0.18), i.e. radius 0.130-0.341 m and azimuth +-54.2 deg. The smallest tilt the
# arm can hold with every joint at least 0.30 rad from its stops, by radius:
#
#     radius (m)   0.13-0.25   0.25-0.27   0.27-0.29   0.29-0.31   0.31-0.33   0.33-0.35
#     hover            10.42       10.42       12.42       14.42       16.42       18.42  deg
#     descend           0.42        0.42        0.42        0.42        0.42        0.42  deg
#
# So the tilt is a cost of the APPROACH only -- at the brick the arm is level to within half
# a degree everywhere -- and the binding case is hover at the outer corners of the band
# (x=0.29, |y|=0.18). 20.0 deg is the smallest round value clearing that 18.42 deg worst
# case, keeping the tilt as small as the grasp geometry allows. It is inside the spec's
# 30 deg tool cone and close to the home pose's own 23.85 deg, so the servo barely has to
# change attitude from home.
#
# The first version of this constant was 15.0 deg, from a sweep that used |y| <= 0.05. That
# is find_home_pose.py's constraint on where HOME may sit, NOT the band bricks scatter over;
# it understated the reach radius (0.294 m vs 0.341 m) and the azimuth (+-11 deg vs
# +-54 deg), and 15.0 deg does not cover the outer corners.
GRASP_TILT_DEG = 20.0
GRASP_TILT_RAD = math.radians(GRASP_TILT_DEG)

# Home pose, URDF joint coordinates. Tool axis within 30 deg of world -z, grasp_center over
# the scatter zone at the achievable height of roughly 0.16-0.19 m above the desk -- the
# spec's original ~0.25 m target is geometrically unreachable at this tool-axis and
# joint-margin constraint; see scripts/find_home_pose.py's module docstring for the sweep
# that established the ceiling and the reasoning for lowering the target instead of either
# constraint.
# Pinned by scripts/find_home_pose.py on 2026-09-05. The sweep's own winning candidate was
# grasp_center (0.2745, 0.0375, 0.1628) at 23.57 deg off -z, but the joint values below are
# that candidate ROUNDED to 2 decimals, and the rounding moves the tool. What this dict
# actually produces, confirmed against both a numpy FK of the vendored URDF and the live
# Isaac reading (Task D3, /tmp/d3r_tilt_sweep.log), is:
#     grasp_center (0.2749, 0.0371, 0.1646), tool axis 23.85 deg off -z.
# Min joint margin 0.30 rad, on link_005_joint -- the tightest of the six by a factor of
# three, and the joint that limits how far the tool can be tilted toward vertical over the
# scatter band (see GRASP_TILT_DEG).
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
#
# Measured by scripts/scripted_grasp_check.py on 2026-09-06 (Task D3 step 4), on the
# global-IK path. The jaw plates are the whole story here, so their collision box (from the
# vendored STLs, expressed relative to grasp_center) is worth restating:
#
#     x  -0.0625 .. +0.0205   across the plate face
#     y  -0.0303 .. +0.0302   along jaw travel (plus the jaws' own +-0.0524 of slide)
#     z  -0.0342 .. +0.0664   along the tool axis: 34.2 mm PAST grasp_center, 66.4 mm back
#
# So a desk-resting brick is necessarily gripped near the plate TIP, not at grasp_center:
# the tip has to clear the desk, which puts grasp_center at least 34.2 mm up, while the
# brick centre sits at 4.8 mm.
#
# READ THE SHIPPED VALUE FROM THE BOTTOM OF THIS BLOCK, NOT FROM HERE. The sweep below
# examined several offsets and this paragraph describes an intermediate one; the pinned
# value is -0.0321 and its tip margin is 2.1 mm, derived at MIN_DESCEND_TILT and restated
# where it is set. For orientation only: -0.030 m would leave the brick 29.4 mm along the
# tool axis with 4.8 mm of tip margin, and the plates clear of the desk (0.00/0.00 N of
# plate contact, against 46-110 N at shallower offsets). Do not read 4.8 mm as the shipped
# clearance -- it is 2.1 mm, and the factor of two decides whether the plates foul the desk.
#
# Swept over -0.038..-0.010 m; outside a narrow
# window the grasp fails in two distinct ways, and both are visible in the telemetry:
# too shallow puts the brick at the tip and the closing jaw skates over it, too deep jams
# the plates on the desk so the arm stalls short and the realised offset is not the
# commanded one.
#
# IMPORTANT, and the reason this number could not be measured before: at GRASP_TILT_DEG the
# plates foul the desk at EVERY offset. The plate box is 82.5 mm long in x, so a 20 deg tilt
# drops its far corner 0.0625 * sin(20 deg) = 21.4 mm below the axial tip -- computed over
# the whole swept range (/tmp/d3_pick_offset.py), the lowest corner sits 13.3 to 37.8 mm
# BELOW the desk at every offset from -0.040 to -0.014. The number above is therefore
# measured with a LEVEL descend (scripted_grasp_check.py --descend_tilt_deg 0.0), which is
# available precisely because GRASP_TILT_DEG's own comment records that the tilt is a cost
# of the APPROACH only and that the arm is level to within 0.42 deg at the brick at every
# radius. Ruling D3-10 makes that the default: GRASP_TILT_DEG keeps its meaning as the
# APPROACH tilt and is untouched, and the descend runs near level.
#
# NEAR level, never level. At exactly zero tilt the tool axis is vertical, so the lean
# azimuth is undefined and the solver's azimuth constraint goes singular -- measured, the
# continuity-constrained descend then failed at waypoint 39 OF 40, the only one at zero
# tilt, on five of six bricks. core.global_ik.MIN_DESCEND_TILT is 0.5 deg for that reason,
# and the z value below is the offset that balances desk clearance against tip margin AT
# that tilt (core.global_ik.descend_pick_offset), 2.1 mm either side. The descend searches
# a ladder of tilts and recomputes this offset per rung, so this is the nominal, not a
# ceiling.
# The x component is ZERO, and that is a measurement, not an omission. grasp_center is not
# centred on the jaw plates' collision box -- it spans -0.0625..+0.0205 in body x, so
# grasp_center sits 21.0 mm toward the short edge, leaving a brick's 15.9 mm half-length
# only 4.6 mm of plate. Moving the brick to the box's centre (-0.021) looks like the fix and
# is NOT: tried on GPU, the jaws then closed on nothing and pushed the brick to -81.9 mm,
# right off the plate, where centring it was supposed to protect it. The collision box is
# the whole finger BODY -- bracket and rail included -- and the gripping surface is the part
# of it near the tool axis. x = 0 is where the jaws actually grip, evidenced by 4/4 clean
# carries at 6.5-6.9 N of symmetric force with the brick there.
PICK_OFFSET_MEASURED: tuple[float, float, float] = (0.0, 0.0, -0.0321)
# Zero, and not for want of measuring. On the differential-IK path the descend was driven
# into the desk until it stalled, and this term was the extra depth that took the jaw tips
# down to the brick. The global-IK path commands the descend to a POSITION instead, so the
# depth is entirely expressed by PICK_OFFSET_MEASURED above and a second additive term would
# just be degenerate with it. Every measurement above was taken with this at 0.0.
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


# Wrist camera (spec section 3.3). Still UNVERIFIED -- the URDF has no camera frame and the
# SDK publishes no optics, so the lens and the exact bracket remain an arrival
# re-measurement item (spec section 7) -- but no longer a free guess: the pose below is
# BOUNDED BY A HARDWARE FACT. On the physical arm the camera is mounted IN FRONT OF the
# wrist motor, distal to it on the jaw side (owner, 2026-09-06), so the motor cannot appear
# in the image at all. Image "right" is the jaw axis (+y), image "down" is +x.
#
# What that fact costs, measured off the vendored collision STLs (each scaled by 0.001 and
# transformed by its collision origin xyz="-0.1854333 0 0.02930628"
# rpy="1.5707963 -1.3962634 -3.1415927"), in link_007 coordinates:
#
#     body                     x                   y                   z
#     link_007 (motor)  -0.0310 .. +0.1079  -0.0830 .. +0.0830  -0.1128 .. -0.0258
#     gripper_left      -0.0625 .. +0.0200   0.0000 .. +0.0302  -0.2038 .. -0.1032
#     gripper_right     -0.0625 .. +0.0205  -0.0303 ..  0.0000  -0.2038 .. -0.1058
#     grasp_center            0                   0                -0.169639
#
# The motor's FRONT FACE is z = -0.1128, so the lens must sit below it. The first
# placeholder here, (-0.05, 0, -0.06) with a 20 deg pitch toward +x, was 52 mm INSIDE that
# z span -- level with the motor, not in front of it -- and the +x pitch then aimed the
# lens back across the housing. Rendered, that filled the lower half to two thirds of every
# wrist frame with motor and never showed the grasp point (/tmp/camcheck/base_*.png).
#
# Three constraints pin the replacement, and they fight each other:
#   1. z < -0.1128, so the whole of link_007 is behind the lens.
#   2. Clear of the jaw sweep. The jaws slide +-0.0524125 along y, so the volume they
#      travel through is x -0.0625..+0.0205, |y| <= 0.0826, z -0.2038..-0.1032 -- wider in
#      y than the static AABBs above. Only an offset in x clears them at EVERY opening.
#   3. grasp_center (0, 0, -0.169639) centred in frame, jaw tips at the edges.
# (1) and (2) cannot both be met near the tool axis: the only part of link_007 reaching
# z = -0.1128 is the jaw rail, a narrow strip at |x| <= 0.0068 spanning the full y width,
# and directly in front of that strip is inside the jaw sweep. So the camera sits off-axis
# in x -- the axis perpendicular to jaw travel, which also puts the jaws left and right in
# the image instead of across it.
#
# x = +0.030 keeps 9.5 mm of bracket clearance past the jaws' +x extreme (+0.0205);
# z = -0.118 is 5 mm in front of the motor face. That leaves only 51.6 mm of standoff to
# grasp_center for a 30 mm lateral offset, so the view is unavoidably oblique -- that is the
# trade-off this pose makes, and it is forced, not chosen. A 30.0 deg pitch toward -x is
# what buys the obliquity back: the optical axis then crosses x = 0 at z = -0.1700, i.e.
# 0.4 mm from grasp_center, which renders at pixel (112.0, 111.6) of 224x224.
#
# The motor is out of frame by construction, not by luck: every ray of the 70 deg frustum
# (vision.WRIST_FOV_DEG) tilted 30 deg off -z still descends, and link_007 lies entirely
# above the lens. Rendered check: 0.0% of 1428 sampled link_007 collision vertices project
# into the frame, against 18.4% for the old pose (/tmp/camcheck/final_*.png).
WRIST_CAM_UNVERIFIED = True
WRIST_CAM_POS_IN_WRIST: tuple[float, float, float] = (0.030, 0.0, -0.118)
_PITCH = math.radians(30.0)  # toward -x, i.e. back toward the tool axis, NOT across the motor
_FORWARD = (-math.sin(_PITCH), 0.0, -math.cos(_PITCH))
_RIGHT = (0.0, 1.0, 0.0)
_DOWN = (math.cos(_PITCH), 0.0, -math.sin(_PITCH))  # forward x right
WRIST_CAM_ROT_IN_WRIST = quat_wxyz_from_axes(_RIGHT, _DOWN, _FORWARD)
