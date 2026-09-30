"""Camera constants shared by the Isaac and MuJoCo scenes (isaaclab-free).

The wrist camera POSE lives in ``robots.frames`` (it is a frame on the arm); what lives here
is the optics and the fixed overhead camera, which follow ``core.layout`` rather than the
robot. ``tasks/manager_based/lego_sort/vision.py`` imports these to build the Isaac
``TiledCameraCfg`` pair, and ``maker_arm_lab.mj.scene`` imports them to build the MuJoCo
cameras, so a change here moves both simulators together.
"""

import math

from maker_arm_lab.core import layout

# Spec section 4.2 invariant: both streams are 224x224 RGB. Square, so the horizontal and
# vertical fields of view are equal for square pixels.
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# Wrist camera field of view, degrees. Kept at 70.0 deliberately, not left alone. The mount
# clears the jaws by offsetting 30 mm along +x over a 51.6 mm standoff to grasp_center
# (robots.frames), and at that range the FOV decides whether the jaws frame the grasp point
# or swallow it. Rendered and measured at the home pose on the Isaac scene:
#
#     FOV     fully-open jaw tip      what enters the top of the frame
#     70     u = 218.0 of 224        nothing but desk (top ray lands at x=0.062)
#     85     u = 193.0               the arm's own base (top ray lands at x=-0.004)
#    100     u = 174.3               ditto, further behind the base (x=-0.097)
#
# So 70 deg is very nearly the exact FOV that puts the fully-open jaw tips ON the frame
# edge: wider pulls the jaws inward, shrinks the brick, and starts imaging the robot's own
# column instead of the workspace. At the home pose the frame covers 0.32 m of desk at
# 0.23 m range, spanning core.layout's scatter band.
WRIST_FOV_DEG = 70.0
WRIST_CLASSIFIER_PX = 448
WRIST_CLASSIFIER_CROP = (152, 376, 116, 340)
"""The MS7f part classifier's wrist view (``MjLegoSortEnv.wrist_crop``): the wrist camera at the
same pose and field of view rendered at 448 px, and the rows and columns of the 224 px crop
around the fingertips at native pixels, the 112 px region of a 224 px frame that the classifier
read before. The pick point sits about 0.089 m from the camera, so the crop spans 0.062 m at
0.28 mm a pixel there: the assumption for the real rig is a wrist camera that gives about 0.3 mm
a pixel at the jaws. The policy's 224 px wrist frames do not change."""

# Overhead camera: straight down over the desk, framing the brick scatter zone AND all four
# bins. Every number is derived from core.layout so that moving the scatter zone or the bins
# re-derives the pose instead of silently de-framing the view.
OVERHEAD_FOV_DEG = 60.0
# Fraction of slack around the region that must be visible.
OVERHEAD_MARGIN = 1.15
OVERHEAD_CAM_POS = layout.overhead_view_pose(OVERHEAD_FOV_DEG, OVERHEAD_MARGIN)
OVERHEAD_B2_BACK_M = 0.139
"""How far the overhead camera's nominal x moves back under MS7e rung B2 (``--release_lean``,
MS7e design section 4.4 item 3), m: 0.3011 -> 0.1621. The episode's camera draw around it is
unchanged (+-0.01 m and +-2 degrees per axis).

The smallest shift that keeps two things in frame at every corner of that draw, by projection
(2026-09-29): every rim corner of a B2 box at least 4 px inside, and the tray inside. The worst
box is a 0.10 m square interior, 0.05 m deep, yawed 45 degrees, at the band's back corner x
-0.07. At 0.139 m its corner keeps 4.18 px (the near edge binds; the sides keep 13.82 px and the
far edge 49.12 px), and the 30 degree leaned tray keeps 67.51 px. With no shift the same corner
leaves the frame by 36.6 px at x -0.06."""

NARROW_FOV_DEG = 9.1
"""The narrow overhead view (MS7f design section 6.3, view 1), degrees: 224 px at 0.5 mm a pixel,
the pixel of a 224 px crop of the real rig's 1920 px frame, at 0.704 m, the B2 camera's
distance to the tray centre. A fixed angle, as a crop of a fixed frame is: over the tray
interior, parts 0.005 to 0.03 m high sit 0.684 to 0.722 m away, 0.486 to 0.513 mm a pixel
(2026-09-30). ``MjLegoSortEnv.narrow_overhead`` turns the overhead camera onto the target."""

# Look straight down with world +x ("away from the arm base", toward the bins) as image-up.
# In the ROS camera convention (+z = view direction, +x = image-right, +y = image-down) that is
#   z_cam = (0, 0, -1)   x_cam = (0, -1, 0)   y_cam = (-1, 0, 0)
# whose quaternion is a 180 deg turn about (1, -1, 0)/sqrt(2). This composition (bins along
# the top edge, arm base at the bottom, +y to the left) was confirmed by rendering.
_SQRT1_2 = math.sqrt(0.5)
OVERHEAD_CAM_ROT = (0.0, _SQRT1_2, -_SQRT1_2, 0.0)

# A ROS-convention camera (x right, y down, z forward) and a MuJoCo camera (x right, y up,
# looks along -z) differ by a half turn about the camera's own x axis.
ROS_TO_MUJOCO_CAM = (0.0, 1.0, 0.0, 0.0)
