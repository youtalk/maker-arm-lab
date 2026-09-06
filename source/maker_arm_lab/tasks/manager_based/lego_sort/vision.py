"""Camera-observation variants of LegoSort (design spec section 5.3).

Adds an ``images`` observation group carrying two 224x224 RGB streams -- a
wrist camera and a fixed overhead camera -- to the joint-position and
IK-relative action spaces. Modeled on isaaclab_tasks'
``manipulation/stack/config/franka/stack_ik_rel_visuomotor_env_cfg.py``
(Isaac Lab v2.3.2) for the camera-cfg + ``mdp.image`` obs-term idiom, with
``TiledCameraCfg`` in place of that file's ``CameraCfg`` (tiled rendering is
the batched-env path; the cfg fields are identical, ``TiledCameraCfg`` only
swaps ``class_type``).

The overhead camera pose is derived from `core.layout` (unchanged). The wrist
camera pose is still UNVERIFIED -- the vendor URDF has no camera frame and the
SDK publishes no optics, so `robots.frames.WRIST_CAM_POS_IN_WRIST` /
`WRIST_CAM_ROT_IN_WRIST` / `WRIST_FOV_DEG` are re-measured when the arm arrives
(spec section 7) -- but it is no longer a free guess. It is bounded by a hardware
fact, that the camera sits IN FRONT OF the wrist motor, and by the arm's own
measured collision geometry; `robots.frames` carries that derivation. The mount
body is `link_007` (the wrist); the offset is expressed in its frame.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.observations import image
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from maker_arm_lab.core import layout
from maker_arm_lab.robots import WRIST_CAM_POS_IN_WRIST, WRIST_CAM_ROT_IN_WRIST, WRIST_LINK

from .lego_sort_env_cfg import LegoSortEnvCfg
from .variants import LegoSortIKRelEnvCfg

# Spec section 5.3 invariant: both streams are 224x224 RGB. Square, so the
# horizontal and vertical fields of view are equal for square pixels (Isaac
# Lab derives vertical_aperture = horizontal_aperture * height / width when
# it is left at None -- isaaclab/sensors/camera/camera.py).
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# Isaac Lab's default horizontal aperture (a 35 mm spherical projector's).
# Kept fixed so that a field of view maps to exactly one focal length.
_APERTURE = 20.955


def _focal_for_fov(fov_deg: float) -> float:
    """Focal length (cm, PinholeCameraCfg's unit) giving ``fov_deg`` at ``_APERTURE``."""
    return 0.5 * _APERTURE / math.tan(math.radians(fov_deg) / 2.0)


# --------------------------------------------------------------------------
# Wrist camera
# --------------------------------------------------------------------------
# UNVERIFIED (spec section 7): the vendor URDF has no camera frame of its own,
# and the vendor SDK publishes no optics, so there is no asset-derived pose to
# mount on the way the overhead camera derives from core.layout below, and the
# real lens is an arrival re-measurement item. WRIST_LINK (link_007, the wrist)
# is the mount body; robots.frames.WRIST_CAM_POS_IN_WRIST /
# WRIST_CAM_ROT_IN_WRIST are the offset in that body's frame, derived there
# from the measured motor/jaw collision bounds and from the hardware fact that
# the camera is mounted in front of the wrist motor (front face z = -0.1128 in
# link_007). What follows is the FOV half of that derivation.
_WRIST_CAM_PRIM_PATH = "{ENV_REGEX_NS}/Robot/" + WRIST_LINK + "/wrist_cam"

# Kept at 70.0 deliberately, not left alone. The mount clears the jaws by
# offsetting 30 mm along +x over a 51.6 mm standoff to grasp_center
# (robots.frames), and at that range the FOV decides whether the jaws frame the
# grasp point or swallow it. Rendered and measured at the home pose:
#
#     FOV     fully-open jaw tip      what enters the top of the frame
#     70     u = 218.0 of 224        nothing but desk (top ray lands at x=0.062)
#     85     u = 193.0               the arm's own base (top ray lands at x=-0.004)
#    100     u = 174.3               ditto, further behind the base (x=-0.097)
#
# So 70 deg is very nearly the exact FOV that puts the fully-open jaw tips ON
# the frame edge: wider pulls the jaws inward, shrinks the brick, and starts
# imaging the robot's own column instead of the workspace. It is wide enough for
# the workspace -- at the home pose the frame covers 0.32 m of desk at 0.23 m
# range, spanning core.layout's scatter band. Widening was the obvious lever for
# the short standoff and it was tested and rejected on the renders, not assumed.
WRIST_FOV_DEG = 70.0

_WRIST_CAM = TiledCameraCfg(
    prim_path=_WRIST_CAM_PRIM_PATH,
    update_period=0.0,
    height=IMAGE_HEIGHT,
    width=IMAGE_WIDTH,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=_focal_for_fov(WRIST_FOV_DEG),
        horizontal_aperture=_APERTURE,
        # Near plane kept small: the mount sits 51.6 mm from grasp_center and
        # closer still to the jaw bodies, because clearing the wrist motor
        # leaves it no more standoff than that (WRIST_CAM_POS_IN_WRIST).
        # Far plane covers the whole desk from any arm pose.
        clipping_range=(0.01, 5.0),
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=WRIST_CAM_POS_IN_WRIST, rot=WRIST_CAM_ROT_IN_WRIST, convention="ros"
    ),
    # NOT the Isaac Lab default (False). At False, `data.pos_w`/`quat_w_ros`
    # are a snapshot taken at scene init -- i.e. the USD's zero-joint rest
    # pose -- and never track the arm. This was measured on openyam-lab's
    # wrist mount (a stale pos_w tens of centimeters and tens of degrees from
    # the true, moving body); the mechanism is a generic Isaac Lab one --
    # any body other than the env root can move after the init snapshot --
    # so it applies to this mount too regardless of its own placeholder
    # status. RENDERING is unaffected (the RTX path uses the prim's live
    # transform) and `mdp.image` never reads `pos_w`, so no observation is
    # wrong at False. But the datagen recorders (datagen.py) store wrist
    # extrinsics, and a silently stale rest-pose extrinsic there is a far
    # worse bug than the XformPrimView read this costs. `overhead_cam` below
    # deliberately leaves it at the default: that camera is parented to the
    # env root and never moves, so its init snapshot is already exact --
    # verified on openyam-lab (it reported its configured pose to 6 dp);
    # Task C10's GPU smoke is what confirms the same holds on this scene.
    update_latest_camera_pose=True,
)

# --------------------------------------------------------------------------
# Overhead camera
# --------------------------------------------------------------------------
# Straight down over the desk, framing the brick scatter zone AND all four
# bins. Every number is derived from maker_arm_lab.core.layout so that moving
# the scatter zone or the bins re-derives the pose instead of silently
# de-framing the view (SCATTER_X already moved once, in openyam-lab's M1
# Task 10; scripts/release_envelope_check.py, Task D4, is what re-validates
# the scatter band and bin positions on this arm).
OVERHEAD_FOV_DEG = 60.0
# Fraction of slack around the region that must be visible.
_OVERHEAD_MARGIN = 1.15

# The derivation itself lives in maker_arm_lab.core.layout, which is torch/stdlib
# only and therefore CPU-testable -- see tests/test_layout.py, which perturbs
# SCATTER_X and asserts the resulting frame still contains the scatter zone and
# the bins. Keeping it here would have left a GPU render as the only thing
# guarding the invariant the pose exists to satisfy.
OVERHEAD_CAM_POS = layout.overhead_view_pose(OVERHEAD_FOV_DEG, _OVERHEAD_MARGIN)

# Look straight down with world +x ("away from the arm base", toward the bins)
# as image-up. In the ROS camera convention Isaac Lab uses for `rot`
# (+z = view direction, +x = image-right, +y = image-down) that is
#   z_cam = (0, 0, -1)   x_cam = (0, -1, 0)   y_cam = (-1, 0, 0)
# whose quaternion is a 180 deg turn about (1, -1, 0)/sqrt(2). Written from
# those axes rather than copied from a reference file. This composition
# (bins along the top edge, arm base at the bottom, +y to the left) was
# confirmed by rendering on openyam-lab; it follows from core.layout alone,
# not from the robot asset, so it is expected to hold here unchanged -- Task
# C10's GPU smoke is what re-confirms it on this scene.
_SQRT1_2 = math.sqrt(0.5)
OVERHEAD_CAM_ROT = (0.0, _SQRT1_2, -_SQRT1_2, 0.0)

_OVERHEAD_CAM = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/overhead_cam",
    update_period=0.0,
    height=IMAGE_HEIGHT,
    width=IMAGE_WIDTH,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=_focal_for_fov(OVERHEAD_FOV_DEG),
        horizontal_aperture=_APERTURE,
        clipping_range=(0.05, 5.0),
    ),
    offset=TiledCameraCfg.OffsetCfg(pos=OVERHEAD_CAM_POS, rot=OVERHEAD_CAM_ROT, convention="ros"),
)


@configclass
class ImagesObsCfg(ObsGroup):
    """224x224 RGB from both cameras, uint8, unconcatenated."""

    wrist_rgb = ObsTerm(
        func=image,
        params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
    )
    overhead_rgb = ObsTerm(
        func=image,
        params={
            "sensor_cfg": SceneEntityCfg("overhead_cam"),
            "data_type": "rgb",
            "normalize": False,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        # Keep the two streams separate: they are (N, 224, 224, 3) uint8 and
        # concatenating them into a flat vector would destroy the layout every
        # visual encoder needs. normalize=False for the same reason -- the
        # encoder owns normalisation, and uint8 keeps the replay buffer small.
        self.concatenate_terms = False


def _add_cameras(cfg: LegoSortEnvCfg) -> None:
    """Attach both cameras and the ``images`` observation group to ``cfg``.

    Warning for anyone adding a THIRD camera here (recording, extra views):
    on openyam-lab, with six or seven ``TiledCamera`` sensors in one scene,
    the last-declared one came back at a pose that was not the one configured
    -- twice, in two independent probe runs (task-12-report.md). It did not
    reproduce there with the two cameras below (``overhead_cam`` is last here
    and was exact, in every env and at every arm pose tested on that arm).
    That has not been independently re-checked on this scene. Do not assume
    it stays true: dump a frame and check ``sensor.data.pos_w`` against the
    configured pose after adding one.
    """
    cfg.scene.wrist_cam = _WRIST_CAM.copy()
    cfg.scene.overhead_cam = _OVERHEAD_CAM.copy()
    cfg.observations.images = ImagesObsCfg()
    # RTX sensors return the PREVIOUS frame's buffer unless the renderer is
    # pumped again after a reset writes the new poses, so the first obs of an
    # episode would otherwise show the previous episode's scatter. Same value
    # as the Isaac Lab visuomotor reference.
    cfg.num_rerenders_on_reset = 3


@configclass
class LegoSortIKRelVisionEnvCfg(LegoSortIKRelEnvCfg):
    """IK-relative action space + camera observations (teleop/recording target)."""

    def __post_init__(self):
        super().__post_init__()
        _add_cameras(self)
