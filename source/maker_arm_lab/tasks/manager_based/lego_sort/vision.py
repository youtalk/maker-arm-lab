"""Camera-observation variants of LegoSort (design spec section 2.3).

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

from maker_arm_lab.core.cameras import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OVERHEAD_CAM_POS,
    OVERHEAD_CAM_ROT,
    OVERHEAD_FOV_DEG,
    WRIST_FOV_DEG,
)
from maker_arm_lab.robots import WRIST_CAM_POS_IN_WRIST, WRIST_CAM_ROT_IN_WRIST, WRIST_LINK

from .lego_sort_env_cfg import LegoSortEnvCfg
from .variants import LegoSortIKRelEnvCfg

# IMAGE_WIDTH / IMAGE_HEIGHT, the FOVs and the overhead pose live in core.cameras (isaaclab-free)
# so the MuJoCo scene shares them; this module owns only the Isaac cfg objects.

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
# Overhead camera: pose and FOV derived in core.cameras from core.layout.
# --------------------------------------------------------------------------

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


@configclass
class LegoSortJointPosVisionEnvCfg(LegoSortEnvCfg):
    """Joint-position action space + camera observations (the datagen base).

    The same two camera streams as above, on ``LegoSortEnvCfg``'s own joint-position action
    space rather than on the IK-relative one. That is the space the scripted oracle drives
    since Task D5 moved it onto ``core.global_ik`` -- the solver emits joint targets, which
    the IK-Rel term would read as Cartesian deltas -- and it is the space the dataset's
    recorded action stream was already in (``mdp.recorders.JointTargetRecorder``). It reaches
    the registry through ``datagen.LegoSortDatagenEnvCfg`` (``MakerArm-LegoSort-Datagen-v0``)
    rather than under a task id of its own: the spec fixes the id list at five (section 2.3).
    """

    def __post_init__(self):
        super().__post_init__()
        _add_cameras(self)
