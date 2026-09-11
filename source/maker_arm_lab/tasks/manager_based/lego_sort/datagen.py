"""Data-generation env: vision + joint-position actions + domain randomization + recorders.

One PROCESS = one generation run = one draw of the run-level axes (bin layout,
color permutation, camera pose, lighting, floor color) passed in through the
constructor. Reset-level axes (brick scatter, desk color, brick mass/friction,
start pose) are event terms firing per episode. Spec section 2.3 (Datagen-v0).

Lighting and the ground plane are single global prims shared by all parallel
envs -- randomizing them per reset would change them mid-episode for the other
envs, so they are constructor fields here, not event terms (spec section 2.3).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from maker_arm_lab.core import layout
from maker_arm_lab.core.layout import (
    sample_run_draw,  # noqa: F401 -- generate_batch imports it here
)
from maker_arm_lab.core.quat import compose_offset_rot, quat_from_rpy_deg
from maker_arm_lab.robots import ARM_JOINT_REGEX

from . import mdp
from .mdp.recorders import DatagenRecorderManagerCfg
from .vision import LegoSortJointPosVisionEnvCfg

# The quaternion helpers moved to core.quat (isaaclab-free) so the MuJoCo scene applies the
# same camera jitter; the old private names stay as aliases for this module's own use.
_quat_from_rpy_deg = quat_from_rpy_deg
_compose_offset_rot = compose_offset_rot


@configclass
class LegoSortDatagenEnvCfg(LegoSortJointPosVisionEnvCfg):
    """Registered as MakerArm-LegoSort-Datagen-v0.

    THE ACTION SPACE HERE AND THE ORACLE'S SOLVER ARE ONE CHOICE, NOT TWO.

    This inherits the JOINT-POSITION action space, whose term reads its input as absolute
    joint targets and passes them through maker_arm_rs's clamp. Task D5 moved it here from
    the IK-Rel space in the same change that moved the oracle that drives datagen
    (tasks.manager_based.lego_sort.oracle_runtime) onto `core.global_ik`, because the global
    solver emits JOINT TARGETS while the IK-Rel term reads its input as CARTESIAN DELTAS:
    changing either one alone would have fed joint angles to a term interpreting them as
    deltas -- silently, with no error, producing a dataset of nonsense. Anything that moves
    one of the two has to move the other. `OracleRig._check_action_space` is the runtime
    guard on this invariant.
    """

    # Run-level draws (one per generation run, sampled by scripts/generate_batch.py)
    bin_x: float = 0.44
    bin_ys: tuple = (-0.21, -0.07, 0.07, 0.21)
    slot_colors: tuple = (0, 1, 2, 3)
    dome_intensity: float = 3000.0
    dome_color_temp: float = 6500.0
    sun_intensity: float = 0.0
    sun_rpy_deg: tuple = (0.0, 0.0, 0.0)
    floor_color: tuple = (0.0, 0.0, 0.0)
    wrist_cam_dpos: tuple = (0.0, 0.0, 0.0)
    overhead_cam_dpos: tuple = (0.0, 0.0, 0.0)
    overhead_cam_drot_deg: tuple = (0.0, 0.0, 0.0)
    # Feature switches
    cameras: bool = True
    desk_color: bool = True
    physics_dr: bool = True
    start_pose_offset_rad: float = 0.05

    def _bin_layout(self) -> layout.BinLayout:
        return layout.BinLayout(
            bin_x=self.bin_x, bin_ys=tuple(self.bin_ys), slot_colors=tuple(self.slot_colors)
        )

    def __post_init__(self):
        super().__post_init__()
        # 6-brick oracle episode budget: 180 s (5400 control steps at 30 Hz).
        # Carried over from openyam-lab, where it was MEASURED in Task 8's
        # oracle_check.py GPU runs at 580-750 control steps per brick, and
        # RE-MEASURED ON THIS ARM by Task D5's own scripts/oracle_check.py, the
        # first thing to run a whole episode here: 4316-4436 control steps for a
        # six-brick sort, every one of six episodes terminating on `success`
        # rather than on this time_out, so the budget holds with roughly 1000
        # steps of headroom. The per-brick figure is comparable (about 650-750
        # including the one-time 240-step SETTLE) even though the control model
        # underneath changed completely -- the oracle now walks planned
        # global-IK paths instead of servoing IK-Rel deltas.
        self.episode_length_s = 180.0

        # Lighting / floor: run-level, via the global prims' spawn cfgs.
        self.scene.light.spawn = sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=self.dome_intensity,
            enable_color_temperature=True,
            color_temperature=self.dome_color_temp,
        )
        if self.sun_intensity > 0.0:
            self.scene.sun = AssetBaseCfg(
                prim_path="/World/sun",
                init_state=AssetBaseCfg.InitialStateCfg(rot=_quat_from_rpy_deg(self.sun_rpy_deg)),
                spawn=sim_utils.DistantLightCfg(intensity=self.sun_intensity),
            )
        self.scene.plane.spawn = sim_utils.GroundPlaneCfg(color=tuple(self.floor_color))

        # Camera pose jitter (run-level), layered on top of each camera's
        # configured base offset (vision.py): the overhead base is derived
        # from core.layout, and the wrist base is the UNVERIFIED placeholder
        # mount (robots.frames.WRIST_CAM_POS_IN_WRIST / WRIST_CAM_ROT_IN_WRIST).
        if self.cameras:
            wrist = self.scene.wrist_cam.offset
            wrist.pos = tuple(p + d for p, d in zip(wrist.pos, self.wrist_cam_dpos))
            over = self.scene.overhead_cam.offset
            over.pos = tuple(p + d for p, d in zip(over.pos, self.overhead_cam_dpos))
            over.rot = _compose_offset_rot(over.rot, self.overhead_cam_drot_deg)
        else:
            self.scene.wrist_cam = None
            self.scene.overhead_cam = None
            self.observations.images = None

        # Contact sensors: the oracle's pinch evidence. This wiring is carried
        # over from openyam-lab, where it was moved here from an M1 script.
        # Filter lists in brick-index order.
        self.scene.robot.spawn.activate_contact_sensors = True
        brick_filters = [f"{{ENV_REGEX_NS}}/brick_{i}" for i in range(self.num_bricks)]
        self.scene.finger_left_brick = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
            history_length=1,
            filter_prim_paths_expr=brick_filters,
        )
        self.scene.finger_right_brick = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
            history_length=1,
            filter_prim_paths_expr=brick_filters,
        )

        # Reset-level randomization (event order: after reset_all/randomize_bricks).
        if self.desk_color:
            self.events.randomize_desk_color = EventTerm(
                func=mdp.randomize_desk_color,
                mode="reset",
                params={"color_range": {"r": (0.3, 0.7), "g": (0.25, 0.55), "b": (0.2, 0.45)}},
            )
        if self.physics_dr:
            for i in range(self.num_bricks):
                setattr(
                    self.events,
                    f"brick_{i}_mass",
                    EventTerm(
                        func=mdp.randomize_rigid_body_mass,
                        mode="reset",
                        params={
                            "asset_cfg": SceneEntityCfg(f"brick_{i}"),
                            "mass_distribution_params": (0.7, 1.3),
                            "operation": "scale",
                        },
                    ),
                )
                setattr(
                    self.events,
                    f"brick_{i}_friction",
                    EventTerm(
                        func=mdp.randomize_rigid_body_material,
                        mode="reset",
                        params={
                            "asset_cfg": SceneEntityCfg(f"brick_{i}", body_names=".*"),
                            "static_friction_range": (0.8, 1.2),
                            "dynamic_friction_range": (0.7, 1.1),
                            "restitution_range": (0.0, 0.0),
                            "num_buckets": 16,
                            "make_consistent": True,
                        },
                    ),
                )
        if self.start_pose_offset_rad > 0.0:
            self.events.randomize_start_pose = EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-self.start_pose_offset_rad, self.start_pose_offset_rad),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[ARM_JOINT_REGEX]),
                },
            )

        # Recorders: paths/mode are set by scripts/generate_batch.py per run.
        self.recorders = DatagenRecorderManagerCfg()
