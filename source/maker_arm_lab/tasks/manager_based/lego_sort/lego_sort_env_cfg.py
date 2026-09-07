"""LegoSort manager-based environment (design spec section 5.3).

Modeled on isaaclab_tasks' stack task
(``manager_based/manipulation/stack/stack_env_cfg.py`` +
``config/franka/stack_joint_pos_env_cfg.py``, Isaac Lab v2.3.2): a static
desk/shelf scene, joint-position arm action + binary gripper action, state
observations, and a reset-scatter event term. Bins (20 static boxes) and
bricks (``num_bricks`` rigid objects) are config-dependent, so they are
attached in ``__post_init__`` rather than declared as scene fields.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from maker_arm_lab.core import layout
from maker_arm_lab.robots import (
    ARM_JOINT_REGEX,
    GRIPPER_CLOSED_POS,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_POS,
    MAKER_ARM_CFG,
)

from . import mdp


@configclass
class LegoSortSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MAKER_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    desk = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        init_state=AssetBaseCfg.InitialStateCfg(pos=layout.DESK_POS),
        spawn=sim_utils.CuboidCfg(
            size=layout.DESK_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.4, 0.3)),
        ),
    )
    shelf = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        init_state=AssetBaseCfg.InitialStateCfg(pos=layout.SHELF_POS),
        spawn=sim_utils.CuboidCfg(
            size=layout.SHELF_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
        ),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.75)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    # bins (20 static boxes) and bricks (num_bricks rigid objects) are attached
    # in LegoSortEnvCfg.__post_init__ -- counts are config-dependent.


@configclass
class ActionsCfg:
    arm_action = mdp.ClampedJointPositionActionCfg(
        asset_name="robot", joint_names=[ARM_JOINT_REGEX], scale=1.0, use_default_offset=False
    )
    # Gripper: URDF-native, 0 = closed, GRIPPER_OPEN_POS = open, both jaws driven to the same
    # target (the mimic was converted to two joints). `raw action >= 0` opens. Direction
    # confirmed physically by scripts/scripted_grasp_check.py (Task D3).
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(GRIPPER_JOINT_NAMES),
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        brick_poses = ObsTerm(func=mdp.brick_poses_w)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.all_bricks_stowed)


@configclass
class EventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # Fires once, right after the scene is built (before any reset/step): catches
    # cfg.num_bricks being mutated after LegoSortEnvCfg construction, which would
    # otherwise silently desync it from the brick_* entities __post_init__ already
    # built (code review finding, Task 7).
    validate_brick_count = EventTerm(func=mdp.validate_brick_count, mode="startup")
    # scatter term appended in __post_init__ (needs num_bricks)


@configclass
class LegoSortEnvCfg(ManagerBasedRLEnvCfg):
    scene: LegoSortSceneCfg = LegoSortSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    commands = None
    rewards = None
    curriculum = None

    num_bricks: int = 6
    num_arms: int = 1
    # Contact-escalation last resort from the design spec's section 5.3 ladder.
    # Carried over from OpenYAM's M1 LegoSort work, where it was never needed
    # (12/12 scripted grasps at real size on that arm); not yet re-validated on
    # the Maker Arm -- scripts/scripted_grasp_check.py (Task D3) is what would
    # tell us if this needs to move. Values other than 1.0 are rejected in
    # __post_init__ -- see the guard there for why.
    brick_scale: float = 1.0

    def _bin_layout(self) -> layout.BinLayout:
        """Bin geometry/color assignment hook; datagen cfg overrides from its fields."""
        return layout.DEFAULT_BIN_LAYOUT

    def __post_init__(self):
        if self.num_arms != 1:
            raise NotImplementedError("LegoSort implements single-arm only (spec section 5.3)")
        if not 4 <= self.num_bricks <= 8:
            raise ValueError("num_bricks must be in [4, 8] (spec section 5.3)")
        if self.brick_scale != 1.0:
            # Guarded rather than left as an unguarded public knob, because
            # what it produces when used is an inconsistent scene, silently:
            # the geometry scales but BRICK_MASS does not (so density falls as
            # the cube), MIN_SEPARATION does not, and layout's bin interior
            # AABB -- what the success predicate grades against -- does not
            # either. Enlarged bricks are a real rung of the spec section 5.3
            # escalation ladder, so this is a "finish the job" marker, not a
            # ban: scale the mass, the separation and the bin AABB together
            # and re-measure, then lift the guard.
            raise NotImplementedError(
                "brick_scale != 1.0 scales brick geometry only -- BRICK_MASS, "
                "MIN_SEPARATION and the graded bin interior AABB do not follow, "
                "so the scene becomes inconsistent. OpenYAM's M1 never needed it "
                "(12/12 scripted grasps at real size on that arm); this has not "
                "been re-checked on the Maker Arm."
            )
        self.decimation = 4
        self.episode_length_s = 60.0
        self.sim.dt = 1.0 / 120.0  # 30 Hz control (spec: 15-30 Hz)
        self.sim.render_interval = self.decimation

        bl = self._bin_layout()
        self.bin_layout = bl  # read by mdp.terminations (Task 2)
        for name, center, size in layout.bin_static_boxes(bl):
            color = name.split("_")[1]
            setattr(
                self.scene,
                name,
                AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/" + name,
                    init_state=AssetBaseCfg.InitialStateCfg(pos=center),
                    spawn=sim_utils.CuboidCfg(
                        size=size,
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=layout.BRICK_RGBA[color][:3]
                            if name.endswith("floor")
                            else (0.55, 0.55, 0.55)
                        ),
                    ),
                ),
            )

        scale = self.brick_scale
        # Default (pre-first-reset) rest pose for each brick, laid out inside
        # layout.SCATTER_X/SCATTER_Y rather than hard-coded. Code review (Task 7)
        # found the brief's original `x = 0.25 + 0.05 * index` put brick_3/brick_4
        # (x=0.40/0.45) inside the shelf pedestal, which occupies x in [0.39, 0.49] --
        # PhysX depenetration silently shoves the overlapping bodies apart before the
        # first physics step, and randomize_brick_poses's reset-mode scatter overwrites
        # this pose on the very first reset regardless, so normal operation never
        # surfaces it -- but the constructed-and-not-yet-reset scene is genuinely
        # invalid, and would bite anything that inspects or steps the env before the
        # first reset, or that disables the scatter event (a plausible Task 10 move).
        # This 4-column x 2-row grid keeps every brick up to num_bricks=8 inside the
        # scatter zone and mutually clear (rows 0.10 m apart vs. the brick's 0.0158 m
        # width -- axis-aligned, since these are unrotated).
        #
        # The column spacing is DERIVED from the zone rather than fixed, so the grid
        # cannot spill out of SCATTER_X when that moves. It was a fixed 0.05 m, which
        # fitted the old (0.18, 0.36) zone with room to spare and then landed the last
        # column exactly on the upper bound once Task 10 shifted the zone to
        # (0.13, 0.29). Four columns inset by COLUMN_MARGIN at each end gives
        # (0.29 - 0.13 - 0.04) / 3 = 0.04 m of spacing today: 8.2 mm of clear air
        # between adjacent 0.0318 m bricks, and a worst-case column-3 centre of
        # 0.27 m. Adding the 0.0177 m worst-case yaw half-diagonal layout.py uses
        # gives 0.2877 m, 102 mm clear of the shelf's near face at x = 0.39 m.
        column_margin = 0.02
        col_spacing = (layout.SCATTER_X[1] - layout.SCATTER_X[0] - 2 * column_margin) / 3
        row_spacing = 0.10
        y_mid = (layout.SCATTER_Y[0] + layout.SCATTER_Y[1]) / 2
        for index in range(self.num_bricks):
            color = layout.BRICK_COLOR_ORDER[index % 4]
            col, row = index % 4, index // 4
            brick_x = layout.SCATTER_X[0] + column_margin + col * col_spacing
            brick_y = y_mid + (row - 0.5) * row_spacing
            setattr(
                self.scene,
                f"brick_{index}",
                RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/" + f"brick_{index}",
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(brick_x, brick_y, layout.SPAWN_Z)
                    ),
                    spawn=sim_utils.CuboidCfg(
                        size=tuple(s * scale for s in layout.BRICK_SIZE),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            solver_position_iteration_count=16,
                            solver_velocity_iteration_count=1,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(mass=layout.BRICK_MASS),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=layout.BRICK_RGBA[color][:3]
                        ),
                    ),
                ),
            )

        self.events.randomize_bricks = EventTerm(
            func=mdp.randomize_brick_poses,
            mode="reset",
            params={
                "pose_range": {
                    "x": layout.scatter_x(bl),
                    "y": layout.SCATTER_Y,
                    "z": (layout.SPAWN_Z, layout.SPAWN_Z),
                    "yaw": (-3.14159, 3.14159),
                },
                "min_separation": layout.MIN_SEPARATION,
                "max_radius": layout.SCATTER_MAX_RADIUS,
                "asset_names": [f"brick_{i}" for i in range(self.num_bricks)],
            },
        )
