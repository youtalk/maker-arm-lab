"""The LegoSort scene as a MuJoCo model: vendored URDF + core.layout, nothing retyped.

One ``build_model`` call is one generation run's draw (``core.layout.sample_run_draw``): bin
layout and color permutation, lighting, floor color, camera jitter. Per-episode randomization
(brick scatter, desk color, brick mass and friction, start pose) is applied by ``env`` on the
compiled model at reset.

Mapping from the Isaac task, in the order the Isaac cfgs declare it:

* Arm actuators: ``MITModeActuator`` is ``tau = kp (q_des - q) + kd (qd_des - qd) + tau_ff``
  clamped to the motor rating, with ``qd_des = 0`` and ``tau_ff`` = gravity. A MuJoCo
  ``position`` actuator is ``kp ctrl - kp q - kv qd`` clamped to ``forcerange``, so with
  ``kp``/``kd`` from ``core.profile`` and ``forcerange`` = the rating it is the same law;
  gravity is taken out of the actuator by ``body.gravcomp = 1`` on the arm links instead of
  being fed forward, which leaves the profile's ``tau_max`` cap on the feed-forward inert.
* ``ARM_ARMATURE`` goes onto ``joint.armature`` with the same role (the explicit-damping
  stability floor for the light wrist joints).
* The gripper's implicit PD (``GRIPPER_STIFFNESS`` / ``GRIPPER_DAMPING`` / effort limit) is a
  second pair of position actuators; both jaws are driven, as in Isaac (the mimic is dropped
  by both importers).
* ``sim.dt = 1/120`` with ``decimation = 4`` is ``timestep = 1/120`` and four ``mj_step`` per
  control step (``env``), 30 Hz control.
* Cameras: Isaac's ``OffsetCfg(convention="ros")`` quaternions are composed with a half turn
  about x (``core.cameras.ROS_TO_MUJOCO_CAM``); square images make Isaac's horizontal FOV and
  MuJoCo's ``fovy`` the same number.
* Lights: Isaac's ``DomeLightCfg`` is an environment light (the whole sky), its optional
  ``DistantLightCfg`` sun the only shadow caster. MuJoCo has no environment light, so the dome
  is a shadowless top-down directional light plus an ambient term of the same strength (a
  uniform sky lights a vertical face at exactly half a horizontal one), and the sun is a second
  directional light with its shadow. The gains are calibrated against MS2 Isaac frames
  (``DOME_GAIN`` and friends below).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from maker_arm_lab.assets.urdf_prep import (
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    MOUNT_RPY,
    VENDORED_URDF,
)
from maker_arm_lab.core import cameras, desk_boxes, layout
from maker_arm_lab.core import parts as _parts
from maker_arm_lab.core import profile as _profile
from maker_arm_lab.core.global_ik import _rpy_matrix
from maker_arm_lab.core.quat import compose_offset_rot, quat_from_rpy_deg, quat_mul
from maker_arm_lab.robots.frames import (
    ARM_ARMATURE,
    GRIPPER_DAMPING,
    GRIPPER_EFFORT_LIMIT_N,
    GRIPPER_STIFFNESS,
    TOOL_BODY,
    WRIST_CAM_POS_IN_WRIST,
    WRIST_CAM_ROT_IN_WRIST,
    WRIST_LINK,
)

# The 8-wide joint vector every consumer of this backend sees: URDF order, arm then jaws.
JOINT_NAMES: tuple[str, ...] = (*ARM_JOINT_NAMES, *GRIPPER_JOINT_NAMES)
ARM_BODIES: tuple[str, ...] = (
    "link_002",
    "link_003",
    "link_004",
    "link_005",
    "link_006",
    "link_007",
    "gripper_left",
    "gripper_right",
)
# 240 Hz physics, eight substeps per 30 Hz control step (Isaac ran PhysX at 120 Hz with an
# unconditionally stable implicit joint drive; MuJoCo's position actuator on the 60 g jaw at
# 1000 N/m has a 49 ms natural period, and 120 Hz gave six samples of it -- the jaws bounced
# past their stops on the first oracle run, 2026-09-10).
PHYSICS_DT = 1.0 / 240.0
CONTROL_SUBSTEPS = 8
# Joint-limit constraint stiffness. MuJoCo's default solref (0.02 s, 1) makes a limit on a
# 60 g jaw about as stiff as a 150 N/m spring, which the 20 N gripper actuator pushes 9 mm past
# (measured); 2 ms puts the same limit near 15 kN/m, a 1 mm excursion at full force.
LIMIT_SOLREF = (0.002, 1.0)
# Contact stiffness for the brick-jaw and brick-desk pairs (the robot's collision geoms and the
# bricks both carry it; MuJoCo averages the pair). At the default (0.02 s, 1) the 2.9 N grip
# sank the jaws 5 mm into a 3 g brick and the pinch read 2.9 mm of stall against the 4.8 mm
# PINCH_MIN_STALL floor; at 5 ms the same close stalls at 6.1 mm with 6.1 N per jaw and the
# brick lifts (measured 2026-09-10, one-brick oracle trials), against Isaac's 7.4-7.7 mm and
# 7.6 N. The direct form (-50000, -300) matched Isaac closer (7.2 mm, 7.6 N) but puts a 50 kN/m
# spring under every resting brick; the time-constant form is the standard, safer choice.
CONTACT_SOLREF = (0.005, 1.0)
# The desk is a hard surface (owner decision 2026-09-18: the parts sank into it). The
# time-constant form scales a contact's stiffness with the lighter body, so a 3 g part under
# the jaws' 2.9 N grip sank 1.9 mm into the desk and 13.0 mm under 20 N, more than its own
# height (measured 2026-09-18 on a lone 2x4 brick). The lever is the impedance, not the time
# constant: at (0.99, 0.999) the same presses read 0.16 mm and 0.55 mm, and a settled
# 50-part heap rests at 0.01 mm mean. `solmix` weights the pair's solref and solimp toward
# the desk's values and nothing else: a geom priority would also hand the pair the desk's
# condim 3 and friction 1.0, which drops the torsional friction of BRICK_CONDIM and the
# per-episode friction draw of mj/env.py for every part-desk pair, in the six-brick task
# too (branch review, 2026-09-18). The jaw-part and part-part pairs the pinch tuning sits on
# are untouched either way.
DESK_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)
DESK_SOLMIX = 1.0e6

# Contact tuning for a 3 g brick between convex-hull finger meshes. Nominal sliding friction
# is the centre of the Isaac static-friction DR range (0.8-1.2); torsional and rolling terms
# are MuJoCo's defaults. condim 4 adds torsional friction so a gripped brick does not spin
# about the pinch axis. ``noslip_iterations`` is the standard remedy for a light object
# creeping out of a soft-contact grip; the GRASP parity check (tests/test_mj_oracle.py) is
# what re-tunes these if it fails.
BRICK_FRICTION = (1.0, 0.005, 0.0001)
BRICK_CONDIM = 4
# MS7e design section 6.4: under the round family, a round part's studs collide and its
# contacts take condim 6, which adds rolling friction to the torsional term of BRICK_CONDIM.
# Studs cannot stop a cylinder on its side from rolling about its own axis, since they lie on
# that axis; the rolling term is what can. The rolling probe of 2026-09-29 dropped each round
# part 48 times into the leaned tray and kicked it 48 times at rest, at 0.1 m/s (the jaw
# speed) across its axis. At 1e-3 a part on its side rolled 1.3 to 8.5 mm (p90 per part) and
# every part came to rest within 0.45 s. At 3e-4 a round 4x4 rolled 22 mm and 37 of 48
# round 1x1 drops rocked in place for the whole 6 s. At 2e-3 a round 2x2 balanced on its rim
# edge in 17 of 48 rests. With condim 4, collision studs or not, the rolls reached 83 to
# 134 mm (p90), and 8 to 27 of 48 kicked parts still rolled after 6 s.
ROUND_CONDIM = 6
ROUND_ROLLING_FRICTION = 0.001  # INITIAL: the lone-part table (MS7e design section 6.2)
ROUND_FRICTION = (*BRICK_FRICTION[:2], ROUND_ROLLING_FRICTION)
NOSLIP_ITERATIONS = 3
IMPRATIO = 10.0
FLOOR_Z = -0.75  # the Isaac GroundPlaneCfg's height

# Light levels, calibrated 2026-09-11 against MS2 Isaac frames rendered at the same run draws
# (campaign runs 4, 6, 11 and 16: dome 1031-3059, sun 281-1985, 3141-7987 K). The first port
# (dome 0.7 diffuse + 0.35 ambient at intensity / 4000, sun 0.6 at 2000, full black-body tint)
# lit a held brick at the 1.4 rad release tilt at a third of Isaac's brightness, cut the desk
# under the sun's shadow to 0.22-0.40 of its lit level (Isaac about 0.84) and turned 3300 K into
# orange; the t2 policy put dark held bricks in the wrong bin and lost the target brick under
# the gripper's shadow (docs/ms3-mujoco-gate.md). With these the held brick reads at Isaac's
# saturation in every reference draw and the shadow keeps 0.52-0.76 of the lit level.
DOME_GAIN = 1.0 / 1000.0  # light level per unit of dome_intensity
DOME_CAP = 2.5  # bright draws clip, as Isaac's do, but the brick hues survive
SUN_GAIN = 0.3 / 2000.0  # a full sun adds 30 % on top of a 1000 dome: a mild shadow
TINT_BLEND = 0.3  # fraction of the black-body tint kept (Isaac's 3141 K read R/B 1.17)
SHADOW_MAP_SIZE = 4096  # the default 1024 over the 6 m floor is 3.5 mm per texel


@dataclass(frozen=True)
class SceneHandles:
    """Model indices the env needs every step, resolved once per model."""

    joint_ids: tuple[int, ...]
    joint_qpos: tuple[int, ...]
    joint_dof: tuple[int, ...]
    arm_actuators: tuple[int, ...]
    gripper_actuators: tuple[int, ...]
    base_body: int
    tool_body: int
    brick_bodies: tuple[int, ...]
    brick_qpos: tuple[int, ...]
    brick_dof: tuple[int, ...]
    brick_geoms: tuple[int, ...]
    body_geoms: tuple[tuple[int, ...], ...]  # every geom of each part body, collider first
    finger_geoms: tuple[frozenset[int], frozenset[int]]
    desk_geom: int
    cameras: tuple[str, ...]
    desk_box_mocap: tuple[int, ...] = ()  # mocap ids of the MS7a desk boxes, in box order


def kelvin_rgb(temperature_k: float) -> tuple[float, float, float]:
    """Approximate linear RGB (0-1) of a black body at ``temperature_k`` (Tanner Helland fit)."""
    t = max(1000.0, min(40000.0, temperature_k)) / 100.0
    red = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    green = (
        99.4708025861 * math.log(t) - 161.1195681661
        if t <= 66
        else 288.1221695283 * (t - 60) ** -0.0755148492
    )
    blue = (
        255.0
        if t >= 66
        else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    )
    return tuple(min(255.0, max(0.0, c)) / 255.0 for c in (red, green, blue))


def dome_tint(temperature_k: float) -> np.ndarray:
    """The dome's RGB tint: the black-body color pulled ``1 - TINT_BLEND`` of the way to white."""
    return 1.0 - TINT_BLEND * (1.0 - np.array(kelvin_rgb(temperature_k)))


def _half(size) -> list[float]:
    return [0.5 * s for s in size]


def _urdf_with_compiler() -> str:
    """The vendored URDF text plus the one ``<mujoco>`` block the importer reads options from."""
    text = VENDORED_URDF.read_text()
    compiler = (
        f'<mujoco><compiler meshdir="{VENDORED_URDF.parent / "meshes"}" '
        'balanceinertia="true" discardvisual="false" strippath="true"/></mujoco>'
    )
    return re.sub(r"(<robot[^>]*>)", lambda m: m.group(1) + "\n  " + compiler, text, count=1)


def _mount_quat() -> np.ndarray:
    """MOUNT_RPY as a (w, x, y, z) quaternion, from the same matrix the global IK uses."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, _rpy_matrix(*MOUNT_RPY).numpy().reshape(9))
    return quat


def _direction_from_rpy_deg(rpy_deg) -> np.ndarray:
    """World direction a light pointing along -z acquires under an rpy (degrees) rotation."""
    w, x, y, z = quat_from_rpy_deg(rpy_deg)
    quat = np.array([w, x, y, z])
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.array([0.0, 0.0, -1.0]), quat)
    return out


def _add_position_actuator(spec, joint: str, kp: float, kd: float, force_limit: float) -> None:
    act = spec.add_actuator(name=f"act_{joint}", target=joint)
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.gainprm = [kp] + [0.0] * 9
    act.biasprm = [0.0, -kp, -kd] + [0.0] * 7
    act.forcerange = [-force_limit, force_limit]
    act.ctrlrange = list(spec.joint(joint).range)


def _add_part_geoms(
    spec, body, index: int, part: _parts.Part, rgba, round_physics: bool = False
) -> mujoco.MjsGeom:
    """One collider (box, cylinder or slope hull) plus visual-only studs (design H3).

    Under ``round_physics`` a round part's studs collide too, and the part's geoms carry
    ``ROUND_CONDIM`` and ``ROUND_FRICTION`` (MS7e design section 6.4). Returns the collider.
    """
    name = f"brick_{index}_geom"
    rolls = round_physics and part.family == "round"
    if part.family == "round":
        geom = body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[part.size[0] / 2, part.size[2] / 2, 0.0],
            rgba=list(rgba),
            friction=list(ROUND_FRICTION if rolls else BRICK_FRICTION),
        )
    elif part.hull is None:
        geom = body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=_half(part.size),
            rgba=list(rgba),
            friction=list(BRICK_FRICTION),
        )
    else:
        spec.add_mesh(name=f"brick_{index}_hull", uservert=[c for v in part.hull for c in v])
        geom = body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"brick_{index}_hull",
            rgba=list(rgba),
            friction=list(BRICK_FRICTION),
        )
    geom.mass = part.mass
    geom.condim = ROUND_CONDIM if rolls else BRICK_CONDIM
    for k, (x, y) in enumerate(_parts.stud_positions(part)):
        stud = body.add_geom(
            name=f"brick_{index}_stud_{k}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[_parts.STUD_RADIUS, _parts.STUD_HEIGHT / 2, 0.0],
            pos=[x, y, part.size[2] / 2 + _parts.STUD_HEIGHT / 2],
            rgba=list(rgba),
        )
        if rolls:
            stud.contype, stud.conaffinity = 3, 3  # build_spec's scene pair
            stud.condim = ROUND_CONDIM
            stud.friction = list(ROUND_FRICTION)
            stud.solref = list(CONTACT_SOLREF)
        else:
            stud.contype, stud.conaffinity = 0, 0
        stud.mass = 0.0
    return geom


def build_spec(
    draw: dict,
    num_bricks: int = 6,
    cameras_on: bool = True,
    parts: Sequence[_parts.Part] | None = None,
    num_desk_boxes: int = 0,
    tray: bool = False,
    tray_lean_deg: float = 0.0,
    deep_box_floor: bool = False,
    camera_back: float = 0.0,
    round_physics: bool = False,
) -> mujoco.MjSpec:
    """The MjSpec for one run draw; ``compile_model`` turns it into a model and handles.

    ``parts`` (MS6) replaces the ``num_bricks`` identical box bricks with one body per part.
    A heap lies on the bare desk: there is no dish (owner decision 2026-09-18). ``tray``
    (MS7a design, the tray) adds the four walls of ``layout.tray_static_boxes``, or with
    ``tray_lean_deg`` > 0 the leaning walls of ``layout.tray_lean_boxes``. ``deep_box_floor``
    (MS7e flat track) extends each desk box floor's collider under the desk top by
    ``desk_boxes.DESK_BOX_FLOOR_DEPTH``, so a thin part cannot drop through it.

    ``num_desk_boxes`` (MS7a design, desk boxes) removes the shelf and its bins and adds that
    many open boxes on the desk, each a mocap body (``desk_box_<j>``) the env moves per
    episode. A static world geom cannot move: its collision bounds are compiled once, and a
    part dropped on a world geom moved by ``geom_pos`` fell through it (measured 2026-09-23).

    ``camera_back`` moves the overhead camera's nominal x back (``overhead_camera_pose``).

    ``round_physics`` (MS7e design section 6.4) gives round parts collision studs and
    ``ROUND_CONDIM``; every other part is built as without it.
    """
    if parts is not None:
        num_bricks = len(parts)
    bl = layout.BinLayout(
        bin_x=draw["bin_x"], bin_ys=tuple(draw["bin_ys"]), slot_colors=tuple(draw["slot_colors"])
    )
    spec = mujoco.MjSpec.from_string(_urdf_with_compiler())
    spec.modelname = "maker_arm_lego_sort"
    # Keep static bodies: base_link carries the Z-up mount and grasp_center is the IK target
    # the rig reads through `find_bodies`; the default fuses both away at compile time.
    spec.compiler.fusestatic = False
    spec.option.timestep = PHYSICS_DT
    # MjSpec defaults to Euler. Explicit damping is unstable once kv * dt / m > 2, and the jaw
    # actuator's kv 200 on a 60 g finger at this step is 14: measured 2026-09-10, the jaws hit
    # 1.3 m/s and oscillated over their whole travel in free air. implicitfast integrates joint
    # damping and the actuators' affine velocity term implicitly and removes that limit.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.noslip_iterations = NOSLIP_ITERATIONS
    spec.option.impratio = IMPRATIO
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

    # Z-up mount (urdf_prep writes the same rotation into the derived URDF Isaac loads).
    spec.body("base_link").quat = _mount_quat()
    # No self-collision, as the Isaac articulation (enabled_self_collisions=False). MuJoCo
    # collides two geoms when (contype_a & conaffinity_b) or (contype_b & conaffinity_a) is
    # non-zero, so robot geoms get contype 1 / conaffinity 2 and the scene geoms 3 / 3:
    # robot-scene and scene-scene pairs collide, robot-robot pairs never do. Without
    # this the two finger hulls collide with each other and the jaws stall 27 mm short of the
    # brick (measured on the first oracle run, 2026-09-10).
    for geom in spec.geoms:
        if geom.contype != 0 or geom.conaffinity != 0:
            geom.contype, geom.conaffinity = 1, 2
            geom.solref = list(CONTACT_SOLREF)
    for name in ARM_BODIES:
        spec.body(name).gravcomp = 1.0
    for name in ARM_JOINT_NAMES:
        spec.joint(name).armature = ARM_ARMATURE
    for name in JOINT_NAMES:
        spec.joint(name).solref_limit = list(LIMIT_SOLREF)

    # Static scene.
    scene_geoms = []
    scene_geoms.append(
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[3.0, 3.0, 0.1],
            pos=[0.0, 0.0, FLOOR_Z],
            rgba=[*draw["floor_color"], 1.0],
        )
    )
    desk = spec.worldbody.add_geom(
        name="desk",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=_half(layout.DESK_SIZE),
        pos=list(layout.DESK_POS),
        rgba=[0.5, 0.4, 0.3, 1.0],
    )
    desk.solmix = DESK_SOLMIX
    desk.solref = list(CONTACT_SOLREF)
    desk.solimp = list(DESK_SOLIMP)
    scene_geoms.append(desk)
    if num_desk_boxes == 0:
        scene_geoms.append(
            spec.worldbody.add_geom(
                name="shelf",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=_half(layout.SHELF_SIZE),
                pos=list(layout.SHELF_POS),
                rgba=[0.6, 0.6, 0.6, 1.0],
            )
        )
        bin_boxes = (
            layout.heap_bin_static_boxes(bl) if parts is not None else layout.bin_static_boxes(bl)
        )
    else:
        bin_boxes = []
    for j in range(num_desk_boxes):
        # Parked off the desk until the first reset draws the episode's poses.
        body = spec.worldbody.add_body(name=f"desk_box_{j}", mocap=True, pos=[-1.0, 0.2 * j, 0])
        depth = desk_boxes.DESK_BOX_FLOOR_DEPTH if deep_box_floor else 0.0
        for name, center, size in desk_boxes.box_geoms(desk_boxes.interior(num_desk_boxes), depth):
            scene_geoms.append(
                body.add_geom(
                    name=f"desk_box_{j}_{name}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=_half(size),
                    pos=list(center),
                    rgba=list(desk_boxes.DESK_BOX_RGBA),
                )
            )
    for name, center, size in bin_boxes:
        color = name.split("_")[1]
        rgba = (
            [*layout.BRICK_RGBA[color][:3], 1.0]
            if name.endswith("floor") and color in layout.BRICK_RGBA
            else [0.55] * 3 + [1.0]
        )
        scene_geoms.append(
            spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=_half(size),
                pos=list(center),
                rgba=rgba,
            )
        )
    walls = []
    if tray:
        walls = (
            layout.tray_lean_boxes(tray_lean_deg)
            if tray_lean_deg > 0.0
            else [(*box, (1.0, 0.0, 0.0, 0.0)) for box in layout.tray_static_boxes()]
        )
    for name, center, size, quat in walls:
        scene_geoms.append(
            spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=_half(size),
                pos=list(center),
                quat=list(quat),
                rgba=[0.35, 0.35, 0.35, 1.0],
            )
        )

    # Bricks: free bodies; the default pose is overwritten by the first reset's scatter.
    for index in range(num_bricks):
        color = layout.BRICK_COLOR_ORDER[index % len(layout.BRICK_COLOR_ORDER)]
        body = spec.worldbody.add_body(
            name=f"brick_{index}", pos=[layout.SCATTER_X[0] + 0.04 * index, 0.0, layout.SPAWN_Z]
        )
        body.add_freejoint(name=f"brick_{index}_free")
        if parts is None:
            geom = body.add_geom(
                name=f"brick_{index}_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=_half(layout.BRICK_SIZE),
                rgba=list(layout.BRICK_RGBA[color]),
                friction=list(BRICK_FRICTION),
            )
            geom.mass = layout.BRICK_MASS
            geom.condim = BRICK_CONDIM
        else:
            geom = _add_part_geoms(
                spec, body, index, parts[index], layout.BRICK_RGBA[color], round_physics
            )
        geom.solref = list(CONTACT_SOLREF)
        scene_geoms.append(geom)
    for geom in scene_geoms:
        geom.contype, geom.conaffinity = 3, 3  # collides with the robot AND with itself

    # Cameras (Isaac offsets are ROS-convention; see core.cameras).
    if cameras_on:
        over_pos, over_rot = overhead_camera_pose(draw, camera_back)
        spec.worldbody.add_camera(
            name="overhead",
            pos=over_pos,
            quat=list(quat_mul(over_rot, cameras.ROS_TO_MUJOCO_CAM)),
            fovy=cameras.OVERHEAD_FOV_DEG,
        )
        wrist_pos = [p + d for p, d in zip(WRIST_CAM_POS_IN_WRIST, draw["wrist_cam_dpos"])]
        spec.body(WRIST_LINK).add_camera(
            name="wrist",
            pos=wrist_pos,
            quat=list(quat_mul(WRIST_CAM_ROT_IN_WRIST, cameras.ROS_TO_MUJOCO_CAM)),
            fovy=cameras.WRIST_FOV_DEG,
        )

    # Lighting (module docstring): a hemisphere dome, half top-down diffuse and half ambient,
    # tinted by its color temperature; the optional sun casts the only shadow.
    spec.visual.headlight.active = 0
    spec.visual.quality.shadowsize = SHADOW_MAP_SIZE
    dome = dome_tint(draw["dome_color_temp"]) * min(DOME_CAP, DOME_GAIN * draw["dome_intensity"])
    dome_light = spec.worldbody.add_light(name="dome", pos=[0.3, 0.0, 2.0], dir=[0.0, 0.0, -1.0])
    dome_light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    dome_light.castshadow = False
    dome_light.diffuse = (0.5 * dome).tolist()
    dome_light.ambient = (0.5 * dome).tolist()
    dome_light.specular = [0.05, 0.05, 0.05]
    # The sun is always present so a per-episode draw can turn it on or off in the compiled
    # model (apply_lighting); at zero strength it adds nothing and casts nothing.
    sun = spec.worldbody.add_light(
        name="sun",
        pos=[0.3, 0.0, 2.0],
        dir=_direction_from_rpy_deg(draw["sun_rpy_deg"]).tolist(),
    )
    sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    sun.castshadow = True
    strength = SUN_GAIN * draw["sun_intensity"]
    sun.diffuse = [strength] * 3
    sun.specular = [0.1 * strength] * 3

    # Actuators: arm from the Rust profile, gripper from robots.frames.
    gains = _profile.arm_gains(_profile.load_profile())
    for joint in ARM_JOINT_NAMES:
        g = gains[joint]
        _add_position_actuator(spec, joint, g["kp"], g["kd"], g["effort_limit"])
    for joint in GRIPPER_JOINT_NAMES:
        _add_position_actuator(
            spec, joint, GRIPPER_STIFFNESS, GRIPPER_DAMPING, GRIPPER_EFFORT_LIMIT_N
        )
    return spec


def compile_model(spec: mujoco.MjSpec, num_bricks: int) -> tuple[mujoco.MjModel, SceneHandles]:
    model = spec.compile()
    joint_ids = tuple(model.joint(n).id for n in JOINT_NAMES)
    finger_geoms = []
    for finger in ("gripper_left", "gripper_right"):
        body = model.body(finger).id
        finger_geoms.append(
            frozenset(
                g
                for g in range(model.ngeom)
                if model.geom_bodyid[g] == body and model.geom_contype[g] != 0
            )
        )
    handles = SceneHandles(
        joint_ids=joint_ids,
        joint_qpos=tuple(int(model.jnt_qposadr[j]) for j in joint_ids),
        joint_dof=tuple(int(model.jnt_dofadr[j]) for j in joint_ids),
        arm_actuators=tuple(model.actuator(f"act_{n}").id for n in ARM_JOINT_NAMES),
        gripper_actuators=tuple(model.actuator(f"act_{n}").id for n in GRIPPER_JOINT_NAMES),
        base_body=model.body("base_link").id,
        tool_body=model.body(TOOL_BODY).id,
        brick_bodies=tuple(model.body(f"brick_{b}").id for b in range(num_bricks)),
        brick_qpos=tuple(
            int(model.jnt_qposadr[model.joint(f"brick_{b}_free").id]) for b in range(num_bricks)
        ),
        brick_dof=tuple(
            int(model.jnt_dofadr[model.joint(f"brick_{b}_free").id]) for b in range(num_bricks)
        ),
        brick_geoms=tuple(model.geom(f"brick_{b}_geom").id for b in range(num_bricks)),
        body_geoms=tuple(
            (model.geom(f"brick_{b}_geom").id,)
            + tuple(
                g
                for g in range(model.ngeom)
                if model.geom_bodyid[g] == model.body(f"brick_{b}").id
                and g != model.geom(f"brick_{b}_geom").id
            )
            for b in range(num_bricks)
        ),
        finger_geoms=(finger_geoms[0], finger_geoms[1]),
        desk_geom=model.geom("desk").id,
        cameras=tuple(model.camera(c).name for c in range(model.ncam)),
        desk_box_mocap=tuple(
            int(model.body_mocapid[model.body(f"desk_box_{j}").id])
            for j in range(model.nmocap)  # the desk boxes are the only mocap bodies
        ),
    )
    return model, handles


def build_model(
    draw: dict,
    num_bricks: int = 6,
    cameras_on: bool = True,
    parts: Sequence[_parts.Part] | None = None,
    num_desk_boxes: int = 0,
    tray: bool = False,
    tray_lean_deg: float = 0.0,
    deep_box_floor: bool = False,
    camera_back: float = 0.0,
    round_physics: bool = False,
) -> tuple[mujoco.MjModel, SceneHandles]:
    """One call for callers that do not need to edit the spec."""
    if parts is not None:
        num_bricks = len(parts)
    spec = build_spec(
        draw,
        num_bricks,
        cameras_on,
        parts,
        num_desk_boxes,
        tray,
        tray_lean_deg,
        deep_box_floor=deep_box_floor,
        camera_back=camera_back,
        round_physics=round_physics,
    )
    return compile_model(spec, num_bricks)


# -- per-episode draws: the same numbers as build_spec, written into a compiled model ----------


def overhead_camera_pose(
    draw: dict, back: float = 0.0
) -> tuple[list[float], tuple[float, float, float, float]]:
    """The overhead camera's world position and ROS-convention quaternion for ``draw``, with
    its nominal x ``back`` m further back (``cameras.OVERHEAD_B2_BACK_M`` under MS7e B2)."""
    pos = [p + d for p, d in zip(cameras.OVERHEAD_CAM_POS, draw["overhead_cam_dpos"])]
    if back:
        pos[0] -= back
    rot = compose_offset_rot(cameras.OVERHEAD_CAM_ROT, draw["overhead_cam_drot_deg"])
    return pos, rot


def apply_bins(model: mujoco.MjModel, bl: layout.BinLayout, heap: bool = False) -> None:
    """Move the static bin geoms to ``bl``. Sizes and colors never change per episode."""
    boxes = layout.heap_bin_static_boxes(bl) if heap else layout.bin_static_boxes(bl)
    for name, center, _size in boxes:
        model.geom_pos[model.geom(name).id] = center


def apply_desk_box_size(model: mujoco.MjModel, j: int, size, floor_depth: float = 0.0) -> None:
    """Resize desk box ``j`` to interior ``size`` (the MS7e box draw, design section 4.2), its
    floor reaching ``floor_depth`` under the desk top as the scene built it (``deep_box_floor``).

    Writes each geom's centre and size from ``desk_boxes.box_geoms``, then every bound the
    collision pipeline caches from them, which ``mj_setConst`` does not refresh (read off
    MuJoCo 3.13.0, 2026-09-29): the geom's bounding sphere (``geom_rbound``), its local box
    (``geom_aabb``) and the body's bounding volume tree (``bvh_aabb``), which lives in the
    body's inertial frame. The tree keeps the compiled partition: a leaf is its geom's box and
    a node the union of its two children, and a child always follows its parent."""
    body = model.body(f"desk_box_{j}").id
    for name, center, full in desk_boxes.box_geoms(size, floor_depth):
        g = model.geom(f"desk_box_{j}_{name}").id
        half = np.asarray(full, dtype=float) / 2.0
        model.geom_pos[g] = center
        model.geom_size[g] = half
        model.geom_rbound[g] = np.linalg.norm(half)
        model.geom_aabb[g] = (0.0, 0.0, 0.0, *half)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.body_iquat[body])
    rot = rot.reshape(3, 3)  # inertial axes in the body frame, as columns
    start, count = model.body_bvhadr[body], model.body_bvhnum[body]
    for node in reversed(range(start, start + count)):
        a, b = model.bvh_child[node]
        if a < 0:
            g = model.bvh_nodeid[node]
            centre = rot.T @ (model.geom_pos[g] - model.body_ipos[body])
            model.bvh_aabb[node] = (*centre, *(np.abs(rot.T) @ model.geom_size[g]))
            continue
        kids = [model.bvh_aabb[start + c] for c in (a, b)]
        lo = np.minimum(*(k[:3] - k[3:] for k in kids))
        hi = np.maximum(*(k[:3] + k[3:] for k in kids))
        model.bvh_aabb[node] = (*(lo + hi) / 2.0, *(hi - lo) / 2.0)


def apply_lighting(model: mujoco.MjModel, draw: dict) -> None:
    """Write ``draw``'s dome, sun and floor color into ``model`` (build_spec's formulas)."""
    dome = dome_tint(draw["dome_color_temp"]) * min(DOME_CAP, DOME_GAIN * draw["dome_intensity"])
    d = model.light("dome").id
    model.light_diffuse[d] = 0.5 * dome
    model.light_ambient[d] = 0.5 * dome
    s = model.light("sun").id
    strength = SUN_GAIN * draw["sun_intensity"]
    model.light_diffuse[s] = strength
    model.light_specular[s] = 0.1 * strength
    model.light_dir[s] = _direction_from_rpy_deg(draw["sun_rpy_deg"])
    model.geom_rgba[model.geom("floor").id, :3] = draw["floor_color"]


def apply_cameras(model: mujoco.MjModel, draw: dict, camera_back: float = 0.0) -> None:
    """Move both cameras to ``draw``'s jitter. A model built without cameras is left alone."""
    if model.ncam == 0:
        return
    pos, rot = overhead_camera_pose(draw, camera_back)
    c = model.camera("overhead").id
    model.cam_pos[c] = pos
    model.cam_quat[c] = quat_mul(rot, cameras.ROS_TO_MUJOCO_CAM)
    w = model.camera("wrist").id
    model.cam_pos[w] = [p + d for p, d in zip(WRIST_CAM_POS_IN_WRIST, draw["wrist_cam_dpos"])]
