"""The MuJoCo LegoSort environment behind the duck-typed surface ``OracleRig`` drives.

Every member below exists because something reads it: ``OracleRig`` (the rig), the batch
loop in ``scripts/generate_batch.py``, ``core.stowed`` (grading), and the rollout loop in
``scripts/eval_policy.py``. The names and shapes are Isaac Lab's, so those consumers run
unchanged on either backend:

    env.device, env.num_envs, env.cfg.num_bricks, env.cfg.bin_layout, env.max_episode_length
    env.scene["robot"].data.{joint_pos, joint_vel, joint_pos_limits, body_pos_w, body_quat_w,
                             computed_torque, applied_torque}, .find_bodies, .find_joints
    env.scene["brick_N"].data.{root_pos_w, root_quat_w, root_lin_vel_w}
    env.scene["finger_left_brick"].data.force_matrix_w   (N,1,B,3), likewise right
    env.scene.env_origins (N,3), zeros: every env is its own world
    env.action_manager.total_action_dim == 7, .get_term("arm_action")._joint_names / .clamped
    env.termination_manager.active_terms, .get_term("success")
    env.obs_buf["images"]["wrist_rgb" | "overhead_rgb"]   (N,224,224,3) uint8
    env.recorder_manager.exported_successful_episode_count
    env.step(action) -> (None, None, terminated, truncated, {}), env.reset(seed=, env_ids=)

MuJoCo-only members (the Isaac env has none of these; core/ reads them through getattr):
    env.bin_layouts (one core.layout.BinLayout per env), env.episode_draws, env.run_draw
    env.context_mode, env.obs_buf["context"], env.context_camera, env.context_privileged
    env.oracle_actor (N,) int16, set by the DAgger loop: 0 records nothing, k records the
        frame as takeover k
    env.gripper_mode ("binary" | "continuous") and env.jaw_targets(raw)
    heap mode (MS6, ``parts`` given): env.parts, env.destination_bins (N,) long,
        env.stow_half_extents, env.target_part (N,) int16 (the overhead marker; from the rig's
        env.oracle_marker_part, which names the next part through RETREAT), env.context_box,
        env.context_camera_box (N,9), env.context_target (N,7), env.context_camera_box_target
        (N,16), env.obs_buf["images"]["overhead_depth"] (N,224,224) uint16,
        env.stamp_markers() (the heap loop stamps the frame it hands the policy),
        env.marker_from_rig (default True; the heap loop's policy mode sets False so a
        takeover keeps the marker on the loop's target), env.tray (MS7a: the heap drops
        inside the four walls of layout.tray_static_boxes)
    desk boxes (MS7a, ``desk_boxes`` > 0, heap mode only): env.desk_boxes, env.desk_box_poses
        (N, M, 3) float64 per-episode (x, y, yaw) of each box, env.class_to_box (N, 4) int64
        (the box of each core.parts.PART_CLASSES class, drawn per episode),
        env.destination_bins (N, K) long (part b of env i goes to box destination_bins[i, b],
        its class's box), env.place_desk_boxes(i, poses),
        env.set_context_box(i, box) (the context row's box part, set per target)

Semantics carried over from ``ManagerBasedRLEnv`` because the rig depends on them: a done
env is auto-reset INSIDE ``step`` (its state tensors already show the next episode when
``step`` returns, and the ``success`` term keeps the terminating step's value until the next
``step``); the six arm targets pass through ``core.profile``'s Rust clamp and the per-env
``clamped`` flag says whether that changed the target this step.

Batching is a Python loop over one ``MjModel``/``MjData`` pair per env. Per-env models are
what make per-episode physics randomization (brick mass and friction, desk color) possible;
the cost is one ``copy`` of a compiled model per env at construction.
"""

from __future__ import annotations

import copy
import math
import random
import re
import types
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

from maker_arm_lab.assets.urdf_prep import gripper_velocity_limit
from maker_arm_lab.core import cameras, layout
from maker_arm_lab.core import desk_boxes as _boxes
from maker_arm_lab.core import profile as _profile
from maker_arm_lab.core.checker import HELD_MIN_FORCE, held_brick_from_forces
from maker_arm_lab.core.heap_oracle import STOW_HALF_EXTENTS
from maker_arm_lab.core.parts import PART_CLASSES, Part
from maker_arm_lab.core.quat import compose_offset_rot, quat_apply, quat_mul, rpy_from_quat
from maker_arm_lab.core.stowed import per_brick_stowed
from maker_arm_lab.core.target import target_brick
from maker_arm_lab.mj import overlay, scene
from maker_arm_lab.mj.context_names import (
    CONTEXT_CAMERA_NAMES,  # noqa: F401 -- re-exported as mj_env.CONTEXT_CAMERA_NAMES
    CONTEXT_DIMS,
    CONTEXT_MAX_BRICKS,
    CONTEXT_PRIVILEGED_NAMES,  # noqa: F401 -- re-exported as mj_env.CONTEXT_PRIVILEGED_NAMES
)
from maker_arm_lab.mj.depth import noisy_depth_mm
from maker_arm_lab.mj.recorder import ShardRecorder
from maker_arm_lab.robots.frames import (
    ARM_JOINT_NAMES,
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
    HOME_JOINT_POS,
    PICK_OFFSET_MEASURED,
)

CONTROL_HZ = 30
# The binary gripper command is turned into a jaw target that moves at the URDF's jaw velocity
# limit (0.1 m/s) instead of jumping. Isaac's implicit PhysX drive absorbed the jump; a MuJoCo
# position actuator with kp 1000 N/m and a 20 N force cap saturates for the whole 52 mm of
# travel, and a saturated actuator has no implicit damping, so the jaw slammed into its stop
# and bounced (measured 2026-09-10: -8.8 mm, then +62.7 mm). Ramping the target keeps the
# demand under the cap in transit and leaves the stall force on a brick at kp x 7.9 mm, the
# same 7.6-7.9 N Isaac measured.
GRIPPER_RATE_M_PER_S = gripper_velocity_limit()
# The Isaac reset-level DR ranges (datagen.LegoSortDatagenEnvCfg.__post_init__), verbatim.
DESK_COLOR_RANGE = {"r": (0.3, 0.7), "g": (0.25, 0.55), "b": (0.2, 0.45)}
BRICK_MASS_SCALE = (0.7, 1.3)
BRICK_FRICTION_SCALE = (0.8, 1.2)  # Isaac's static_friction_range; MuJoCo has one coefficient
HEAP_PARK_X = 3.0
"""Where a heap part waits (m along x, well past the 0.9 m desk) until its drop batch."""
# Per-frame camera jitter (MS4 lever 1): resampled on every rendered frame, on top of the
# per-episode camera draw, so the policy cannot read the camera pose off a static background.
# The context feature reports the per-episode pose, not this jitter, which stands in for the
# calibration error of the real rig.
FRAME_CAM_JITTER_M = 0.005
FRAME_CAM_JITTER_DEG = 1.0
# The point on the tool axis a desk-resting brick occupies, as OracleRig.pick_point builds it.
# The privileged channel's target rule measures from THIS point, not from the tool body, so it
# is the same distance `EnvOracle._next_brick` takes its nearest candidate by.
_PICK_OFFSET = torch.tensor(PICK_OFFSET_MEASURED)


def _rot6(q: np.ndarray) -> np.ndarray:
    """First two columns of the rotation matrix of a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ],
        dtype=np.float32,
    )


def _random_quat(rng: random.Random) -> tuple[float, float, float, float]:
    """A uniformly random (w, x, y, z) rotation (Shoemake's subgroup algorithm)."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    a, b = math.sqrt(1.0 - u1), math.sqrt(u1)
    return (
        a * math.sin(2.0 * math.pi * u2),
        a * math.cos(2.0 * math.pi * u2),
        b * math.sin(2.0 * math.pi * u3),
        b * math.cos(2.0 * math.pi * u3),
    )


class _Robot:
    """``scene["robot"]``: joint and body state views plus the two name lookups the rig uses."""

    def __init__(self, env: MjLegoSortEnv):
        self._env = env
        self.joint_names = list(scene.JOINT_NAMES)
        n, nbody = env.num_envs, env.models[0].nbody
        # float32, as Isaac Lab's ArticulationData: the rig mixes these with float32 constants.
        self._joint_pos = np.zeros((n, 8), dtype=np.float32)
        self._joint_vel = np.zeros((n, 8), dtype=np.float32)
        self._body_pos = np.zeros((n, nbody, 3), dtype=np.float32)
        self._body_quat = np.zeros((n, nbody, 4), dtype=np.float32)
        self._computed = np.zeros((n, 8), dtype=np.float32)
        self._applied = np.zeros((n, 8), dtype=np.float32)
        limits = np.array(
            [env.models[0].jnt_range[j] for j in env.handles.joint_ids], dtype=np.float32
        )
        self.data = types.SimpleNamespace(
            joint_pos=torch.from_numpy(self._joint_pos),
            joint_vel=torch.from_numpy(self._joint_vel),
            joint_pos_limits=torch.from_numpy(np.broadcast_to(limits, (n, 8, 2)).copy()),
            body_pos_w=torch.from_numpy(self._body_pos),
            body_quat_w=torch.from_numpy(self._body_quat),
            computed_torque=torch.from_numpy(self._computed),
            applied_torque=torch.from_numpy(self._applied),
        )

    def find_bodies(self, name: str):
        return ([self._env.models[0].body(name).id], [name])

    def find_joints(self, names, preserve_order: bool = False):
        """Isaac's ``find_joints``: each entry is a name or a regex; indices into the 8-vector."""
        found: list[tuple[int, str]] = []
        for pattern in names:
            for index, joint in enumerate(self.joint_names):
                if joint == pattern or re.fullmatch(pattern, joint):
                    found.append((index, joint))
        if not preserve_order:
            found.sort()
        return ([i for i, _ in found], [n for _, n in found])

    def refresh(self, i: int) -> None:
        env, h = self._env, self._env.handles
        m, d = env.models[i], env.datas[i]
        self._joint_pos[i] = d.qpos[list(h.joint_qpos)]
        self._joint_vel[i] = d.qvel[list(h.joint_dof)]
        self._body_pos[i] = d.xpos
        self._body_quat[i] = d.xquat
        acts = list(h.arm_actuators) + list(h.gripper_actuators)
        kp = m.actuator_gainprm[acts, 0]
        kv = -m.actuator_biasprm[acts, 2]
        self._computed[i] = kp * (d.ctrl[acts] - self._joint_pos[i]) - kv * self._joint_vel[i]
        self._applied[i] = d.actuator_force[acts]


class _Brick:
    def __init__(self, num_envs: int):
        self._pos = np.zeros((num_envs, 3), dtype=np.float32)
        self._quat = np.zeros((num_envs, 4), dtype=np.float32)
        self._vel = np.zeros((num_envs, 3), dtype=np.float32)
        self.data = types.SimpleNamespace(
            root_pos_w=torch.from_numpy(self._pos),
            root_quat_w=torch.from_numpy(self._quat),
            root_lin_vel_w=torch.from_numpy(self._vel),
        )


class _Contact:
    def __init__(self, num_envs: int, num_bricks: int):
        self._force = np.zeros((num_envs, 1, num_bricks, 3), dtype=np.float32)
        self.data = types.SimpleNamespace(force_matrix_w=torch.from_numpy(self._force))


class _Scene:
    def __init__(self, env: MjLegoSortEnv):
        self.env_origins = torch.zeros(env.num_envs, 3)
        self._items: dict[str, object] = {"robot": _Robot(env)}
        for b in range(env.cfg.num_bricks):
            self._items[f"brick_{b}"] = _Brick(env.num_envs)
        for side in ("left", "right"):
            self._items[f"finger_{side}_brick"] = _Contact(env.num_envs, env.cfg.num_bricks)

    def __getitem__(self, key: str):
        return self._items[key]

    def keys(self):
        return self._items.keys()


class MjLegoSortEnv:
    def __init__(
        self,
        num_envs: int,
        draw: dict,
        *,
        num_bricks: int = 6,
        parts: Sequence[Part] | None = None,
        cameras: bool = True,
        start_pose_offset_rad: float = 0.05,
        desk_color: bool = True,
        physics_dr: bool = True,
        episode_draws: bool = False,
        frame_jitter: bool = False,
        spatial_scale: float = 1.0,
        context: str = "none",
        episode_length_s: float = 180.0,
        export_dir: Path | str | None = None,
        shard_name: str = "shard",
        export_all: bool = False,
        gripper_mode: str = "binary",
        desk_boxes: int = 0,
        tray: bool = False,
    ):
        # MS6 phase B design 5.4: "binary" is the raw >= 0 opens contract every corpus before
        # phase B was recorded with; "continuous" reads action[:, 6] as the jaw gap in meters.
        if gripper_mode not in ("binary", "continuous"):
            raise ValueError(f"gripper_mode must be 'binary' or 'continuous', got {gripper_mode!r}")
        self.gripper_mode = gripper_mode
        self.device = "cpu"
        self.num_envs = num_envs
        self.cameras = cameras
        bl = layout.BinLayout(
            bin_x=draw["bin_x"],
            bin_ys=tuple(draw["bin_ys"]),
            slot_colors=tuple(draw["slot_colors"]),
        )
        self.cfg = types.SimpleNamespace(
            num_bricks=num_bricks,
            bin_layout=bl,
            start_pose_offset_rad=start_pose_offset_rad,
            desk_color=desk_color,
            physics_dr=physics_dr,
            episode_draws=episode_draws,
            frame_jitter=frame_jitter,
            spatial_scale=spatial_scale,
            episode_length_s=episode_length_s,
            cameras=cameras,
        )
        if context not in CONTEXT_DIMS:
            raise ValueError(f"context must be one of {sorted(CONTEXT_DIMS)}, got {context!r}")
        self.parts = tuple(parts) if parts is not None else None
        if tray and self.parts is None:
            raise ValueError("the tray holds a heap; the six-brick task has none")
        self.tray = tray
        if self.parts is not None:
            # MS6 rework section 6: heap mode has one bin, at y = 0.0. Flatten the layout
            # this env starts with so self.bin_layouts, the built box, the release target and
            # the grader's bin centre all agree from the first reset, not only after an
            # episode_draws redraw.
            bl = replace(bl, bin_ys=(0.0, 0.0, 0.0, 0.0))
            num_bricks = len(self.parts)
            self.cfg.num_bricks = num_bricks
            if context == "privileged":
                raise ValueError(
                    "the privileged channel holds six bricks; a heap serves camera, camera_box "
                    "or camera_box_target"
                )
        elif context == "camera_box_target":
            raise ValueError("the target channel is a heap channel; six bricks serve camera")
        elif num_bricks > CONTEXT_MAX_BRICKS:
            raise ValueError(f"context channels hold {CONTEXT_MAX_BRICKS} bricks, got {num_bricks}")
        if desk_boxes < 0 or (desk_boxes > 0 and self.parts is None):
            raise ValueError(f"desk_boxes needs a heap and a count >= 0, got {desk_boxes}")
        # MS7a design, desk boxes: N boxes on the desk replace the shelf and its heap bin.
        self.desk_boxes = desk_boxes
        self.desk_box_poses = np.zeros((num_envs, desk_boxes, 3))
        # The episode's box per class, in core.parts.PART_CLASSES order (MS7a design, classes).
        self.class_to_box = np.zeros((num_envs, len(PART_CLASSES)), dtype=np.int64)
        self.context_mode = context
        self.context_camera = np.zeros((num_envs, CONTEXT_DIMS["camera"]), dtype=np.float32)
        self.context_privileged = np.zeros((num_envs, CONTEXT_DIMS["privileged"]), dtype=np.float32)
        self.context_box = np.zeros((num_envs, 3), dtype=np.float32)
        self.context_camera_box = np.zeros((num_envs, CONTEXT_DIMS["camera_box"]), dtype=np.float32)
        # One buffer: the 9-wide camera+box row, then the target row as a view into it, so a
        # write to context_target lands in the served vector.
        self.context_camera_box_target = np.zeros(
            (num_envs, CONTEXT_DIMS["camera_box_target"]), dtype=np.float32
        )
        self.context_target = self.context_camera_box_target[:, CONTEXT_DIMS["camera_box"] :]
        served = {
            "none": np.zeros((num_envs, 0), dtype=np.float32),
            "camera": self.context_camera,
            "privileged": self.context_privileged,
            "camera_box": self.context_camera_box,
            "camera_box_target": self.context_camera_box_target,
        }[context]
        self.max_episode_length = round(episode_length_s * CONTROL_HZ)
        # The run-level draw, and the draw in force per env: equal to run_draw for every env
        # unless episode_draws redraws bins, lights and cameras at each reset.
        self.run_draw = dict(draw)
        self.episode_draws = [self.run_draw] * num_envs
        self.bin_layouts: list[layout.BinLayout] = [bl] * num_envs
        base_model, self.handles = scene.build_model(
            draw, num_bricks, cameras, parts=self.parts, num_desk_boxes=desk_boxes, tray=tray
        )
        self.models = [base_model] + [copy.copy(base_model) for _ in range(num_envs - 1)]
        self.datas = [mujoco.MjData(m) for m in self.models]
        self._nominal_inertia = base_model.body_inertia[list(self.handles.brick_bodies)].copy()
        self._nominal_mass = base_model.body_mass[list(self.handles.brick_bodies)].copy()
        self.renderers = [mujoco.Renderer(m, 224, 224) for m in self.models] if cameras else []
        self._cam_wrist = base_model.camera("wrist").id if cameras else -1
        self._cam_overhead = base_model.camera("overhead").id if cameras else -1
        self.profile = _profile.sim_profile()
        self.scene = _Scene(self)
        self._arm_term = types.SimpleNamespace(
            _joint_names=list(ARM_JOINT_NAMES),
            clamped=torch.zeros(num_envs, dtype=torch.bool),
        )
        self.action_manager = types.SimpleNamespace(
            total_action_dim=7, get_term=lambda name: self._arm_term
        )
        self._success = torch.zeros(num_envs, dtype=torch.bool)
        self.termination_manager = types.SimpleNamespace(
            active_terms=["time_out", "success"], get_term=lambda name: self._success
        )
        self._wrist = np.zeros((num_envs, 224, 224, 3), dtype=np.uint8)
        self._overhead = np.zeros((num_envs, 224, 224, 3), dtype=np.uint8)
        self.obs_buf = {
            "images": {
                "wrist_rgb": torch.from_numpy(self._wrist),
                "overhead_rgb": torch.from_numpy(self._overhead),
            },
            "context": torch.from_numpy(served),
        }
        self._np_rng = np.random.default_rng(0)
        if self.parts is not None:
            if desk_boxes:
                self.destination_bins = torch.zeros(num_envs, num_bricks, dtype=torch.long)
                self.stow_half_extents = torch.tensor(_boxes.stow_region(desk_boxes)[1])
            else:
                self.destination_bins = torch.zeros(num_envs, dtype=torch.long)
                self.stow_half_extents = torch.tensor(STOW_HALF_EXTENTS)
            self.target_part = torch.full((num_envs,), -1, dtype=torch.int16)
            if cameras:
                self._depth = np.zeros((num_envs, 224, 224), dtype=np.uint16)
                self.obs_buf["images"]["overhead_depth"] = torch.from_numpy(self._depth)
        self.episode_length_buf = np.zeros(num_envs, dtype=np.int64)
        self._grip_ctrl = np.full((num_envs, 2), GRIPPER_OPEN_POS)
        self.recorder = (
            ShardRecorder(
                Path(export_dir) / f"{shard_name}.hdf5",
                num_envs,
                export_all=export_all,
                cameras=cameras,
                depth=self.parts is not None,
                gripper_mode=gripper_mode,
            )
            if export_dir is not None
            else None
        )
        self.recorder_manager = (
            self.recorder
            if self.recorder is not None
            else types.SimpleNamespace(exported_successful_episode_count=0)
        )
        self._rng = random.Random(0)

    # -- gym-ish surface ---------------------------------------------------------

    def reset(self, seed: int | None = None, env_ids: torch.Tensor | None = None):
        if seed is not None:
            # Isaac Lab's configure_seed seeds these three; the scatter sampler in core.layout
            # draws from the global `random`, so the seed reaches it the same way.
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed % 2**32)
        ids = range(self.num_envs) if env_ids is None else [int(i) for i in env_ids.tolist()]
        for i in ids:
            self._reset_env(i)
        self._refresh(ids, images=True)
        return None, {}

    def step(self, action: torch.Tensor):
        raw = torch.as_tensor(action).detach().to("cpu", torch.float32)
        if raw.shape != (self.num_envs, 7):
            raise ValueError(
                f"expected an action of shape ({self.num_envs}, 7), got {tuple(raw.shape)}"
            )
        grip_target, grip_fraction = self.jaw_targets(raw[:, 6])
        arm = _profile.clamp_targets_tensor(
            raw[:, :6].double(), grip_fraction, self.profile, self._arm_term.clamped
        )
        targets = np.concatenate(
            [arm.numpy(), np.stack([grip_target, grip_target], axis=1)], axis=1
        )
        h = self.handles
        robot = self.scene["robot"]
        actor = getattr(self, "oracle_actor", None)
        recorded = [False] * self.num_envs
        heap = self.parts is not None
        if heap:
            if getattr(self, "marker_from_rig", True):
                self._update_target_parts(actor)
            self.stamp_markers()
        if self.recorder is not None:
            phase = getattr(self, "oracle_cycle_phase", None)
            brick = getattr(self, "oracle_cycle_brick", None)
            for i in range(self.num_envs):
                who = 1 if actor is None else int(actor[i])
                if who == 0:
                    continue  # a policy-driven DAgger frame: not recorded (design 6.4)
                recorded[i] = True
                self.recorder.pre_step(
                    i,
                    robot._joint_pos[i],
                    raw[i].numpy(),
                    int(phase[i]) if phase is not None else -1,
                    int(brick[i]) if brick is not None else -1,
                    self._wrist[i] if self.cameras else None,
                    self._overhead[i] if self.cameras else None,
                    context_camera=self.context_camera[i],
                    context_privileged=None if heap else self.context_privileged[i],
                    actor=who,
                    depth=self._depth[i] if heap and self.cameras else None,
                    target_part=int(self.target_part[i]) if heap else -1,
                    context_box=self.context_box[i] if heap else None,
                    context_target=self.context_target[i] if heap else None,
                )
        step_limit = GRIPPER_RATE_M_PER_S / CONTROL_HZ
        self._grip_ctrl += np.clip(targets[:, 6:] - self._grip_ctrl, -step_limit, step_limit)
        for i in range(self.num_envs):
            d = self.datas[i]
            d.ctrl[list(h.arm_actuators)] = targets[i, :6]
            d.ctrl[list(h.gripper_actuators)] = self._grip_ctrl[i]
            mujoco.mj_step(self.models[i], d, nstep=scene.CONTROL_SUBSTEPS)
        self.episode_length_buf += 1
        self._refresh(range(self.num_envs), images=False)
        success = per_brick_stowed(self).all(dim=-1)
        truncated = torch.from_numpy(self.episode_length_buf >= self.max_episode_length)
        self._success = success.clone()
        terminated = success.clone()
        if self.recorder is not None:
            clamped = self._arm_term.clamped.numpy()
            for i in range(self.num_envs):
                if recorded[i]:
                    self.recorder.post_step(i, targets[i], bool(clamped[i]))
        done = (terminated | truncated).nonzero().flatten().tolist()
        for i in done:
            if self.recorder is not None:
                self.recorder.end(i, bool(success[i]))
            self._reset_env(i)
        self._refresh(range(self.num_envs), images=True)
        return None, None, terminated, truncated, {}

    def jaw_targets(self, raw_gripper: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        """The per-jaw prismatic target (N,) and the clamp's opening fraction (N,) for the
        raw gripper column, in this env's gripper mode."""
        g = raw_gripper.detach().to("cpu", torch.float64)
        if self.gripper_mode == "continuous":
            per_jaw = (0.5 * g).clamp(GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS)
        else:
            # torch.where promotes bare Python floats to float32 before any later .to() call
            # runs, which rounds GRIPPER_OPEN_POS; keep both branches float64 from the start.
            per_jaw = torch.where(
                g >= 0.0,
                torch.full_like(g, GRIPPER_OPEN_POS),
                torch.full_like(g, GRIPPER_CLOSED_POS),
            )
        return per_jaw.numpy(), per_jaw / GRIPPER_OPEN_POS

    def _update_target_parts(self, actor) -> None:
        """Heap mode: the rig's cycle stash names the marker's part on the envs it drives.

        The marker follows ``oracle_marker_part``, which names the NEXT part through RETREAT
        (the rig's ``_stash_cycle_state``); ``oracle_cycle_brick`` stays the accounting id the
        recorder and the converter segment cycles by, and is the fallback for a stash that
        has no marker id (a test stub).
        """
        brick = getattr(self, "oracle_cycle_brick", None)
        if brick is None:
            return
        marker = getattr(self, "oracle_marker_part", brick)
        target = getattr(self, "oracle_cycle_target", None)
        for i in range(self.num_envs):
            if actor is None or int(actor[i]) > 0:
                self.target_part[i] = int(marker[i])
                if self.desk_boxes and int(marker[i]) >= 0:
                    self.set_context_box(i, int(self.destination_bins[i, int(marker[i])]))
                if target is None:
                    continue
                row = target[i].numpy()
                # A takeover that resumed at LIFT on a held part names the part but ranks no
                # grasp until REOPEN's regrasp, so the rig's row is zeros; the corpus never
                # shows a zero row on a marked part. Keep the loop's row, the one the policy
                # pinched under (campaign n round 2, 2026-09-22). The teacher's own batches
                # (actor None) and an unmarked stash (-1) copy the rig's row as before.
                marked_no_grasp = int(marker[i]) >= 0 and not row.any()
                if actor is not None and marked_no_grasp and self.context_target[i].any():
                    continue
                self.context_target[i] = row

    def set_context_box(self, i: int, box: int) -> None:
        """Serve box ``box``'s pose as env ``i``'s context box part (``CONTEXT_BOX_NAMES``).

        Writes ``context_box`` and its copies in the camera_box and camera_box_target rows, so
        a recorded frame and a served one agree. With desk boxes (MS7a design, desk boxes) the
        row is the box's (x, y, yaw) and follows the current target's box: the rig's marker in
        ``step`` and the heap loop's selection both call this. Without them it is the one heap
        bin, ``(bin_x, bin_y, 0)``, for the whole episode, as in MS6.
        """
        if self.desk_boxes:
            row = self.desk_box_poses[i, box]
        else:
            bl = self.bin_layouts[i]
            row = (bl.bin_x, bl.bin_ys[box], 0.0)
        self.context_box[i] = row
        self.context_camera_box[i, 6:] = self.context_box[i]
        self.context_camera_box_target[i, 6:9] = self.context_box[i]

    def place_desk_boxes(self, i: int, poses) -> None:
        """Stand env ``i``'s desk boxes at ``poses``, one (x, y, yaw) per box, on the desk.

        Writes the mocap bodies; the next ``mj_forward`` or step moves the geoms.
        """
        d = self.datas[i]
        for j, (x, y, yaw) in enumerate(poses):
            self.desk_box_poses[i, j] = (x, y, yaw)
            mocap = self.handles.desk_box_mocap[j]
            d.mocap_pos[mocap] = (x, y, 0.0)
            d.mocap_quat[mocap] = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))

    def stamp_markers(self) -> None:
        """Draw the target marker into the overhead frame about to be observed (design 9).

        Projected through the pose the frame was rendered with: ``cam_xpos``/``cam_xmat`` are
        those ``_refresh`` left after the per-frame jitter, and no physics ran since. ``step``
        calls it before the recorder reads the frame, and the heap loop calls it again after
        it moves ``target_part`` and before the policy reads the frame. A second stamp at the
        same point writes the same pixels (``overlay.stamp_marker`` sets fixed values).
        """
        if not self.cameras:
            return
        h, c = self.handles, self._cam_overhead
        for i in range(self.num_envs):
            t = int(self.target_part[i])
            if t < 0:
                continue
            m, d = self.models[i], self.datas[i]
            uv = overlay.project_point(
                d.cam_xpos[c],
                d.cam_xmat[c],
                float(m.cam_fovy[c]),
                224,
                224,
                d.xpos[h.brick_bodies[t]],
            )
            if uv is not None:
                overlay.stamp_marker(self._overhead[i], *uv)

    def close(self) -> None:
        for r in self.renderers:
            r.close()
        self.renderers = []
        if self.recorder is not None:
            self.recorder.close()

    # -- internals -----------------------------------------------------------------

    def _reset_env(self, i: int) -> None:
        m, d, h = self.models[i], self.datas[i], self.handles
        if self.cfg.episode_draws:
            ep = layout.sample_run_draw(self._rng, self.cfg.spatial_scale)
            ep["slot_colors"] = self.run_draw["slot_colors"]  # the permutation is per run
            bl = layout.BinLayout(
                bin_x=ep["bin_x"], bin_ys=tuple(ep["bin_ys"]), slot_colors=tuple(ep["slot_colors"])
            )
            if self.parts is not None:
                # MS6 rework section 6: heap mode has one bin, at y = 0.0. The release target,
                # the grader's bin centre and the built box are then the same box.
                bl = replace(bl, bin_ys=(0.0, 0.0, 0.0, 0.0))
            self.episode_draws[i] = ep
            self.bin_layouts[i] = bl
            if not self.desk_boxes:  # desk boxes build no bins
                scene.apply_bins(m, bl, heap=self.parts is not None)
            scene.apply_lighting(m, ep)
            scene.apply_cameras(m, ep)
        # Physics DR first: mj_setConst re-derives the model constants AT qpos0 and resets the
        # data to it, so it has to run before the pose below is written.
        if self.cfg.desk_color:
            m.geom_rgba[h.desk_geom, :3] = [random.uniform(*DESK_COLOR_RANGE[c]) for c in "rgb"]
        if self.cfg.physics_dr:
            for b, body in enumerate(h.brick_bodies):
                scale = random.uniform(*BRICK_MASS_SCALE)
                m.body_mass[body] = self._nominal_mass[b] * scale
                m.body_inertia[body] = self._nominal_inertia[b] * scale
                m.geom_friction[h.brick_geoms[b], 0] = scene.BRICK_FRICTION[0] * random.uniform(
                    *BRICK_FRICTION_SCALE
                )
            mujoco.mj_setConst(m, d)
        mujoco.mj_resetData(m, d)
        offset = self.cfg.start_pose_offset_rad
        for k, name in enumerate(ARM_JOINT_NAMES):
            lo, hi = m.jnt_range[h.joint_ids[k]]
            q = HOME_JOINT_POS[name] + (random.uniform(-offset, offset) if offset > 0.0 else 0.0)
            d.qpos[h.joint_qpos[k]] = min(max(q, lo), hi)
        for adr in h.joint_qpos[6:]:
            d.qpos[adr] = GRIPPER_OPEN_POS
        if self.parts is None:
            bl = self.bin_layouts[i]
            poses = layout.sample_brick_poses(
                self.cfg.num_bricks,
                layout.MIN_SEPARATION,
                {
                    "x": layout.scatter_x(bl),
                    "y": layout.SCATTER_Y,
                    "z": (layout.SPAWN_Z, layout.SPAWN_Z),
                    "yaw": (-math.pi, math.pi),
                },
                max_radius=layout.SCATTER_MAX_RADIUS,
            )
            for b, pose in enumerate(poses):
                adr = h.brick_qpos[b]
                d.qpos[adr : adr + 3] = pose[:3]
                d.qpos[adr + 3 : adr + 7] = (
                    math.cos(pose[5] / 2.0),
                    0.0,
                    0.0,
                    math.sin(pose[5] / 2.0),
                )
                d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
        else:
            self._drop_heap(i)
        # Hold the pose the reset left: the first command the rig sends is this same pose.
        d.ctrl[list(h.arm_actuators)] = d.qpos[list(h.joint_qpos[:6])]
        d.ctrl[list(h.gripper_actuators)] = GRIPPER_OPEN_POS
        self._grip_ctrl[i] = GRIPPER_OPEN_POS
        pos, rot = scene.overhead_camera_pose(self.episode_draws[i])
        self.context_camera[i] = np.array([*pos, *rpy_from_quat(rot)], dtype=np.float32)
        if self.parts is not None:
            self.context_camera_box[i, :6] = self.context_camera[i]
            self.context_camera_box_target[i, :6] = self.context_camera[i]
            # Desk boxes: box 0 until the first target names its own box (the only box of
            # rung 1). MS6: the one heap bin.
            self.set_context_box(i, 0 if self.desk_boxes else int(self.destination_bins[i]))
            self.context_target[i] = 0.0
        mujoco.mj_forward(m, d)
        self.episode_length_buf[i] = 0
        if self.recorder is not None:
            self.recorder.begin(
                i,
                self._initial_state(i),
                parts=[p.name for p in self.parts] if self.parts is not None else None,
            )

    def _drop_heap(self, i: int) -> None:
        """Design section 6: a destination, a color per part, a drop on the desk or into the
        tray (``self.tray``), a settle."""
        m, d, h = self.models[i], self.datas[i], self.handles
        k = self.cfg.num_bricks
        zone = None
        if self.desk_boxes:
            # MS7a design, desk boxes: the boxes stand first, from the same per-reset rng, and
            # the heap drops on the tray footprint beside them, walls or not.
            self.place_desk_boxes(i, _boxes.sample_boxes(self._rng, self.desk_boxes))
            self.class_to_box[i] = _boxes.class_map(self._rng, self.desk_boxes)
            self.destination_bins[i] = torch.tensor(
                _boxes.part_destinations(self.parts, self.class_to_box[i])
            )
            zone = layout.tray_bounds()
        else:
            # MS6 rework section 6: heap mode has one bin, at y = 0.0. The release target,
            # the grader's bin centre and the built box are then the same box.
            self.destination_bins[i] = 0
        z_top = layout.HEAP_DROP_Z_MIN + layout.HEAP_DROP_Z_PER_PART * k
        # Hold the arm where the reset left it while the heap settles; nothing is recorded.
        d.ctrl[list(h.arm_actuators)] = d.qpos[list(h.joint_qpos[:6])]
        d.ctrl[list(h.gripper_actuators)] = GRIPPER_OPEN_POS
        for b, part in enumerate(self.parts):
            rgba = layout.BRICK_RGBA[self._rng.choice(layout.BRICK_COLOR_ORDER)]
            for g in h.body_geoms[b]:
                m.geom_rgba[g] = rgba
            # Parked far off the desk until its batch is released; nothing is there to hit.
            adr = h.brick_qpos[b]
            d.qpos[adr : adr + 3] = (HEAP_PARK_X, 0.1 * b, 0.0)
            d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
        # Batches released HEAP_DROP_BATCH_STEPS apart, each over points clear of one another,
        # so the parts pile up on the desk instead of being written into each other in the air.
        for start in range(0, k, layout.HEAP_DROP_BATCH):
            batch = list(range(start, min(start + layout.HEAP_DROP_BATCH, k)))
            points = layout.sample_heap_batch(
                self._rng,
                self.bin_layouts[i],
                [self.parts[b].half_diagonal for b in batch],
                zone=zone,
                tray=self.tray,
            )
            for b, (x, y) in zip(batch, points):
                adr = h.brick_qpos[b]
                d.qpos[adr : adr + 3] = (x, y, self._rng.uniform(layout.HEAP_DROP_Z_MIN, z_top))
                d.qpos[adr + 3 : adr + 7] = _random_quat(self._rng)
                d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
            mujoco.mj_forward(m, d)
            mujoco.mj_step(m, d, nstep=layout.HEAP_DROP_BATCH_STEPS)
        mujoco.mj_step(m, d, nstep=layout.HEAP_SETTLE_STEPS)

    def _initial_state(self, i: int) -> dict[str, np.ndarray]:
        d, h, bl = self.datas[i], self.handles, self.bin_layouts[i]
        state = {
            "robot_joint_position": d.qpos[list(h.joint_qpos)].astype(np.float32),
            "robot_joint_velocity": d.qvel[list(h.joint_dof)].astype(np.float32),
            "robot_root_pose": np.concatenate([d.xpos[h.base_body], d.xquat[h.base_body]]).astype(
                np.float32
            ),
            "robot_root_velocity": np.zeros(6, dtype=np.float32),
            "brick_root_pose": np.stack([d.qpos[a : a + 7] for a in h.brick_qpos]).astype(
                np.float32
            ),
            "brick_root_velocity": np.stack([d.qvel[a : a + 6] for a in h.brick_dof]).astype(
                np.float32
            ),
            "bin_layout": np.array([bl.bin_x, *bl.bin_ys], dtype=np.float32),
        }
        if self.desk_boxes:
            # One destination per part and every box pose, so forensics rebuild the scene.
            state["destination_bin"] = self.destination_bins[i].numpy().astype(np.int64)
            state["desk_box_poses"] = self.desk_box_poses[i].astype(np.float32)
            state["class_to_box"] = self.class_to_box[i].copy()
        elif self.parts is not None:
            state["destination_bin"] = np.array([int(self.destination_bins[i])], dtype=np.int64)
        return state

    def _refresh(self, ids, images: bool) -> None:
        h = self.handles
        robot = self.scene["robot"]
        left = self.scene["finger_left_brick"]._force
        right = self.scene["finger_right_brick"]._force
        force = np.zeros(6)
        for i in ids:
            m, d = self.models[i], self.datas[i]
            robot.refresh(i)
            for b, body in enumerate(h.brick_bodies):
                brick = self.scene[f"brick_{b}"]
                brick._pos[i] = d.xpos[body]
                brick._quat[i] = d.xquat[body]
                brick._vel[i] = d.qvel[h.brick_dof[b] : h.brick_dof[b] + 3]
            left[i] = 0.0
            right[i] = 0.0
            for k in range(d.ncon):
                con = d.contact[k]
                geoms = (con.geom1, con.geom2)
                for side, finger in enumerate(h.finger_geoms):
                    if geoms[0] in finger or geoms[1] in finger:
                        other = geoms[1] if geoms[0] in finger else geoms[0]
                        if other in h.brick_geoms:
                            mujoco.mj_contactForce(m, d, k, force)
                            world = con.frame.reshape(3, 3).T @ force[:3]
                            (left if side == 0 else right)[i, 0, h.brick_geoms.index(other)] += (
                                world
                            )
            if images and self.cameras:
                if self.cfg.frame_jitter:
                    pos, rot = scene.overhead_camera_pose(self.episode_draws[i])
                    jm, jd = FRAME_CAM_JITTER_M, FRAME_CAM_JITTER_DEG
                    dpos = [self._rng.uniform(-jm, jm) for _ in range(3)]
                    drot = [self._rng.uniform(-jd, jd) for _ in range(3)]
                    m.cam_pos[self._cam_overhead] = [p + j for p, j in zip(pos, dpos)]
                    m.cam_quat[self._cam_overhead] = quat_mul(
                        compose_offset_rot(rot, drot), cameras.ROS_TO_MUJOCO_CAM
                    )
                    # Fixed-camera world poses are derived by mj_camlight, which mj_step already
                    # ran before this edit; re-derive them or the renderer uses the old pose.
                    mujoco.mj_camlight(m, d)
                r = self.renderers[i]
                r.update_scene(d, camera=self._cam_wrist)
                self._wrist[i] = r.render()
                r.update_scene(d, camera=self._cam_overhead)
                self._overhead[i] = r.render()
                if self.parts is not None:
                    r.enable_depth_rendering()
                    try:
                        r.update_scene(d, camera=self._cam_overhead)
                        depth_m = r.render()
                    finally:
                        r.disable_depth_rendering()
                    self._depth[i] = noisy_depth_mm(depth_m, self._overhead[i], self._np_rng)
        if self.parts is None:
            # MS4 c2 design 5.2 and 5.3: the privileged channel after the forces and the brick
            # states are current, because the target rule reads both. A heap has no privileged
            # channel (the loop below is a no-op for it), so skip these two grader passes.
            stowed = per_brick_stowed(self)
            held = held_brick_from_forces(
                torch.from_numpy(left).norm(dim=-1).squeeze(1),
                torch.from_numpy(right).norm(dim=-1).squeeze(1),
                min_force=HELD_MIN_FORCE,
            )
        nb = self.cfg.num_bricks
        if self.parts is None:
            for i in ids:
                d = self.datas[i]
                row = self.context_privileged[i]
                row[:] = 0.0
                for b, adr in enumerate(h.brick_qpos):
                    row[9 * b : 9 * b + 3] = d.qpos[adr : adr + 3]
                    row[9 * b + 3 : 9 * b + 9] = _rot6(d.qpos[adr + 3 : adr + 7])
                brick_pos = torch.stack(
                    [self.scene[f"brick_{b}"].data.root_pos_w[i] for b in range(nb)]
                )
                # The PICK POINT, not the tool body: `EnvOracle._next_brick` takes its nearest
                # candidate by the distance from `SceneSnapshot.pick_point`, which is the tool
                # body plus the rotated PICK_OFFSET_MEASURED. The two agree only with a level
                # tool and part by ~3 cm at a tilted approach, which flips this one-hot on near
                # ties -- a mislabelled target in the channel the policy is asked to read.
                pick_point_xy = (
                    robot.data.body_pos_w[i, h.tool_body]
                    + quat_apply(robot.data.body_quat_w[i, h.tool_body], _PICK_OFFSET)
                )[:2]
                target = target_brick(
                    int(held[i]),
                    brick_pos,
                    pick_point_xy,
                    stowed[i],
                    layout.top_down_x_limit(self.bin_layouts[i]),
                )
                if target >= 0:
                    row[54 + target] = 1.0
                row[60 : 60 + nb] = stowed[i].numpy().astype(np.float32)
