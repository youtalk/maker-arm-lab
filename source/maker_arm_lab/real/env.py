"""``RealEnv``: the real arm and rig behind the env surface ``core/loop.py`` reads (MS5 design
section 4.3).

One env, the MS7d scene: the tray, four desk boxes, the continuous gripper and the 16-wide
``camera_box_target`` context. Names and shapes are ``mj.env``'s (its module docstring lists
them), so the loop, ``OracleRig`` and the scripts run unchanged:

- measured: ``scene["robot"].data`` from the arm's snapshot and FK (the tool body is the only
  body, index 0), ``obs_buf["images"]`` from the two sources, ``arm_torques(i)``;
- served: ``obs_buf["context"]`` (the camera row, the perceived box of the target's class and
  the loop's target row), ``stamp_markers()``, ``set_context_box(i, box)``,
  ``marker_points``, ``target_part``;
- configured: ``parts`` (placeholder slots), ``class_to_box`` (the episode sheet's), ``tray``,
  ``tray_lean_deg``, ``desk_boxes``, ``destination_bins``, ``stow_half_extents``;
- blind: ``scene["brick_b"]`` parked at ``perception.PARKED_CENTRE`` and at rest, zero finger
  forces, NaN ``desk_box_poses`` and a success term that never fires. A rig has no simulator
  truth: the loop's metrics and the rig's episode stats read these and count no stow. The
  operator counts the boxes (MS5 design section 7.1); the loop's census gives the cycles.

The loop's perception is ``perception_backend`` (``real.perceive.RealPerception``), which the
caller sets once the worker client exists (``core.loop.backend_of``).

``step`` runs the simulator's command path (the SDK clamp on the six arm targets, the gap to
an opening fraction), pushes one row due one period ahead on the arm clock, waits for the next
tick and observes. After its last row the plan holds it, so a stalled loop holds the arm where
the last command put it, as the lockstep simulator's time stands still. ``dilation`` below 1
stretches the period (0.5 runs 15 steps a second, MS5 design section 8). The overhead view
emulates the simulator's nominal overhead camera (``camera_model.sim_overhead``): the
context's camera row is that camera's pose and the marker projects through it.

Every step checks the watchdogs of MS5 design section 8. One that trips calls ``trip``: the
arm holds (``arm.hold``) and ``EvalFailure("watchdog_<reason>")`` ends the run; a step after
that raises at once and pushes nothing.

Frame age, a declared decision: ``FRAME_AGE_CAP_S`` counts from a frame's arrival on this
host's monotonic clock, not from its capture stamp. A capture-stamp age needs the
iPhone-to-Thor clock offset that the P7 read measures, so the capture-to-arrival latency is
not inside the 300 ms cap until wave 2 revisits it with the P7 numbers. The same cap, counted
the same way, applies to the wrist frame.
"""

from __future__ import annotations

import math
import random
import re
import time
import types
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import torch

from maker_arm_lab.assets.urdf_prep import (
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    VENDORED_URDF,
    arm_joint_limits,
)
from maker_arm_lab.core import cameras, layout, overlay
from maker_arm_lab.core import desk_boxes as _boxes
from maker_arm_lab.core import dynamics_model as dm
from maker_arm_lab.core import profile as _profile
from maker_arm_lab.core.eval_loop import EvalFailure
from maker_arm_lab.core.global_ik import GlobalIK
from maker_arm_lab.core.parts import Part
from maker_arm_lab.core.perception import PARKED_CENTRE
from maker_arm_lab.core.quat import rpy_from_quat
from maker_arm_lab.mj.context_names import CONTEXT_CAMERA_NAMES, CONTEXT_DIMS
from maker_arm_lab.real import camera_model, pacing
from maker_arm_lab.real.arm import (
    contact_residual,
    jaw_fraction,
    pick_jacobian,
    pick_point,
    reference_acceleration,
)
from maker_arm_lab.real.frames import FrameBundle
from maker_arm_lab.real.perceive import camera_block
from maker_arm_lab.real.rig import NUM_BOXES, RigConfig
from maker_arm_lab.robots.frames import (
    GRIPPER_OPEN_POS,
    JAW_FULL_GAP,
    TOOL_BODY,
    WRIST_CAM_POS_IN_WRIST,
    WRIST_CAM_ROT_IN_WRIST,
)

FRAME_AGE_CAP_S = 0.300
"""MS5 design section 8: a frame older than this (since its arrival) at a policy call holds."""
PICK_FLOOR_MARGIN_M = 0.005  # INITIAL: A0 sets it on the real desk
"""m under the desk plane the FK pick point may reach before the floor watchdog holds."""
FIRST_FRAME_TIMEOUT_S = 5.0
SLOT_PART = Part("rig_slot", "brick", 0, 0, (0.008, 0.008, 0.008), False, 0.0)
"""What every slot names: a rig knows no catalog part per slot. Only the blind metrics and the
payload read it."""


def _grasp_center_in_wrist() -> np.ndarray:
    """``grasp_center`` in ``link_007``: the vendored URDF's fixed joint, which has no
    rotation."""
    root = ET.parse(VENDORED_URDF).getroot()
    joint = next(j for j in root.iter("joint") if j.get("name") == "grasp_center_joint")
    origin = joint.find("origin")
    if any(float(v) != 0.0 for v in origin.get("rpy", "0 0 0").split()):
        raise ValueError("grasp_center_joint has a rotation; the wrist camera pose assumes none")
    return np.array([float(v) for v in origin.get("xyz").split()])


class _Robot:
    """``scene["robot"]``: the measured joint state, the tool body and the two lookups."""

    def __init__(self):
        self.joint_names = [*ARM_JOINT_NAMES, *GRIPPER_JOINT_NAMES]
        limits = [*arm_joint_limits(), (0.0, GRIPPER_OPEN_POS), (0.0, GRIPPER_OPEN_POS)]
        self.data = types.SimpleNamespace(
            joint_pos=torch.zeros(1, 8),
            joint_vel=torch.zeros(1, 8),
            joint_pos_limits=torch.tensor([limits], dtype=torch.float32),
            body_pos_w=torch.zeros(1, 1, 3),
            body_quat_w=torch.zeros(1, 1, 4),
            computed_torque=torch.zeros(1, 8),
            applied_torque=torch.zeros(1, 8),
        )

    def find_bodies(self, name: str):
        if name != TOOL_BODY:
            raise KeyError(f"the rig tracks only {TOOL_BODY}, not {name}")
        return ([0], [name])

    def find_joints(self, names, preserve_order: bool = False):
        """Isaac's ``find_joints``, as ``mj.env``'s: each entry a name or a regex."""
        found = [
            (index, joint)
            for pattern in names
            for index, joint in enumerate(self.joint_names)
            if joint == pattern or re.fullmatch(pattern, joint)
        ]
        if not preserve_order:
            found.sort()
        return ([i for i, _ in found], [n for _, n in found])


class _Blind:
    """A slot as the metrics read it: parked, at rest."""

    def __init__(self):
        self.data = types.SimpleNamespace(
            root_pos_w=torch.tensor([PARKED_CENTRE], dtype=torch.float32),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_lin_vel_w=torch.zeros(1, 3),
        )


class _Scene:
    def __init__(self, num_slots: int):
        self.env_origins = torch.zeros(1, 3)
        self._items: dict[str, object] = {"robot": _Robot()}
        for b in range(num_slots):
            self._items[f"brick_{b}"] = _Blind()
        for side in ("left", "right"):
            force = torch.zeros(1, 1, num_slots, 3)
            self._items[f"finger_{side}_brick"] = types.SimpleNamespace(
                data=types.SimpleNamespace(force_matrix_w=force)
            )

    def __getitem__(self, key: str):
        return self._items[key]

    def keys(self):
        return self._items.keys()


class RealEnv:
    def __init__(
        self,
        arm,
        overhead,
        wrist,
        rig: RigConfig,
        *,
        num_slots: int,
        class_to_box,
        episode_length_s: float,
        dilation: float = 1.0,
        draw: dict | None = None,
        clock=time.monotonic,
        sleep=time.sleep,
        ik: GlobalIK | None = None,
        dynamics=None,
    ):
        if not 0.0 < dilation <= 1.0:
            raise ValueError(f"dilation must be in (0, 1], got {dilation}")
        if sorted(int(b) for b in class_to_box) != list(range(NUM_BOXES)):
            raise ValueError(f"class_to_box must map the classes onto boxes 0..3: {class_to_box}")
        self.arm, self.overhead_source, self.wrist_source, self.rig = arm, overhead, wrist, rig
        self.device, self.num_envs = "cpu", 1
        self.gripper_mode, self.context_mode = "continuous", "camera_box_target"
        self.tray, self.tray_lean_deg, self.desk_boxes = True, float(rig.tray_lean_deg), NUM_BOXES
        draw = dict(draw) if draw is not None else layout.sample_run_draw(random.Random(0))
        bl = layout.BinLayout(
            bin_x=draw["bin_x"],
            bin_ys=(0.0, 0.0, 0.0, 0.0),
            slot_colors=tuple(draw.get("slot_colors", (0, 1, 2, 3))),
        )
        self.cfg = types.SimpleNamespace(
            num_bricks=int(num_slots),
            bin_layout=bl,
            episode_draws=False,
            frame_jitter=False,
            spatial_scale=1.0,
            episode_length_s=float(episode_length_s),
            cameras=True,
        )
        self.bin_layouts, self.run_draw, self.episode_draws = [bl], draw, [draw]
        self.parts = (SLOT_PART,) * int(num_slots)
        self.class_to_box = np.array([[int(b) for b in class_to_box]], dtype=np.int64)
        self.destination_bins = torch.zeros(1, int(num_slots), dtype=torch.long)
        interior = _boxes.interior(NUM_BOXES)
        self.stow_half_extents = torch.tensor(_boxes.stow_region(interior)[1])
        self.desk_box_poses = np.full((1, NUM_BOXES, 3), np.nan)
        self.desk_box_sizes = np.tile(interior, (1, NUM_BOXES, 1))
        self.perceived_box_poses = None
        self.perception_backend = None
        self.recorder = None
        self.target_part = torch.full((1,), -1, dtype=torch.int16)
        self.marker_points = torch.full((1, 3), math.nan)
        self.oracle_actor = torch.zeros(1, dtype=torch.int16)
        self.max_episode_length = round(float(episode_length_s) * pacing.CONTROL_HZ)
        self.dilation = float(dilation)
        self.period = 1.0 / (pacing.CONTROL_HZ * self.dilation)
        self.profile = _profile.sim_profile()
        self._arm_term = types.SimpleNamespace(
            _joint_names=list(ARM_JOINT_NAMES), clamped=torch.zeros(1, dtype=torch.bool)
        )
        self.action_manager = types.SimpleNamespace(
            total_action_dim=7, get_term=lambda name: self._arm_term
        )
        self._success = torch.zeros(1, dtype=torch.bool)
        self.termination_manager = types.SimpleNamespace(
            active_terms=["time_out", "success"], get_term=lambda name: self._success
        )
        self.ik = ik if ik is not None else GlobalIK()
        self.dynamics = dynamics if dynamics is not None else dm.dynamics(dm.urdf_bodies())
        self.scene = _Scene(int(num_slots))
        self._sim_camera, self._sim_pose = camera_model.sim_overhead()
        self.context_camera = np.array(
            [[*cameras.OVERHEAD_CAM_POS, *rpy_from_quat(cameras.OVERHEAD_CAM_ROT)]],
            dtype=np.float32,
        )
        self.context_box = np.zeros((1, 3), dtype=np.float32)
        self.context_camera_box_target = np.zeros(
            (1, CONTEXT_DIMS["camera_box_target"]), dtype=np.float32
        )
        self.context_camera_box_target[:, : len(CONTEXT_CAMERA_NAMES)] = self.context_camera
        self.context_target = self.context_camera_box_target[:, CONTEXT_DIMS["camera_box"] :]
        self._overhead = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        self._wrist = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        self.obs_buf = {
            "images": {
                "wrist_rgb": torch.from_numpy(self._wrist),
                "overhead_rgb": torch.from_numpy(self._overhead),
            },
            "context": torch.from_numpy(self.context_camera_box_target),
        }
        self.ticks = pacing.TickLog()
        self.holding = False
        self._steps = 0
        self._targets: deque = deque(maxlen=3)
        self._gc_in_wrist = _grasp_center_in_wrist()
        self._wait_for_frames(clock, sleep)
        self._refresh(self.arm.read())
        self.pacer = pacing.Pacer(self.period, clock, sleep)

    # -- the watchdogs ------------------------------------------------------------------------

    def trip(self, reason: str, detail: str):
        """MS5 design section 8: hold the arm where it is and end the run.

        Always raises ``EvalFailure("watchdog_<reason>", detail)``, never returns: callers
        (``real.perceive`` among them) rely on a trip not coming back. Only the first trip
        commands the hold."""
        if not self.holding:
            self.holding = True
            try:
                self.arm.hold()
            except Exception as err:  # noqa: BLE001 -- the failure still ends the run
                detail = f"{detail} (the hold failed: {err})"
        raise EvalFailure(f"watchdog_{reason}", detail)

    def _wait_for_frames(self, clock, sleep) -> None:
        deadline = clock() + FIRST_FRAME_TIMEOUT_S
        while self.overhead_source.frames.get() is None or self.wrist_source.frames.get() is None:
            if clock() > deadline:
                self.trip("frame", f"no camera frame within {FIRST_FRAME_TIMEOUT_S} s")
            sleep(0.01)

    # -- the gym-ish surface ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        if self.holding:
            raise EvalFailure("watchdog_hold", "a step after the arm was held")
        raw = torch.as_tensor(action).detach().to("cpu", torch.float32)
        if raw.shape != (1, 7):
            raise ValueError(f"expected an action of shape (1, 7), got {tuple(raw.shape)}")
        fraction = (raw[:, 6].double() / JAW_FULL_GAP).clamp(0.0, 1.0)
        arm = _profile.clamp_targets_tensor(
            raw[:, :6].double(), fraction, self.profile, self._arm_term.clamped
        )
        target = [float(v) for v in arm[0]]
        grip = _profile.gripper_angle(self.profile, float(fraction[0]))
        now = self.arm.now()
        if now is None:
            self.trip("arm", "the SDK loop stopped ticking")
        self.arm.push(now + self.period, self.period, [*target, grip])
        self._targets.append(target)
        tick = self.ticks.close(self.pacer.wait())
        self._steps += 1
        late = pacing.over_cap(tick)
        if late is not None:
            self.trip("tick", late)
        self._refresh(self.arm.read())
        truncated = torch.tensor([self._steps >= self.max_episode_length])
        return None, None, torch.zeros(1, dtype=torch.bool), truncated, {}

    def _refresh(self, reading) -> None:
        if reading is None:
            self.trip("arm", "no arm snapshot")
        if reading.state != "enabled" or not reading.alive:
            self.trip("arm", f"the arm is {reading.state} (fault {reading.fault})")
        self._reading = reading
        per_jaw = jaw_fraction(self.profile, reading.gripper) * GRIPPER_OPEN_POS
        data = self.scene["robot"].data
        data.joint_pos[0] = torch.tensor([*reading.q, per_jaw, per_jaw])
        data.joint_vel[0, :6] = torch.tensor(reading.dq)
        tau = torch.tensor([*reading.tau, 0.0, 0.0])
        data.computed_torque[0], data.applied_torque[0] = tau, tau
        pos, rot = self.ik.fk(torch.tensor(reading.q, dtype=torch.float64))
        self._tool = (pos.numpy(), rot.numpy())
        data.body_pos_w[0, 0] = pos.float()
        data.body_quat_w[0, 0] = torch.tensor(camera_model.matrix_quat(rot.numpy()))
        pick = pick_point(self.ik, reading.q)
        if pick[2] < self.rig.desk_z - PICK_FLOOR_MARGIN_M:
            self.trip("floor", f"the FK pick point is at z {pick[2]:.4f} m")
        self._observe_images()

    def _observe_images(self) -> None:
        now = time.monotonic_ns()
        frames = []
        for name, source in (("overhead", self.overhead_source), ("wrist", self.wrist_source)):
            latest = source.frames.get()
            if latest is None:
                self.trip("frame", f"no {name} frame")
            age = (now - latest.arrival_mono_ns) / 1e9
            if age > FRAME_AGE_CAP_S:
                error = getattr(source, "error", None)
                self.trip("frame", f"the {name} frame is {1e3 * age:.0f} ms old ({error})")
            frames.append(latest)
        self._overhead_latest, self._wrist_latest = frames
        self._overhead[0] = self._overhead_latest.view
        self._wrist[0] = self._wrist_latest.view

    def stamp_markers(self) -> None:
        """Stamp the frozen marker into the overhead frame about to be observed, projected
        through the nominal camera the view emulates. A second stamp writes the same pixels."""
        if not bool(self.marker_points[0].isfinite().all()):
            return
        uv = camera_model.project(self._sim_camera, self._sim_pose, self.marker_points[0].tolist())
        if uv is not None:
            overlay.stamp_marker(self._overhead[0], *uv)

    def set_context_box(self, i: int, box: int) -> None:
        """Serve box ``box``'s perceived pose as the context box part (``mj.env``'s rule)."""
        poses = (
            self.desk_box_poses if self.perceived_box_poses is None else self.perceived_box_poses
        )
        self.context_box[i] = poses[i, box]
        self.context_camera_box_target[i, 6:9] = self.context_box[i]

    def arm_torques(self, i: int) -> tuple[torch.Tensor, ...]:
        """MS7d L6 and L7 on the rig: the contact residual (``real.arm``), the model's gravity
        torque, the measured torque and the pick Jacobian, float64. ``core.torque`` reads the
        residual as the constraint torque; at the loop's ``all=0`` scales it adds nothing."""
        r = self._reading
        ddq = reference_acceleration(self._targets, self.period)
        residual = contact_residual(self.dynamics, r.q, r.dq, ddq, r.tau)
        gravity = self.dynamics.gravity(list(r.q))
        jac = pick_jacobian(self.ik, r.q)
        f64 = lambda v: torch.tensor(np.asarray(v), dtype=torch.float64)
        return f64(residual), f64(gravity), f64(r.tau), f64(jac)

    # -- what the perception backend reads -------------------------------------------------

    def overhead_frame(self) -> FrameBundle:
        """The worker's view of the frame the policy last saw."""
        return self.overhead_source.bundle(self._overhead_latest)

    def wrist_frame(self) -> np.ndarray:
        return self._wrist_latest.raw

    def wrist_camera_header(self) -> dict:
        """The wrist camera now: FK to ``grasp_center``, back to ``link_007``, then the mount
        (``robots.frames``) and the rig's hand-eye rotation."""
        pos, rot = self._tool
        link = pos - rot @ self._gc_in_wrist
        cam_pos = link + rot @ np.asarray(WRIST_CAM_POS_IN_WRIST)
        cam_rot = (
            rot
            @ camera_model.quat_matrix(WRIST_CAM_ROT_IN_WRIST)
            @ camera_model.quat_matrix(self.rig.wrist_rot)
        )
        return camera_block(self.wrist_source.camera, cam_pos, camera_model.matrix_quat(cam_rot))

    @property
    def overhead_coverage(self) -> float | None:
        warp = getattr(self.overhead_source, "warp", None)
        return None if warp is None else warp.coverage

    def close(self) -> None:
        """The caller owns the arm and the sources; the env holds nothing to close."""
