"""The rig as the loop needs it (MS5 design sections 4.1 and 7.1): the calibrated geometry in
one JSON file, and the episode sheet's box draw with labels a rig can perceive. Stdlib and
``core`` only.

Box labels. The simulator knows box j by index; on the desk the four near-black boxes look
alike. So the sheet and the loop both label boxes by where they stand. Boxes 0 and 2 stand on
the +y side and boxes 1 and 3 on the -y side, as ``desk_boxes.sample_boxes`` places them, and
on each side the box with the smaller x takes the lower label. Two boxes of one side never
share an x: the side band is 4 cm deep in |y| and a box is 10 cm across, so their centres
differ by more than a box width in x. The sheet relabels its draw by this rule and carries
the class map through the relabeling, which keeps the class map uniform over the boxes. The
loop labels the perceived boxes by the same rule (``label_boxes``); the sheet is never a loop
input.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from maker_arm_lab.core import cameras, desk_boxes
from maker_arm_lab.core.parts import BASE_CLASSES
from maker_arm_lab.core.profile import load_profile

NUM_BOXES = 4
NUM_MOTORS = 7
"""Six arm joints and the gripper, the order of ``maker_arm_rs.profile()["joints"]``."""
CAP_KEYS = ("tau_max", "kp", "kd")
"""The profile keys ``caps`` scales. The error clamp ``e_max`` follows kp (``capped_e_max``)."""
ARM_DOF = 6
E_MAX_DEFAULT_RAD = 0.03
"""The SDK's default error clamp per arm joint (``TrackingParams::default().e_max``,
INITIAL there); ``tests/test_real_rig.py`` pins it to the SDK's behaviour."""
CALIBRATION_KEYS = ("direction", "offset")


@dataclass(frozen=True)
class RigConfig:
    """The rig's geometry and the arm session's settings. Every pose is in the arm's base
    frame; camera poses are world from camera, ROS optical frame, (w, x, y, z)."""

    overhead_pos: tuple[float, float, float] = tuple(cameras.OVERHEAD_CAM_POS)  # INITIAL: A2
    overhead_quat: tuple[float, float, float, float] = tuple(
        cameras.OVERHEAD_CAM_ROT
    )  # INITIAL: A2
    overhead_k: tuple[float, ...] | None = None
    """None: Conduit's ``camera/wide/camera_info`` gives it (P6 measures it)."""
    overhead_size: tuple[int, int] | None = None
    wrist_k: tuple[float, ...] | None = None
    """None: the wrist source already serves the simulated wrist view (the twin, wave 2)."""
    wrist_size: tuple[int, int] | None = None
    wrist_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # INITIAL: A2 hand-eye
    """The real wrist camera's axes in the simulated mount's frame (``robots.frames``)."""
    wrist_device: str = "/dev/video0"  # INITIAL: A2 confirms the UVC device
    can_interface: str = "can0"
    rt: tuple[int, int] | None = (80, 5)
    """SCHED_FIFO priority and core of the SDK loop: the Thor's T2 setting."""
    tracking_params: dict = field(default_factory=lambda: {"tau_th": [2.0] * 6})  # INITIAL: A1
    tray_lean_deg: float = 30.0
    desk_z: float = 0.0
    caps: dict | None = None
    """The session's reduced caps: ``{"tau_max": [7], "kp": [7], "kd": [7]}``, each a scale in
    (0, 1] of ``maker_arm_rs.profile()``'s per-joint value (``reduced_profile``). None, or a
    key left out, keeps the profile's value. The exposure steps of the MS5 sessions set it."""
    calibration: dict | None = None
    """The motor to URDF map: ``{"direction": [7], "offset": [7]}``, joint = direction *
    (motor - offset). None keeps the profile's identity map (session 2 measures it). Only the
    identity is accepted until the profile's limits are re-expressed in joint coordinates."""
    gripper_map: tuple[float, float] | None = None
    """The gripper's motor angle to jaw gap line, ``gap = a * angle + b`` in m, from the
    caliper (``scripts/real_gripper_map.py``). None keeps the provisional map of
    ``core.profile`` (``real.arm.jaw_fraction``)."""

    @classmethod
    def from_json(cls, path) -> RigConfig:
        data = json.loads(Path(path).read_text())
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown rig keys {sorted(unknown)}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _seven(name: str, values) -> list[float]:
    if isinstance(values, (str, bytes)) or len(values) != NUM_MOTORS:
        raise ValueError(f"{name} needs {NUM_MOTORS} values, got {values!r}")
    out = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name}[{i}] = {v!r} is not a finite number")
        out.append(float(v))
    return out


def reduced_profile(rig: RigConfig) -> dict:
    """``maker_arm_rs.profile()`` with the rig's ``caps`` scaling each joint's ``tau_max``,
    ``kp`` and ``kd`` (the controller's ``e_max`` follows kp: ``session_tracking_params``) and
    its ``calibration`` setting each joint's ``direction`` and
    ``offset``, which for now must be the identity (direction 1, offset 0): the profile's
    limits are still the motor frame's. A scale outside (0, 1], a non-identity calibration or
    an unknown key raises ValueError naming it (a group
    that is not an object, TypeError); the SDK
    checks the result again when the arm connects (``Arm.sim``, ``Arm.socketcan``)."""
    profile = load_profile()
    joints = profile["joints"]
    for group, allowed, value in (
        ("caps", CAP_KEYS, rig.caps),
        ("calibration", CALIBRATION_KEYS, rig.calibration),
    ):
        if value is None:
            continue
        if not isinstance(value, dict):
            raise TypeError(f"{group} must be an object, got {value!r}")
        unknown = sorted(set(value) - set(allowed))
        if unknown:
            raise ValueError(f"{group} has unknown keys {unknown}; allowed {list(allowed)}")
    for key, scales in (rig.caps or {}).items():
        for i, s in enumerate(_seven(f"caps.{key}", scales)):
            if not 0.0 < s <= 1.0:
                raise ValueError(f"caps.{key}[{i}] = {s} is outside (0, 1]")
            joints[i][key] = float(joints[i][key]) * s
    identity = {"direction": 1.0, "offset": 0.0}
    for key, values in (rig.calibration or {}).items():
        for i, v in enumerate(_seven(f"calibration.{key}", values)):
            if v != identity[key]:
                raise ValueError(
                    f"calibration.{key}[{i}] = {v}: only the identity calibration is accepted "
                    "while the profile's joint limits (q_lo, q_hi) are still in the motor frame; "
                    "the SDK clamp and check_traj read them as joint coordinates, so resolve "
                    "the limit frame first"
                )
            joints[i][key] = v
    return profile


def _kp_scales(caps: dict | None) -> list[float]:
    if caps is not None and not isinstance(caps, dict):
        raise TypeError(f"caps must be an object, got {caps!r}")
    if not caps or "kp" not in caps:
        return [1.0] * NUM_MOTORS
    scales = _seven("caps.kp", caps["kp"])
    for i, s in enumerate(scales):
        if not 0.0 < s <= 1.0:
            raise ValueError(f"caps.kp[{i}] = {s} is outside (0, 1]")
    return scales


def capped_e_max(kp_scales: Sequence[float] | None) -> list[float]:
    """The error clamp per arm joint that keeps the spring ceiling ``kp * e_max`` at the
    profile's (ruling R33): ``E_MAX_DEFAULT_RAD / kp_scale``. A reduced kp alone would also
    cut the most torque the spring can add to the gravity feed-forward, by the same scale, and
    an arm whose model is light then sags to its limit (at kp 0.25 the bench's j2 ran 1.5 rad
    with the model's mass at 0.65 of the plant's). The reduced ``tau_max`` still bounds the
    command. None: the profile's kp (scale 1)."""
    scales = [1.0] * NUM_MOTORS if kp_scales is None else list(kp_scales)
    return [E_MAX_DEFAULT_RAD / float(s) for s in scales[:ARM_DOF]]


def session_tracking_params(rig: RigConfig, params: dict) -> dict:
    """``params`` (the rig's ``tracking_params`` with any override) with ``e_max`` set by
    ruling R33 from the rig's ``caps.kp`` (``capped_e_max``). An explicit ``e_max`` (six
    values) wins only where it does not exceed that; past it, ValueError naming the joint."""
    out = dict(params)
    cap = capped_e_max(_kp_scales(rig.caps))
    if "e_max" not in out:
        out["e_max"] = cap
        return out
    given = out["e_max"]
    if isinstance(given, (str, bytes)) or not hasattr(given, "__len__") or len(given) != ARM_DOF:
        raise ValueError(f"e_max needs {ARM_DOF} values, got {given!r}")
    for j, (v, c) in enumerate(zip(given, cap)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"e_max[{j}] = {v!r} is not a finite number")
        if v > c * (1.0 + 1e-12):
            raise ValueError(
                f"e_max[{j}] = {v} exceeds {c:.6g} rad, {E_MAX_DEFAULT_RAD} over the joint's kp "
                "scale: past it the spring could push more than the profile's kp * e_max"
            )
    out["e_max"] = [float(v) for v in given]
    return out


def fold_box_yaw(yaw: float) -> float:
    """A square box's yaw in [-pi/4, pi/4): every quarter turn looks the same."""
    return (float(yaw) + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0


def label_boxes(poses: Sequence[Sequence[float]]) -> list[tuple[float, float, float]]:
    """The four (x, y, yaw) poses in label order: +y near, -y near, +y far, -y far. Raises
    ValueError unless exactly two boxes stand on each side."""
    if len(poses) != NUM_BOXES:
        raise ValueError(f"expected {NUM_BOXES} boxes, got {len(poses)}")
    boxes = [tuple(float(v) for v in p) for p in poses]
    plus = sorted((p for p in boxes if p[1] > 0.0), key=lambda p: p[0])
    minus = sorted((p for p in boxes if p[1] <= 0.0), key=lambda p: p[0])
    if len(plus) != 2 or len(minus) != 2:
        raise ValueError(f"expected two boxes on each side, got {len(plus)} and {len(minus)}")
    return [plus[0], minus[0], plus[1], minus[1]]


def episode_sheet(seed: int, episode: int) -> dict:
    """One real episode's sheet: the four box poses the operator places, in label order, and
    the class map over those labels, drawn with the simulator's own rules from
    ``random.Random(f"ms5-sheet:{seed}:{episode}")``."""
    rng = random.Random(f"ms5-sheet:{seed}:{episode}")
    drawn = [tuple(float(v) for v in p) for p in desk_boxes.sample_boxes(rng, NUM_BOXES)]
    classes = desk_boxes.class_map(rng, NUM_BOXES)
    labelled = label_boxes(drawn)
    label_of = [labelled.index(p) for p in drawn]
    return {
        "seed": seed,
        "episode": episode,
        "box_poses": [list(p) for p in labelled],
        "class_to_box": [label_of[b] for b in classes],
        "classes": list(BASE_CLASSES),
    }
