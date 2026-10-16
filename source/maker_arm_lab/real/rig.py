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
    (motor - offset). None keeps the profile's identity map (session 2 measures it)."""

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
    ``kp`` and ``kd`` and its ``calibration`` setting each joint's ``direction`` and
    ``offset``. A scale outside (0, 1] or an unknown key raises ValueError naming it (a group
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
    for key, values in (rig.calibration or {}).items():
        for i, v in enumerate(_seven(f"calibration.{key}", values)):
            joints[i][key] = v
    return profile


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
