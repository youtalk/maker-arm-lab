"""The perception worker's wire format (MS5 design section 4.2): numpy and the stdlib only.

The worker runs in its own venv with CUDA torch and the segmentation model; the loop's
``maker_arm_lab.real.perceive`` is its client. Neither imports the other's code: this file is
the whole contract, as ``lerobot_stack/policy_protocol.py`` is for the policy, and its
framing is the same: a UTF-8 JSON header, then (name, npy bytes) frame pairs.

Every request that carries an image carries its camera (``camera_header``): ``k`` (row-major
3x3), ``size`` (width, height) and ``pose`` (world from camera, ROS optical frame: ``pos``
and ``quat`` (w, x, y, z)). World is the arm's base frame.

    hello   {protocol}                                  -> hello_ack {protocol, worker_id}
    boxes   {camera, depth_camera} + rgb_jpeg[, depth]   -> boxes {poses: [[x, y, yaw], ...]}
    parts   {camera, depth_camera} + rgb_jpeg[, depth]   -> parts {detections: [detection, ...]}
    marker  {camera} + rgb_jpeg                          -> marker {point: [x, y, z] | null}
    wrist   {camera, prior: [x, y]} + wrist_rgb          -> wrist {look: [x, y, yaw] | null}
    lift    {camera} + wrist_rgb                          -> lift {cls: str | null}
    shutdown                                             -> shutdown_ack

A detection is ``{centre: [x, y, z], quat: [w, x, y, z], size: [3 extents along the body
axes], cls: one of PART_CLASSES, score}``, in the worker's order of preference. ``marker`` is
the FK pick point (between the jaws, ``grasp_center``) as the gripper tag shows it: the
detected tag pose composed with the tag-to-pick-point offset that A2 measures, in world
coordinates. It is not the tag's own centre: the client subtracts the FK pick point from it
(``perception.MarkerBias``'s rule), so any other point would bias every detection by the
tag's mount offset. ``wrist`` finds the part nearest ``prior`` in the wrist frame and returns
its centre and the heading of its body x axis; ``lift`` classifies the part between the
jaws. ``depth_camera`` is null when no depth came with the frame. Errors reply
``{type: error, error, message}``. ``rgb_jpeg`` is the JPEG as a 1-D uint8 array, ``depth``
(H, W) uint16 millimetres (0 invalid), ``wrist_rgb`` (H, W, 3) uint8.
"""

import io
import json
import math

import numpy as np

PROTOCOL_VERSION = 1
PART_CLASSES = ("slope", "tall", "narrow", "wide")
"""The rig's four classes, ``maker_arm_lab.core.parts.BASE_CLASSES`` (MS7a), pinned by
tests/test_perception_protocol.py."""
DETECTION_LENGTHS = {"centre": 3, "quat": 4, "size": 3}


def _pack(array) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(array), allow_pickle=False)
    return buf.getvalue()


def _unpack(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def send_message(socket, header: dict, arrays: dict | None = None) -> None:
    frames = [json.dumps(header).encode()]
    for name, array in (arrays or {}).items():
        frames += [name.encode(), _pack(array)]
    socket.send_multipart(frames)


def recv_message(socket) -> tuple[dict, dict]:
    frames = socket.recv_multipart()
    header = json.loads(frames[0].decode())
    rest = frames[1:]
    if len(rest) % 2:
        raise ValueError("an odd number of frames after the header")
    return header, {rest[i].decode(): _unpack(rest[i + 1]) for i in range(0, len(rest), 2)}


def camera_header(k, size, pos, quat) -> dict:
    return {
        "k": [float(v) for v in k],
        "size": [int(v) for v in size],
        "pose": {"pos": [float(v) for v in pos], "quat": [float(v) for v in quat]},
    }


def validate_detection(d: dict) -> None:
    """Raise ValueError unless ``d`` is one well-formed detection."""
    for key, n in DETECTION_LENGTHS.items():
        values = d.get(key)
        ok = isinstance(values, list) and len(values) == n
        if not ok or not all(isinstance(v, int | float) and math.isfinite(v) for v in values):
            raise ValueError(f"detection {key} must be {n} finite numbers, got {values!r}")
    if d.get("cls") not in PART_CLASSES:
        raise ValueError(f"detection cls must be one of {PART_CLASSES}, got {d.get('cls')!r}")
