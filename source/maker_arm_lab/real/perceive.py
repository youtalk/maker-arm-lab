"""The real perception backend (MS5 design section 4.3 and P2): what ``core.loop`` calls
under ``perceived`` when the env is a rig, backed by the perception worker over ZMQ.

``RealPerception`` has ``core.loop.SimPerception``'s seven members. The loop builds its models
from it per env and hands them what ``overhead``, ``wrist_look`` and ``lift_class`` return;
here those reads take the rig's frames, not the simulator's truth:

- ``overhead`` takes the frame the policy last saw (JPEG, depth, both cameras) and the FK pick
  point. ``RealPerceiver.perceive`` sends it to the worker and puts the detections in the
  loop's slots, the first ``num_slots`` in the worker's order.
- ``RealPerceiver.reset_episode`` perceives the boxes and labels them
  (``real.rig.label_boxes``); anything but two boxes a side holds the arm.
- ``RealMarker`` looks for the gripper tag in the frame the last perceive call sent and keeps
  the running mean of look minus FK point, as ``perception.MarkerBias`` does. The worker's
  ``point`` is the FK pick point as the tag shows it (the tag pose plus the offset A2
  measures, ``perception_protocol``), not the tag's centre.
- ``wrist_look`` returns the frozen marker point, the selection's perceived centre, as the
  look's prior; ``RealWrist.look`` finds the target near it in the wrist frame.
- ``lift_class`` classifies the part between the jaws in the wrist frame; the perceiver's
  ``classify`` returns that class as it is, and ``RealClassViews.view`` takes a fresh wrist
  frame for the third view.

No noise is drawn: the knob scales the loop passes are not read, and the caller runs the loop
at ``all=0`` so that ``core.torque.ForceSensor`` adds none either. The truth metrics
(``Detection.sim_index`` against the truth, ``MarkerBias.error_mm``) have no truth to read: a
slot is only a slot, and the error reads NaN. Every worker call tags the tick
(``TickLog.note_perception``); a timeout, a worker error or a malformed reply holds the arm
through ``env.trip``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import zmq

from maker_arm_lab.core import desk_boxes
from maker_arm_lab.core.perception import Detection, MarkerBias, PerceivedState
from maker_arm_lab.real.rig import fold_box_yaw, label_boxes

PERCEPTION_STACK = Path(__file__).resolve().parents[3] / "perception_stack"
WORKER_TIMEOUT_MS = 2000
"""A worker call with no reply within this holds the arm; the perception tick cap (1 s)
usually trips first."""


def _load_protocol():
    """``perception_stack/perception_protocol.py``, loaded by path: it lives outside the
    package so the worker's venv can take the file alone."""
    name = "perception_protocol"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PERCEPTION_STACK / "perception_protocol.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


proto = _load_protocol()


def _finite(values, n: int) -> bool:
    """Whether ``values`` is an ``n``-long sequence of finite numbers: a marker point or a
    wrist look that fails this is a malformed reply, not a miss."""
    return (
        isinstance(values, list | tuple)
        and len(values) == n
        and all(isinstance(v, int | float) and math.isfinite(v) for v in values)
    )


def camera_block(camera, pos, quat) -> dict:
    """The wire's camera block for a ``camera_model.Pinhole`` at ``pos`` and ``quat``."""
    k = (camera.fx, 0.0, camera.cx, 0.0, camera.fy, camera.cy, 0.0, 0.0, 1.0)
    return proto.camera_header(k, (camera.width, camera.height), pos, quat)


class WorkerError(RuntimeError):
    pass


class WorkerClient:
    """A REQ socket to the worker; ``call`` returns the reply or raises ``WorkerError``."""

    def __init__(self, endpoint: str, timeout_ms: int = WORKER_TIMEOUT_MS):
        self._socket = zmq.Context.instance().socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        header, _ = self.call("hello", {"protocol": proto.PROTOCOL_VERSION})
        self.worker_id = str(header.get("worker_id", ""))

    def call(self, kind: str, header: dict | None = None, arrays: dict | None = None):
        proto.send_message(self._socket, {"type": kind, **(header or {})}, arrays)
        try:
            reply, out = proto.recv_message(self._socket)
        except zmq.Again as err:
            raise WorkerError(f"{kind}: no reply within the timeout") from err
        except ValueError as err:
            raise WorkerError(f"{kind}: malformed reply frame: {err}") from err
        expected = {"hello": "hello_ack", "shutdown": "shutdown_ack"}.get(kind, kind)
        if reply.get("type") != expected:
            raise WorkerError(f"{kind}: {reply}")
        return reply, out

    def close(self) -> None:
        self._socket.close(0)


@dataclass(frozen=True)
class OverheadLook:
    """One perceive call's input: the frame bundle and the FK pick point."""

    header: dict
    arrays: dict
    pick: tuple[float, float, float]


class RealPerceiver:
    """``perception.Perceiver``'s surface on the worker."""

    def __init__(self, backend: RealPerception):
        self.backend = backend
        self.views = None  # the loop sets ``class_views``' model here
        self._box_poses: tuple[tuple[float, float, float], ...] = ()
        self.last_look: OverheadLook | None = None

    @property
    def box_poses(self) -> tuple[tuple[float, float, float], ...]:
        return self._box_poses

    @property
    def bias(self) -> tuple[float, float, float]:
        """No truth on a rig: the camera's bias is unknown."""
        return (math.nan, math.nan, math.nan)

    @property
    def box_sizes(self) -> tuple[tuple[float, float, float], ...]:
        """The rig's boxes are the MS7a box (``desk_boxes.interior``): no size is perceived."""
        return desk_boxes.interiors(len(self._box_poses))

    def reset_episode(self, true_boxes, true_sizes=()) -> None:
        """Perceive and label the four boxes in the current frame. ``true_boxes`` is the
        env's blind (NaN) poses and ``true_sizes`` the MS7e box draw's sizes; neither is read."""
        frame = self.backend.env.overhead_frame()
        reply, _ = self.backend.request("boxes", frame.header, frame.arrays)
        try:
            poses = [(x, y, fold_box_yaw(yaw)) for x, y, yaw in reply.get("poses", [])]
            self._box_poses = tuple(label_boxes(poses))
        except ValueError as err:
            self.backend.env.trip("boxes", str(err))

    def perceive(self, look: OverheadLook) -> PerceivedState:
        self.last_look = look
        reply, _ = self.backend.request("parts", look.header, look.arrays)
        detections = []
        for slot, d in enumerate(reply.get("detections", [])[: self.backend.num_slots]):
            try:
                proto.validate_detection(d)
            except ValueError as err:
                self.backend.env.trip("perception", str(err))
            detections.append(
                Detection(
                    centre=tuple(float(v) for v in d["centre"]),
                    quat=tuple(float(v) for v in d["quat"]),
                    size=tuple(float(v) for v in d["size"]),
                    cls=d["cls"],
                    sim_index=slot,
                )
            )
        return PerceivedState(detections, self._box_poses)

    def classify(self, true_cls: str, part: int | None = None) -> str:
        """The wrist class ``lift_class`` measured, as it is."""
        return true_cls


class RealMarker:
    """``perception.MarkerBias``'s surface: the gripper tag, looked for in the frame of the
    last perceive call.

    The worker's ``point`` is the pick point as the tag shows it: the detected tag pose
    composed with the tag-to-pick-point offset that A2 measures. ``look`` subtracts the FK
    pick point (``rig.pick_point``) from it, which is ``MarkerBias``'s definition of the
    marker's world point, so the estimate is the camera's bias alone. A worker that returned
    the tag's own centre would add the tag's mount offset to the estimate, and ``correct``
    would move every detection by it."""

    def __init__(self, backend: RealPerception, perceiver: RealPerceiver):
        self.backend, self.perceiver = backend, perceiver
        self.looks = 0
        self.misses = 0
        self._sum = (0.0, 0.0, 0.0)

    @property
    def estimate(self) -> tuple[float, float, float]:
        return tuple(s / max(self.looks, 1) for s in self._sum)

    def look(self, fk_point) -> None:
        last = self.perceiver.last_look
        header = {"camera": last.header["camera"]}
        reply, _ = self.backend.request("marker", header, {"rgb_jpeg": last.arrays["rgb_jpeg"]})
        point = reply.get("point")
        if point is None:
            self.misses += 1
            return
        if not _finite(point, 3):
            self.backend.env.trip("perception", f"marker point {point!r}")
            return
        self._sum = tuple(s + (float(v) - float(p)) for s, v, p in zip(self._sum, point, fk_point))
        self.looks += 1

    correct = MarkerBias.correct

    def error_mm(self) -> float:
        return math.nan


class RealWrist:
    """``perception.WristCamera``'s surface: the target near the prior in the wrist frame."""

    def __init__(self, backend: RealPerception):
        self.backend = backend

    def look(self, xy, yaw: float, hidden: bool):
        env = self.backend.env
        header = {"camera": env.wrist_camera_header(), "prior": [float(xy[0]), float(xy[1])]}
        reply, _ = self.backend.request("wrist", header, {"wrist_rgb": env.wrist_frame()})
        look = reply.get("look")
        if look is None:
            return None
        if not _finite(look, 3):
            env.trip("perception", f"wrist look {look!r}")
            return None
        return tuple(float(v) for v in look)


class RealClassViews:
    """``perception.ClassViews``' surface: a third view is a fresh wrist classification."""

    def __init__(self, backend: RealPerception):
        self.backend = backend

    def view(self, part: int, true_cls: str, rng=None) -> str:
        cls, _ = self.backend.lift_class(self.backend.env, 0)
        return true_cls if cls is None else cls


class RealPerception:
    """The loop's perception backend on a rig (``core.loop.backend_of``)."""

    def __init__(self, env, client: WorkerClient, num_slots: int):
        self.env, self.client, self.num_slots = env, client, int(num_slots)

    def perceiver(self, run_seed: int, i: int, scales: dict, class_names=()) -> RealPerceiver:
        return RealPerceiver(self)

    def marker(self, perceiver, run_seed: int, i: int) -> RealMarker:
        return RealMarker(self, perceiver)

    def class_views(
        self, run_seed: int, i: int, scales: dict, k: int, class_names=()
    ) -> RealClassViews:
        return RealClassViews(self)

    def wrist(self, run_seed: int, i: int, scales: dict) -> RealWrist:
        return RealWrist(self)

    def overhead(self, env, rig, i: int) -> OverheadLook:
        frame = env.overhead_frame()
        return OverheadLook(frame.header, frame.arrays, tuple(rig.pick_point(i).tolist()))

    def wrist_look(self, env, rig, i: int, b: int):
        """The prior of the look: the frozen marker point, not a truth read."""
        x, y, _ = env.marker_points[i].tolist()
        return x, y, math.nan, False

    def lift_class(self, env, i: int):
        header = {"camera": env.wrist_camera_header()}
        reply, _ = self.request("lift", header, {"wrist_rgb": env.wrist_frame()})
        cls = reply.get("cls")
        if cls is None:
            return None, -1
        if cls not in proto.PART_CLASSES:
            # The loop maps the class to a box (class_to_box[cls]); an unknown one would
            # raise there, mid-carry, instead of holding the arm here.
            self.env.trip("perception", f"lift class {cls!r}")
        return str(cls), 0

    def request(self, kind: str, header: dict, arrays: dict):
        self.env.ticks.note_perception()
        try:
            return self.client.call(kind, header, arrays)
        except WorkerError as err:
            self.env.trip("perception", str(err))
