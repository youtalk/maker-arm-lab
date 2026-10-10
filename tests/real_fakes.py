"""Test doubles for the MS5 real env. Each task that needs a new one adds it here."""

from __future__ import annotations

import threading
import time

import numpy as np
import zmq

from maker_arm_lab.assets.urdf_prep import ARM_JOINT_NAMES
from maker_arm_lab.core.profile import gripper_angle, sim_profile
from maker_arm_lab.real.arm import ArmReading
from maker_arm_lab.real.camera_model import Pinhole, sim_overhead
from maker_arm_lab.real.frames import FrameBundle
from maker_arm_lab.real.perceive import camera_block, proto
from maker_arm_lab.real.rig import RigConfig
from maker_arm_lab.real.sources import Latest
from maker_arm_lab.robots.frames import HOME_JOINT_POS


class FakeClock:
    """A monotonic clock that advances only when slept on or told to."""

    def __init__(self, start: float = 100.0):
        self.t = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeWorker:
    """An in-process perception worker on a loopback port. It answers each request from what
    the test set (``wrist="prior"`` echoes the prior with yaw 0) and keeps every request.
    ``delay_s`` delays every reply but the hello's. Setting ``malformed_reply`` true after
    construction makes every reply but the hello's an unparseable frame, for the client's
    frame-level error path."""

    def __init__(
        self, boxes=(), detections=(), marker=None, wrist="prior", lift_cls="wide", delay_s=0.0
    ):
        self.boxes = [list(b) for b in boxes]
        self.detections = list(detections)
        self.marker, self.wrist, self.lift_cls, self.delay_s = marker, wrist, lift_cls, delay_s
        self.malformed_reply = False
        self.requests: list[tuple[str, dict, dict]] = []
        self._socket = zmq.Context.instance().socket(zmq.REP)
        port = self._socket.bind_to_random_port("tcp://127.0.0.1")
        self.endpoint = f"tcp://127.0.0.1:{port}"
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _reply(self, kind: str, header: dict) -> dict:
        if kind == "hello":
            return {"type": "hello_ack", "protocol": proto.PROTOCOL_VERSION, "worker_id": "fake"}
        if kind == "boxes":
            return {"type": "boxes", "poses": self.boxes}
        if kind == "parts":
            return {"type": "parts", "detections": self.detections}
        if kind == "marker":
            return {"type": "marker", "point": self.marker}
        if kind == "wrist":
            look = [*header["prior"], 0.0] if self.wrist == "prior" else self.wrist
            return {"type": "wrist", "look": look}
        if kind == "lift":
            return {"type": "lift", "cls": self.lift_cls}
        return {"type": "error", "error": "exception", "message": f"unknown {kind!r}"}

    def _serve(self) -> None:
        while True:
            header, arrays = proto.recv_message(self._socket)
            kind = header.get("type")
            self.requests.append((kind, header, arrays))
            if kind == "shutdown":
                proto.send_message(self._socket, {"type": "shutdown_ack"})
                break
            if kind != "hello":
                time.sleep(self.delay_s)
            if kind != "hello" and self.malformed_reply:
                self._socket.send_multipart([b"not json"])
            else:
                proto.send_message(self._socket, self._reply(kind, header))
        self._socket.close(0)

    def kinds(self) -> list[str]:
        return [kind for kind, _, _ in self.requests]

    def close(self) -> None:
        req = zmq.Context.instance().socket(zmq.REQ)
        req.setsockopt(zmq.RCVTIMEO, 2000)
        req.setsockopt(zmq.LINGER, 0)
        req.connect(self.endpoint)
        proto.send_message(req, {"type": "shutdown"})
        try:
            proto.recv_message(req)
        except zmq.Again:
            pass
        req.close(0)
        self._thread.join(timeout=2.0)


HOME = [HOME_JOINT_POS[n] for n in ARM_JOINT_NAMES]
_JPEG = b"\xff\xd8\xff\xd9"


class FakeArm:
    """``SdkArm``'s surface with no motors: the measured pose is the last row pushed, at once;
    the loop clock is ``clock`` since construction."""

    def __init__(self, clock, q=None, gripper=None):
        self.clock, self.t0 = clock, clock()
        self.q = list(HOME if q is None else q)
        self.gripper = gripper_angle(sim_profile(), 1.0) if gripper is None else gripper
        self.tau = (0.0,) * 6
        self.state, self.alive = "enabled", True
        self.pushes: list[tuple[float, float, list[float]]] = []
        self.holds = 0
        self.stops = 0

    def now(self):
        return self.clock() - self.t0 if self.alive else None

    def read(self) -> ArmReading:
        return ArmReading(
            self.clock() - self.t0,
            tuple(self.q),
            (0.0,) * 6,
            tuple(self.tau),
            self.gripper,
            self.state,
            self.alive,
            None,
        )

    def push(self, due, dt, row):
        self.pushes.append((due, dt, list(row)))
        self.q, self.gripper = list(row[:6]), row[6]

    def telemetry(self):
        return {"max_interval": 0.005, "late_ticks": 0, "gaps": 0}

    def hold(self):
        self.holds += 1

    def stop(self):
        self.stops += 1


class _Fresh:
    """A ``LatestFrame`` whose frame is always ``source.age_s`` old."""

    def __init__(self, source):
        self.source = source

    def get(self):
        s = self.source
        return Latest(s.view, s.raw, time.monotonic_ns() - int(s.age_s * 1e9), 1)


class FakeOverhead:
    def __init__(self, age_s=0.0):
        self.age_s = age_s
        self.view = np.full((224, 224, 3), 90, dtype=np.uint8)
        self.raw = _JPEG
        self.frames = _Fresh(self)
        self.warp = None
        self.error = None

    def bundle(self, latest):
        camera, pose = sim_overhead()
        header = {"camera": camera_block(camera, pose.pos, pose.quat()), "depth_camera": None}
        return FrameBundle(header, {"rgb_jpeg": np.frombuffer(latest.raw, dtype=np.uint8)})


class FakeWrist:
    def __init__(self, age_s=0.0):
        self.age_s = age_s
        self.view = np.full((224, 224, 3), 60, dtype=np.uint8)
        self.raw = self.view
        self.frames = _Fresh(self)
        self.camera = Pinhole.sim(70.0)


def make_env(
    clock=None,
    arm=None,
    overhead=None,
    wrist=None,
    rig=None,
    num_slots=6,
    episode_length_s=60.0,
    dilation=1.0,
    class_to_box=(0, 1, 2, 3),
):
    from maker_arm_lab.real.env import RealEnv

    clock = clock or FakeClock()
    arm = arm or FakeArm(clock)
    env = RealEnv(
        arm,
        overhead or FakeOverhead(),
        wrist or FakeWrist(),
        rig or RigConfig(),
        num_slots=num_slots,
        class_to_box=list(class_to_box),
        episode_length_s=episode_length_s,
        dilation=dilation,
        clock=clock,
        sleep=clock.sleep,
    )
    return env, arm, clock
