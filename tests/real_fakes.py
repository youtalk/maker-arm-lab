"""Test doubles for the MS5 real env. Each task that needs a new one adds it here."""

from __future__ import annotations

import threading
import time

import zmq

from maker_arm_lab.real.perceive import proto


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
