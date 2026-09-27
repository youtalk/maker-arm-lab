"""The Conduit stream into the loop process (MS5 design section 4.2): the only module under
``source/`` that imports rclpy.

The Conduit topic contract (Conduit design 2026-09-26-ar-aligned-depth, its section 5):
namespace ``/conduit``, frame ``camera_back_optical_frame``, four topics, and every message of
one ARKit frame carries that frame's capture stamp. ``ConduitSubscriber`` turns each message
into a ``real.frames`` record, stamped on arrival before anything else runs, and hands it to
the callback given for its topic, on the executor's own thread. It subscribes best effort,
which matches a reliable publisher as well as a best-effort one.

rclpy loads in the lab's uv venv (Python 3.12, numpy 2) with ROS 2 Jazzy sourced and PyYAML
installed (``uv sync --group real``); checked on the Jetson Thor 2026-09-26. The middleware is
the environment's ``RMW_IMPLEMENTATION``, which must match the Conduit build's transport. A
message whose conversion or callback raises is dropped (``real.frames.relay``): the executor
keeps spinning, ``failures`` counts it, ``error`` keeps the last one and ``on_error`` hears
of it, so a consumer that stops getting frames can name the cause. The CPU suite never
imports this module (``tests/test_real_ros_boundary.py``).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo as CameraInfoMsg
from sensor_msgs.msg import CompressedImage, Image

from maker_arm_lab.real.frames import Arrival, CameraInfo, DepthFrame, JpegFrame, relay

NAMESPACE = "/conduit"
TOPICS = {
    "jpeg": "camera/wide/image_raw/compressed",
    "info": "camera/wide/camera_info",
    "depth": "camera/wide/depth/image_raw",
    "depth_info": "camera/wide/depth/camera_info",
}


def _arrival(header) -> Arrival:
    arrival_ns, mono_ns = time.time_ns(), time.monotonic_ns()
    stamp_ns = int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
    return Arrival(stamp_ns, arrival_ns, mono_ns)


def _jpeg(msg) -> JpegFrame:
    at = _arrival(msg.header)
    return JpegFrame(at, bytes(msg.data), msg.header.frame_id)


def _depth(msg) -> DepthFrame:
    at = _arrival(msg.header)
    if msg.encoding != "16UC1":
        raise ValueError(f"depth encoding {msg.encoding!r}; the contract is 16UC1")
    raw = np.frombuffer(bytes(msg.data), dtype=">u2" if msg.is_bigendian else "<u2")
    depth = raw.reshape(msg.height, msg.step // 2)[:, : msg.width].astype(np.uint16)
    return DepthFrame(at, depth, msg.header.frame_id)


def _info(msg) -> CameraInfo:
    at = _arrival(msg.header)
    k = tuple(float(v) for v in msg.k)
    return CameraInfo(at, int(msg.width), int(msg.height), k, msg.header.frame_id)


class ConduitSubscriber:
    """Subscribes to each topic it has a callback for and spins them on a daemon thread.
    ``on_error(key, err)`` hears of every dropped message; it must not raise."""

    def __init__(
        self,
        namespace: str = NAMESPACE,
        *,
        on_jpeg=None,
        on_info=None,
        on_depth=None,
        on_depth_info=None,
        on_error=None,
        node_name: str = "maker_arm_lab_conduit",
    ):
        self.failures = 0
        self.error: str | None = None
        self._on_error = on_error
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        ns = namespace.rstrip("/")
        wiring = (
            ("jpeg", CompressedImage, on_jpeg, _jpeg),
            ("info", CameraInfoMsg, on_info, _info),
            ("depth", Image, on_depth, _depth),
            ("depth_info", CameraInfoMsg, on_depth_info, _info),
        )
        for key, msg_type, callback, convert in wiring:
            if callback is not None:
                self._node.create_subscription(
                    msg_type,
                    f"{ns}/{TOPICS[key]}",
                    relay(key, convert, callback, self._failed),
                    qos_profile_sensor_data,
                )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, name="conduit", daemon=True)

    def _failed(self, key: str, err: Exception) -> None:
        self.failures += 1
        self.error = f"{key}: {type(err).__name__}: {err}"
        if self._on_error is not None:
            self._on_error(key, err)

    def start(self) -> ConduitSubscriber:
        self._thread.start()
        return self

    def close(self) -> None:
        self._executor.shutdown()
        self._thread.join(timeout=2.0)
        self._node.destroy_node()
        rclpy.try_shutdown()
