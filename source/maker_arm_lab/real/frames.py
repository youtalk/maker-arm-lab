"""Camera messages as the loop keeps them: numpy and stdlib only, no rclpy.

``real.ros`` turns each Conduit message into one of these; the latency tools, the image
sources and the tests build them directly. Each record keeps two receive clocks:
``arrival_ns`` is the Thor's wall clock (``time.time_ns``, chrony-disciplined), comparable with
the capture stamp the iPhone writes, and ``arrival_mono_ns`` is ``time.monotonic_ns``, which
ages a frame and times intervals without jumps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Arrival:
    stamp_ns: int  # the header stamp: the ARKit capture time, on the iPhone's clock
    arrival_ns: int  # the Thor's wall clock at receipt
    arrival_mono_ns: int  # the Thor's monotonic clock at receipt

    @property
    def latency_ms(self) -> float:
        """Arrival minus the capture stamp, ms: the true latency plus the clock offset."""
        return (self.arrival_ns - self.stamp_ns) / 1e6


@dataclass(frozen=True)
class JpegFrame:
    at: Arrival
    data: bytes
    frame_id: str


@dataclass(frozen=True)
class DepthFrame:
    at: Arrival
    depth_mm: np.ndarray  # (H, W) uint16 millimetres, 0 invalid
    frame_id: str


@dataclass(frozen=True)
class CameraInfo:
    at: Arrival
    width: int
    height: int
    k: tuple[float, ...]  # row-major 3x3
    frame_id: str
