"""The rig's image sources (MS5 design section 4.2 and P2): the latest frame of each camera,
kept by a background thread and warped to the view the policy was trained on.

``LatestFrame`` is the one-slot store a source writes and the env reads. ``ConduitOverhead``
takes the Conduit messages: ``real.ros.ConduitSubscriber`` calls it on its executor thread,
tests call it directly. The first camera info builds the overhead warp unless the rig gave a
calibrated camera; later ones only record how far fx drifted (ARKit refocuses). Each JPEG is
decoded at the warp's DCT reduction and warped. A JPEG before any camera model, or of another
size than the camera's, is dropped (``dropped``, ``error``); so is a message whose
conversion or callback raised (``on_error``, which ``real.ros.ConduitSubscriber`` calls), so
the frame-age watchdog names the cause. ``bundle`` hands the worker the raw JPEG, the depth
frame of the same capture stamp (else the latest) and both cameras; the depth shares the RGB
frame's pose (one ARKit frame, one optical frame). ``UvcWrist`` reads a V4L2 camera with
OpenCV on its own thread and warps each frame to the simulated wrist view; ``WristFrames``
checks each frame against the camera first, and a frame of another size (a camera that
ignored the requested size) is dropped the same way. OpenCV comes from the ``real`` group and
is imported only there.
"""

from __future__ import annotations

import io
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image

from maker_arm_lab.real import camera_model
from maker_arm_lab.real.camera_model import Pinhole, Pose, ViewWarp
from maker_arm_lab.real.frames import CameraInfo, DepthFrame, FrameBundle, JpegFrame
from maker_arm_lab.real.perceive import camera_block

DEPTH_KEEP = 8
"""Depth frames kept to match a JPEG's stamp."""


@dataclass(frozen=True)
class Latest:
    view: np.ndarray  # (224, 224, 3) uint8, what the policy reads
    raw: object  # the camera's own frame: JPEG bytes (overhead) or (H, W, 3) uint8 (wrist)
    arrival_mono_ns: int
    stamp_ns: int | None  # the capture stamp; None for a camera without one


class LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Latest | None = None

    def put(self, latest: Latest) -> None:
        with self._lock:
            self._latest = latest

    def get(self) -> Latest | None:
        with self._lock:
            return self._latest


def decode_jpeg(data: bytes, reduce: int = 1) -> tuple[np.ndarray, tuple[int, int]]:
    """(H, W, 3) uint8 RGB of a JPEG decoded at 1/``reduce`` on the DCT, and the JPEG's full
    (width, height)."""
    image = Image.open(io.BytesIO(data))
    size = image.size
    if reduce > 1:
        image.draft("RGB", (size[0] // reduce, size[1] // reduce))
    return np.asarray(image.convert("RGB")), size


class ConduitOverhead:
    def __init__(self, pose: Pose, camera: Pinhole | None = None, plane_z: float = 0.0):
        self.pose, self.plane_z = pose, float(plane_z)
        self.frames = LatestFrame()
        self.camera: Pinhole | None = None
        self.warp: ViewWarp | None = None
        self.depth_camera: Pinhole | None = None
        self._depths: deque[DepthFrame] = deque(maxlen=DEPTH_KEEP)
        self.k_drift = 0.0
        """The largest relative change of fx a camera info showed against the camera in use."""
        self.dropped = 0
        self.error: str | None = None
        if camera is not None:
            self._use(camera)

    def _use(self, camera: Pinhole) -> None:
        self.camera = camera
        self.warp = camera_model.overhead_warp(camera, self.pose, self.plane_z)

    def on_info(self, info: CameraInfo) -> None:
        camera = Pinhole.from_k(info.width, info.height, info.k)
        if self.camera is None:
            self._use(camera)
        else:
            self.k_drift = max(self.k_drift, abs(camera.fx / self.camera.fx - 1.0))

    def on_depth_info(self, info: CameraInfo) -> None:
        self.depth_camera = Pinhole.from_k(info.width, info.height, info.k)

    def on_depth(self, frame: DepthFrame) -> None:
        self._depths.append(frame)

    def on_error(self, key: str, err: Exception) -> None:
        """A Conduit message dropped because its conversion or callback raised."""
        self.dropped += 1
        self.error = f"{key}: {type(err).__name__}: {err}"

    def on_jpeg(self, frame: JpegFrame) -> None:
        warp, camera = self.warp, self.camera
        if warp is None:
            self.dropped += 1
            return
        image, size = decode_jpeg(frame.data, warp.jpeg_reduce())
        if size != (camera.width, camera.height):
            self.dropped += 1
            self.error = f"a {size[0]}x{size[1]} JPEG for a {camera.width}x{camera.height} camera"
            return
        view = warp.apply(image)
        self.frames.put(Latest(view, frame.data, frame.at.arrival_mono_ns, frame.at.stamp_ns))

    def bundle(self, latest: Latest) -> FrameBundle:
        quat = self.pose.quat()
        header = {"camera": camera_block(self.camera, self.pose.pos, quat), "depth_camera": None}
        arrays = {"rgb_jpeg": np.frombuffer(latest.raw, dtype=np.uint8)}
        depths = list(self._depths)
        if depths and self.depth_camera is not None:
            depth = next((d for d in depths if d.at.stamp_ns == latest.stamp_ns), depths[-1])
            header["depth_camera"] = camera_block(self.depth_camera, self.pose.pos, quat)
            arrays["depth"] = depth.depth_mm
        return FrameBundle(header, arrays)


class WristFrames:
    """The wrist camera's frames: ``camera`` is its raw frame's pinhole; ``warp`` None means
    the camera already serves the simulated 224x224 wrist view. A frame that is not
    ``camera``'s size, or whose warp raises, is dropped (``dropped``, ``error``)."""

    def __init__(self, camera: Pinhole, warp: ViewWarp | None):
        self.camera, self.warp = camera, warp
        self.frames = LatestFrame()
        self.dropped = 0
        self.error: str | None = None

    def on_frame(self, rgb: np.ndarray, arrival_mono_ns: int) -> None:
        want = (self.camera.height, self.camera.width, 3)
        if rgb.shape != want:
            self.dropped += 1
            self.error = (
                f"a {rgb.shape[1]}x{rgb.shape[0]} wrist frame for a "
                f"{self.camera.width}x{self.camera.height} camera"
            )
            return
        try:
            view = rgb if self.warp is None else self.warp.apply(rgb)
        except Exception as err:  # noqa: BLE001 -- the capture thread must keep running
            self.dropped += 1
            self.error = f"wrist warp: {type(err).__name__}: {err}"
            return
        self.frames.put(Latest(view, rgb, arrival_mono_ns, None))


class UvcWrist(WristFrames):
    """A V4L2 wrist camera, read with OpenCV on its own thread."""

    def __init__(self, device: str, camera: Pinhole, warp: ViewWarp | None):
        import cv2

        super().__init__(camera, warp)
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open the wrist camera {device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="wrist", daemon=True)

    def start(self) -> UvcWrist:
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, bgr = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            arrival = time.monotonic_ns()
            self.on_frame(np.ascontiguousarray(bgr[..., ::-1]), arrival)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()
