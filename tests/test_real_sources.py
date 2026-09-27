"""MS5 design section 4.2 and P2: the overhead source on Conduit messages, fed directly."""

import ast
import io
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from maker_arm_lab.core.cameras import WRIST_FOV_DEG
from maker_arm_lab.real.camera_model import Pinhole, overhead_warp, sim_overhead, wrist_warp
from maker_arm_lab.real.frames import Arrival, CameraInfo, DepthFrame, JpegFrame, relay
from maker_arm_lab.real.sources import (
    ConduitOverhead,
    Latest,
    LatestFrame,
    WristFrames,
    decode_jpeg,
)

REAL = Pinhole(640, 480, 312.8, 312.8, 320.0, 240.0)
POSE = sim_overhead()[1]
FRAME_ID = "camera_back_optical_frame"


def _jpeg(image) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _image(seed=0, shape=(480, 640, 3)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def _info(camera):
    k = (camera.fx, 0.0, camera.cx, 0.0, camera.fy, camera.cy, 0.0, 0.0, 1.0)
    return CameraInfo(Arrival(0, 0, 0), camera.width, camera.height, k, FRAME_ID)


def _frame(data, stamp=1):
    return JpegFrame(Arrival(stamp, stamp, time.monotonic_ns()), data, FRAME_ID)


def test_the_latest_frame_keeps_only_the_last_put():
    store = LatestFrame()
    assert store.get() is None
    first, second = (Latest(np.zeros((1,)), b"", k, None) for k in (1, 2))
    store.put(first)
    store.put(second)
    assert store.get() is second


def test_a_jpeg_before_any_camera_model_is_dropped():
    src = ConduitOverhead(POSE)
    src.on_jpeg(_frame(_jpeg(_image())))
    assert src.frames.get() is None and src.dropped == 1


def test_the_first_camera_info_builds_the_warp_and_a_jpeg_becomes_the_view():
    src = ConduitOverhead(POSE)
    src.on_info(_info(REAL))
    data = _jpeg(_image())
    src.on_jpeg(_frame(data, stamp=5))
    latest = src.frames.get()
    decoded, size = decode_jpeg(data)
    want = overhead_warp(REAL, POSE).apply(decoded)
    assert size == (640, 480) and latest.view.shape == (224, 224, 3)
    assert np.array_equal(latest.view, want) and latest.raw == data and latest.stamp_ns == 5


def test_a_later_camera_info_records_drift_and_keeps_the_warp():
    src = ConduitOverhead(POSE)
    src.on_info(_info(REAL))
    warp = src.warp
    src.on_info(_info(replace(REAL, fx=REAL.fx * 1.01, fy=REAL.fy * 1.01)))
    assert src.warp is warp and src.k_drift == pytest.approx(0.01)


def test_a_calibrated_camera_wins_over_camera_info():
    src = ConduitOverhead(POSE, camera=REAL)
    src.on_info(_info(replace(REAL, fx=400.0, fy=400.0)))
    assert src.camera == REAL and src.warp is not None


def test_a_frame_of_another_size_is_dropped_with_an_error():
    src = ConduitOverhead(POSE, camera=REAL)
    src.on_jpeg(_frame(_jpeg(_image(shape=(300, 400, 3)))))
    assert src.frames.get() is None and "400x300" in src.error


def test_a_corrupt_message_is_dropped_with_its_cause_and_the_next_frame_still_lands():
    """MS5 final review M3: through ``frames.relay``, as ``real.ros`` wires it, a JPEG that
    does not decode is counted and named; the executor thread would keep spinning."""
    src = ConduitOverhead(POSE, camera=REAL)
    call = relay("jpeg", lambda msg: msg, src.on_jpeg, src.on_error)
    call(_frame(b"\xff\xd8 not a jpeg"))
    assert src.frames.get() is None and src.dropped == 1
    assert src.error.startswith("jpeg: UnidentifiedImageError")
    call(_frame(_jpeg(_image()), stamp=9))
    assert src.frames.get().stamp_ns == 9


def test_a_wrist_frame_of_the_camera_size_is_warped_to_the_view():
    camera = Pinhole(640, 480, 400.0, 400.0, 320.0, 240.0)
    warp = wrist_warp(camera)
    src = WristFrames(camera, warp)
    rgb = _image()
    src.on_frame(rgb, 7)
    latest = src.frames.get()
    assert np.array_equal(latest.view, warp.apply(rgb)) and latest.raw is rgb
    assert latest.arrival_mono_ns == 7 and src.dropped == 0


@pytest.mark.parametrize("warped", [False, True])
def test_a_wrist_frame_of_another_size_is_dropped_with_an_error(warped):
    """MS5 final review M2: a UVC camera that ignored the requested size."""
    camera = Pinhole(640, 480, 400.0, 400.0, 320.0, 240.0) if warped else Pinhole.sim(WRIST_FOV_DEG)
    src = WristFrames(camera, wrist_warp(camera) if warped else None)
    src.on_frame(_image(shape=(720, 1280, 3)), 1)
    assert src.frames.get() is None and src.dropped == 1
    assert f"1280x720 wrist frame for a {camera.width}x{camera.height}" in src.error


def test_the_bundle_carries_the_jpeg_the_depth_of_its_stamp_and_both_cameras():
    src = ConduitOverhead(POSE, camera=REAL)
    src.on_depth_info(_info(Pinhole(256, 192, 200.0, 200.0, 128.0, 96.0)))
    for stamp, mm in ((7, 700), (8, 800)):
        src.on_depth(
            DepthFrame(Arrival(stamp, stamp, 0), np.full((192, 256), mm, np.uint16), FRAME_ID)
        )
    data = _jpeg(_image())
    src.on_jpeg(_frame(data, stamp=7))
    bundle = src.bundle(src.frames.get())
    assert bundle.arrays["rgb_jpeg"].tobytes() == data and int(bundle.arrays["depth"][0, 0]) == 700
    assert bundle.header["camera"]["size"] == [640, 480]
    assert bundle.header["depth_camera"]["size"] == [256, 192]
    assert bundle.header["camera"]["pose"]["quat"] == pytest.approx(list(POSE.quat()))


def test_without_depth_the_bundle_says_so():
    src = ConduitOverhead(POSE, camera=REAL)
    src.on_jpeg(_frame(_jpeg(_image())))
    bundle = src.bundle(src.frames.get())
    assert bundle.header["depth_camera"] is None and "depth" not in bundle.arrays


def test_opencv_loads_only_inside_the_wrist_source():
    path = Path(__file__).resolve().parents[1] / "source/maker_arm_lab/real/sources.py"
    body = ast.parse(path.read_text()).body
    names = {a.name for n in body if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in body if isinstance(n, ast.ImportFrom) and n.module}
    assert "cv2" not in names
