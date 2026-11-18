"""MS5 sensing plan task 11: the wrist camera's undistortion, its rig field, the iPad's
checkerboard and ``scripts/real_wrist_intrinsics.py``'s one token line."""

import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from helpers import ROOT, load_script
from maker_arm_lab.real.camera_model import Pinhole
from maker_arm_lab.real.frames import Arrival, JpegFrame
from maker_arm_lab.real.layouts import IPAD_AIR_11, checker_png
from maker_arm_lab.real.rig import RigConfig
from maker_arm_lab.real.sources import UvcWrist, WristFrames

SCRIPT = ROOT / "scripts" / "real_wrist_intrinsics.py"
K = (500.0, 0, 319.5, 0, 500.0, 239.5, 0, 0, 1)
DIST = (-0.25, 0.08, 0.001, -0.001, 0.0)


def test_the_screen_is_one_to_one():
    assert IPAD_AIR_11.mm_per_px == pytest.approx(25.4 / 264.0)
    assert IPAD_AIR_11.to_px((0.0, 0.0)) == pytest.approx((1180.0, 820.0))
    assert IPAD_AIR_11.from_px(*IPAD_AIR_11.to_px((0.05, -0.02))) == pytest.approx((0.05, -0.02))


def test_the_checkerboard_squares_are_208_px_and_the_board_fits_the_screen():
    img = np.asarray(checker_png().convert("L"))
    assert img.shape == (1640, 2360)
    left, top = (2360 - 10 * 208) // 2, (1640 - 7 * 208) // 2  # 9x6 inner corners: 10x7 squares
    assert img[top + 100, left + 100] < 64 and img[top + 100, left + 208 + 100] > 192
    assert img[top + 207, left + 100] < 64 and img[top + 208, left + 100] > 192
    dark = np.argwhere(img < 64)
    assert dark[:, 1].min() >= left and dark[:, 1].max() < 2360 - left
    assert dark[:, 0].min() >= top and dark[:, 0].max() < 1640


def test_the_rig_round_trips_wrist_dist(tmp_path):
    rig = RigConfig(wrist_k=K, wrist_size=(640, 480), wrist_dist=DIST)
    path = tmp_path / "rig.json"
    path.write_text(rig.to_json())
    assert RigConfig.from_json(path).wrist_dist == DIST
    assert RigConfig().wrist_dist is None


def _dot_rows(image):
    """The dots' intensity-weighted centroids, one array per grid row."""
    cv2 = pytest.importorskip("cv2")
    gray = image.max(axis=2).astype(np.float64)
    count, labels = cv2.connectedComponents((gray > 20).astype(np.uint8))
    ys, xs = np.mgrid[: gray.shape[0], : gray.shape[1]]
    pts = []
    for i in range(1, count):
        w = gray * (labels == i)
        pts.append(((w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()))
    pts = np.array(pts)
    return [pts[np.abs(pts[:, 1] - y) < 25] for y in sorted({round(p[1], -1) for p in pts})]


def test_undistortion_straightens_a_synthetically_distorted_grid():
    cv2 = pytest.importorskip("cv2")
    camera = Pinhole.from_ros_k(640, 480, K)
    k = np.array(K).reshape(3, 3)
    xs, ys = np.meshgrid(np.linspace(-0.55, 0.55, 9), np.linspace(-0.4, 0.4, 7))
    ideal = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=1)
    zero = np.zeros(3)
    distorted, _ = cv2.projectPoints(ideal, zero, zero, k, np.array(DIST))
    image = np.zeros((480, 640, 3), np.uint8)
    for u, v in distorted.reshape(-1, 2):
        cv2.circle(image, (round(u * 16), round(v * 16)), 4 * 16, (255, 255, 255), -1, shift=4)

    def worst_residual(src):
        src.on_frame(image, 1)
        out = src.frames.get().view
        worst = 0.0
        for row in _dot_rows(out):
            a, b = np.polyfit(row[:, 0], row[:, 1], 1)
            worst = max(worst, np.abs(row[:, 1] - (a * row[:, 0] + b)).max())
        return worst

    assert worst_residual(WristFrames(camera, None)) > 3.0
    assert worst_residual(WristFrames(camera, None, DIST)) < 0.3


def test_without_dist_there_are_no_maps_and_the_frame_is_untouched():
    src = WristFrames(Pinhole.from_ros_k(640, 480, K), None)
    assert src._maps is None


def test_uvc_wrist_hands_dist_to_the_base_class(monkeypatch):
    cv2 = pytest.importorskip("cv2")

    class Cap:
        def __init__(self, *a):
            pass

        def isOpened(self):
            return True

        def set(self, *a):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", Cap)
    cam = Pinhole.from_ros_k(640, 480, K)
    assert UvcWrist("/dev/null", cam, None, DIST)._maps is not None
    assert UvcWrist("/dev/null", cam, None)._maps is None


def _run(args):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "AMENT_PREFIX_PATH")}
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    return out.returncode, [x for x in out.stdout.splitlines() if x.startswith("WRIST_K_")]


def test_the_intrinsics_script_has_no_device_argument():
    code, tokens = _run(["--device", "/dev/video0"])
    assert code == 2 and len(tokens) == 1 and tokens[0].startswith("WRIST_K_FAIL")
    assert json.loads(tokens[0].split(" ", 1)[1])["reason"] == "usage"


def test_the_intrinsics_usage_error_prints_one_fail_line():
    code, tokens = _run([])
    assert code == 2 and len(tokens) == 1 and tokens[0].startswith("WRIST_K_FAIL")


def _board_object_points():
    obj = np.zeros((54, 3))
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.02001
    return obj - obj.mean(axis=0)


def test_calibrate_views_recovers_a_known_camera():
    cv2 = pytest.importorskip("cv2")
    script = load_script(SCRIPT)
    k, dist = np.array(K).reshape(3, 3), np.array((-0.2, 0.05, 0, 0, 0))
    rng, obj, views = np.random.default_rng(7), _board_object_points(), []
    while len(views) < 14:
        rvec = np.radians(rng.uniform(-30, 30, 3))
        tvec = np.array(
            [rng.uniform(-0.03, 0.03), rng.uniform(-0.02, 0.02), rng.uniform(0.15, 0.3)]
        )
        pts = cv2.projectPoints(obj, rvec, tvec, k, dist)[0].reshape(-1, 2)
        if pts.min() >= 5 and pts[:, 0].max() < 635 and pts[:, 1].max() < 475:
            views.append((pts + rng.normal(0, 0.1, pts.shape)).astype(np.float32).reshape(-1, 1, 2))
    got_k, got_dist, rms = script.calibrate_views(views, (9, 6), 0.02001, (640, 480))
    assert len(got_dist) == 5 and rms < 0.3
    for got, want in zip((got_k[0], got_k[4], got_k[2], got_k[5]), (500, 500, 319.5, 239.5)):
        assert got == pytest.approx(want, abs=1.0)


def test_a_rendered_checker_frame_is_detected():
    cv2 = pytest.importorskip("cv2")
    board = np.asarray(checker_png().convert("L"))
    src = np.float32([[140, 92], [2220, 92], [2220, 1548], [140, 1548]])
    dst = np.float32([[110, 90], [530, 130], [510, 380], [130, 410]])
    frame = cv2.warpPerspective(
        board,
        cv2.getPerspectiveTransform(src, dst),
        (640, 480),
        flags=cv2.INTER_AREA,
        borderValue=255,
    )
    found, corners = cv2.findChessboardCorners(frame, (9, 6))
    assert found and len(corners) == 54


def test_a_wrong_size_stream_ends_in_a_fail_that_names_the_size_and_calibrates_nothing():
    cv2 = pytest.importorskip("cv2")
    script = load_script(SCRIPT)
    w, h = 640, 480
    src = WristFrames(
        Pinhole.from_ros_k(w, h, (1, 0, w / 2 - 0.5, 0, 1, h / 2 - 0.5, 0, 0, 1)), None
    )
    buf = io.BytesIO()
    Image.fromarray(np.zeros((240, 320, 3), np.uint8)).save(buf, "JPEG")
    src.on_jpeg(JpegFrame(Arrival(1, 2, 3), buf.getvalue(), "wrist_camera"))

    class Sub:
        error, closed = None, 0

        def close(self):
            self.closed += 1

    sub = Sub()
    args = script.build_parser().parse_args(
        ["--rig", "r", "--rig_out", "o", "--frames", "5", "--timeout_s", "0.2"]
    )
    kept, reason = script.wait_for_views(src, sub, cv2, (9, 6), args)
    assert kept == [] and "320x240" in reason and sub.closed == 1
