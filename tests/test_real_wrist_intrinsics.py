"""MS5 sensing plan task 11: the wrist camera's undistortion, its rig field, the iPad's
checkerboard and ``scripts/real_wrist_intrinsics.py``'s one token line."""

import io
import json
import math
import os
import subprocess
import sys
import time
import types

import numpy as np
import pytest
from PIL import Image

import maker_arm_lab.real
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
    bar_row = img[(top + 7 * 208 + 1640) // 2]  # the scale bar: 100 mm at 264 ppi
    assert (bar_row < 64).sum() == pytest.approx(100 / 25.4 * 264, abs=2)


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


def _synthetic_views(n, bad_noise=None, seed=7):
    """n views through K (500, ..., 319.5, 239.5) with 0.1 px noise, plus one 5 px view."""
    cv2 = pytest.importorskip("cv2")
    k, dist = np.array(K).reshape(3, 3), np.array((-0.2, 0.05, 0, 0, 0))
    rng, obj, views = np.random.default_rng(seed), _board_object_points(), []
    while len(views) < n + (bad_noise is not None):
        noise = bad_noise if bad_noise is not None and len(views) == n else 0.1
        rvec = np.radians(rng.uniform(-30, 30, 3))
        tvec = np.array(
            [rng.uniform(-0.03, 0.03), rng.uniform(-0.02, 0.02), rng.uniform(0.15, 0.3)]
        )
        pts = cv2.projectPoints(obj, rvec, tvec, k, dist)[0].reshape(-1, 2)
        if pts.min() >= 5 and pts[:, 0].max() < 635 and pts[:, 1].max() < 475:
            views.append(
                (pts + rng.normal(0, noise, pts.shape)).astype(np.float32).reshape(-1, 1, 2)
            )
    return views


def _assert_recovers(c):
    for got, want in zip((c.k[0], c.k[4], c.k[2], c.k[5]), (500, 500, 319.5, 239.5)):
        assert got == pytest.approx(want, abs=1.0)


def test_calibrate_views_recovers_a_known_camera():
    script = load_script(SCRIPT)
    c = script.calibrate_views(_synthetic_views(14), (9, 6), 0.02001, (640, 480))
    assert len(c.dist) == 5 and c.rms_px < 0.3 and c.dropped == ()
    _assert_recovers(c)


def test_one_bad_view_is_dropped_and_the_camera_still_recovers():
    script = load_script(SCRIPT)
    c = script.calibrate_views(_synthetic_views(16, 5.0), (9, 6), 0.02001, (640, 480))
    assert c.dropped == (16,) and len(c.view_rms_px) == 17 and c.rms_px < 0.3
    _assert_recovers(c)


def test_the_std_devs_are_small_for_a_good_set():
    script = load_script(SCRIPT)
    c = script.calibrate_views(_synthetic_views(14), (9, 6), 0.02001, (640, 480))
    assert set(c.std_px) == {"fx", "fy", "cx", "cy"}
    assert all(0 < v < 1.0 for v in c.std_px.values())


def test_still_accepts_a_still_board_and_rejects_a_moving_one():
    script = load_script(SCRIPT)
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 600, (54, 1, 2)).astype(np.float32)
    hist = [base + rng.normal(0, 0.05, base.shape).astype(np.float32) for _ in range(3)]
    assert script.still(hist, 0.5) is True
    moved = [*hist[:2], hist[2] + np.float32([1.0, 0.0])]
    assert script.still(moved, 0.5) is False
    assert script.still(hist[:2], 0.5) is False


def test_an_existing_save_dir_fails_before_the_session(tmp_path, monkeypatch):
    script = load_script(SCRIPT)
    monkeypatch.setitem(sys.modules, "maker_arm_lab.real.ros", types.ModuleType("ros"))
    (tmp_path / "views").mkdir()
    args = script.build_parser().parse_args(
        [
            "--rig",
            "r",
            "--rig_out",
            str(tmp_path / "out.json"),
            "--save_dir",
            str(tmp_path / "views"),
        ]
    )
    token, code = script._run(args)
    assert token.startswith("WRIST_K_FAIL ") and "save_dir" in token and code == 1


def test_views_json_lists_every_kept_view(tmp_path):
    script = load_script(SCRIPT)
    rgbs = [np.full((480, 640, 3), 10 * i, np.uint8) for i in range(3)]
    corners = [np.full((54, 1, 2), float(i), np.float32) for i in range(3)]
    calib = script.Calib((1,) * 9, (0,) * 5, 0.2, (0.1, 0.1, 2.0), {}, (2,))
    save = tmp_path / "views"
    script.save_views(save, rgbs, corners, calib)
    assert sorted(p.name for p in save.glob("*.png")) == [f"view_0{i}.png" for i in range(3)]
    entries = json.loads((save / "views.json").read_text())
    assert [e["file"] for e in entries] == [f"view_0{i}.png" for i in range(3)]
    assert [e.get("dropped", False) for e in entries] == [False, False, True]
    assert entries[2]["rms_px"] == 2.0 and len(entries[1]["corners"]) == 54


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


class Sub:
    def __init__(self, error=None):
        self.error, self.closed = error, 0

    def close(self):
        self.closed += 1


def _source(w=640, h=480):
    return WristFrames(
        Pinhole.from_ros_k(w, h, (1, 0, w / 2 - 0.5, 0, 1, h / 2 - 0.5, 0, 0, 1)), None
    )


def _args(script, timeout_s):
    return script.build_parser().parse_args(
        ["--rig", "r", "--rig_out", "o", "--frames", "5", "--timeout_s", str(timeout_s)]
    )


def test_a_wrong_size_stream_ends_early_in_a_fail_that_names_the_size():
    cv2 = pytest.importorskip("cv2")
    script = load_script(SCRIPT)
    src = _source()
    buf = io.BytesIO()
    Image.fromarray(np.zeros((240, 320, 3), np.uint8)).save(buf, "JPEG")
    for i in range(script.DROP_FAIL):
        src.on_jpeg(JpegFrame(Arrival(1, 2, 3 + i), buf.getvalue(), "wrist_camera"))
    sub, t0 = Sub(), time.monotonic()
    kept, reason = script.wait_for_views(src, sub, cv2, (9, 6), _args(script, 30))
    assert kept == [] and "320x240" in reason and sub.closed == 1
    assert time.monotonic() - t0 < 5


def test_a_stream_with_no_frame_ends_on_the_silent_rule(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    script = load_script(SCRIPT)
    monkeypatch.setattr(script, "SILENT_FAIL_S", 0.3)
    sub, t0 = Sub("qos mismatch"), time.monotonic()
    kept, reason = script.wait_for_views(
        _source(), sub, cv2, (9, 6), _args(script, 30), "/wrist_camera/image_raw/compressed"
    )
    assert kept == [] and sub.closed == 1 and time.monotonic() - t0 < 5
    assert "no frame arrived on /wrist_camera/image_raw/compressed" in reason
    assert "qos mismatch" in reason


def test_ctrl_c_ends_in_one_fail_token(monkeypatch, capsys):
    script = load_script(SCRIPT)

    def interrupted(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(script, "_run", interrupted)
    code = script.main(["--rig", "r", "--rig_out", "o"])
    lines = capsys.readouterr().out.splitlines()
    assert code != 0 and len(lines) == 1 and lines[0].startswith("WRIST_K_FAIL ")
    assert json.loads(lines[0].split(" ", 1)[1])["reason"] == "interrupted"


def test_a_missing_output_folder_fails_before_the_session(tmp_path, monkeypatch):
    script = load_script(SCRIPT)
    monkeypatch.setitem(sys.modules, "maker_arm_lab.real.ros", types.ModuleType("ros"))
    args = script.build_parser().parse_args(
        ["--rig", "r", "--rig_out", str(tmp_path / "nope" / "out.json")]
    )
    token, code = script._run(args)
    assert token.startswith("WRIST_K_FAIL ") and "rig_out" in token and code == 1


def test_a_640x480_camera_at_70_degrees_covers_three_quarters_of_the_wrist_view():
    from maker_arm_lab.real.camera_model import wrist_warp

    fx = 320 / math.tan(math.radians(35))
    cam = Pinhole.from_ros_k(640, 480, (fx, 0, 319.5, 0, fx, 239.5, 0, 0, 1))
    assert len(cam.fov_deg()) == 2
    assert wrist_warp(cam).coverage == pytest.approx(0.75, abs=0.02)


def test_run_on_a_wrong_size_stream_returns_one_fail_line_and_writes_no_rig(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    script = load_script(SCRIPT)
    buf = io.BytesIO()
    Image.fromarray(np.zeros((240, 320, 3), np.uint8)).save(buf, "JPEG")

    class Subscriber:
        error, failures = None, 0

        def __init__(self, on_wrist=None, **kw):
            self.on_wrist = on_wrist

        def start(self):
            self.on_wrist(JpegFrame(Arrival(1, 2, 3), buf.getvalue(), "wrist_camera"))
            return self

        def close(self):
            pass

    fake = types.ModuleType("maker_arm_lab.real.ros")
    fake.WRIST_TOPIC, fake.ConduitSubscriber = "/wrist_camera/image_raw/compressed", Subscriber
    monkeypatch.setitem(sys.modules, "maker_arm_lab.real.ros", fake)
    monkeypatch.setattr(maker_arm_lab.real, "ros", fake, raising=False)
    rig, out = tmp_path / "rig.json", tmp_path / "out.json"
    rig.write_text(RigConfig().to_json())
    args = script.build_parser().parse_args(
        ["--rig", str(rig), "--rig_out", str(out), "--timeout_s", "0.2"]
    )
    token, code = script._run(args)
    assert token.startswith("WRIST_K_FAIL ") and "320x240" in token and "\n" not in token
    assert code != 0 and not out.exists()
