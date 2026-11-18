"""MS5 design section 4.2: the perception worker, served with a fake segmenter."""

import ast
import io
import json
import math
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from helpers import ROOT, load_script
from maker_arm_lab.core import cameras
from maker_arm_lab.real.camera_model import Pinhole, quat_matrix
from maker_arm_lab.real.perceive import camera_block
from maker_arm_lab.real.rig import RigConfig

WORKER = ROOT / "perception_stack" / "worker.py"
worker = load_script(WORKER)
proto = load_script(ROOT / "perception_stack" / "perception_protocol.py")

RGB_CAM = Pinhole(640, 480, 415.7, 415.7, 320.0, 240.0)
DEPTH_CAM = Pinhole(320, 240, 207.85, 207.85, 160.0, 120.0)
POS = np.array([0.20, 0.0, 0.80])
QUAT = tuple(cameras.OVERHEAD_CAM_ROT)
ROT = quat_matrix(QUAT)
RIG = {
    "tray_centre": [0.20, 0.0],
    "tray_interior": [0.13, 0.20],
    "tray_floor_z": 0.0,
    "desk_z": 0.0,
    "box_interior": [0.09, 0.09, 0.04],
    "tray_lean_deg": 30.0,
}
PARTS = [(0.20, -0.04, 0.3, 0.032, 0.016), (0.20, 0.04, -0.4, 0.032, 0.016)]
BOX = (0.15, 0.20, 0.3, 0.095, 0.095)  # 0.09 interior plus two 5 mm walls
RIM_Z = 0.04


class FakeSegmenter:
    """One mask per prompt: the bright pixels in a window around the point."""

    def masks(self, rgb, points):
        out = np.zeros((len(points), *rgb.shape[:2]), bool)
        for i, (x, y) in enumerate(points):
            r, c = int(y), int(x)
            window = np.zeros(rgb.shape[:2], bool)
            window[max(r - 18, 0) : r + 18, max(c - 18, 0) : c + 18] = True
            out[i] = window & (rgb[..., 1] > 200)
        return out


def footprint(cam, obj, plane_z):
    """The pixels of ``cam`` where the ray meets the plane inside the (cx, cy, yaw, l, w) box."""
    cx, cy, yaw, length, width = obj
    v, u = np.mgrid[0 : cam.height, 0 : cam.width] + 0.5
    ray = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones(u.shape)], -1) @ ROT.T
    t = (plane_z - POS[2]) / ray[..., 2]
    dx, dy = POS[0] + ray[..., 0] * t - cx, POS[1] + ray[..., 1] * t - cy
    c, s = math.cos(yaw), math.sin(yaw)
    return (np.abs(dx * c + dy * s) <= length / 2) & (np.abs(dy * c - dx * s) <= width / 2)


def frame(with_box=True, wall=0.005):
    rgb = np.full((480, 640, 3), 120, np.uint8)
    depth = np.full((240, 320), 800, np.uint16)
    for part in PARTS:
        rgb[footprint(RGB_CAM, part, 0.0096)] = 255
        depth[footprint(DEPTH_CAM, part, 0.0096)] = 790
    if with_box:
        box = (*BOX[:3], 0.09 + 2 * wall, 0.09 + 2 * wall)
        rgb[footprint(RGB_CAM, box, RIM_Z)] = 30
        depth[footprint(DEPTH_CAM, box, RIM_Z)] = 760
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=95)
    return np.frombuffer(buf.getvalue(), np.uint8), depth


def call(kind, jpeg, depth, rig=RIG):
    cam = camera_block(RGB_CAM, POS, QUAT)
    dcam = None if depth is None else camera_block(DEPTH_CAM, POS, QUAT)
    arrays = {"rgb_jpeg": jpeg} | ({} if depth is None else {"depth": depth})
    header = {"type": kind, "camera": cam, "depth_camera": dcam}
    return worker.Worker(FakeSegmenter(), rig).handle(header, arrays)


def test_parts_returns_one_valid_detection_per_part():
    jpeg, depth = frame()
    reply, _ = call("parts", jpeg, depth)
    assert reply["type"] == "parts" and len(reply["detections"]) == 2
    for d in reply["detections"]:
        proto.validate_detection(d)
    found = sorted((d["centre"][:2] for d in reply["detections"]), key=lambda c: c[1])
    for got, part in zip(found, sorted(PARTS, key=lambda p: p[1]), strict=True):
        assert math.dist(got, part[:2]) < 0.006
    scores = [d["score"] for d in reply["detections"]]
    assert scores == sorted(scores, reverse=True)


def test_boxes_returns_the_boxes_in_the_bands():
    jpeg, depth = frame()
    for depth_in in (depth, None):
        reply, _ = call("boxes", jpeg, depth_in)
        assert reply["type"] == "boxes" and len(reply["poses"]) == 1
        assert math.dist(reply["poses"][0][:2], BOX[:2]) < 0.006


def test_the_rig_wall_sets_the_box_size_the_worker_wants():
    jpeg, depth = frame(wall=0.0015)
    reply, _ = call("boxes", jpeg, depth, RIG | {"box_wall": 0.0015})
    assert len(reply["poses"]) == 1
    jpeg, depth = frame()  # a 0.1 m outer box against walls of 0.02 m: 0.13 m wanted
    assert call("boxes", jpeg, depth, RIG | {"box_wall": 0.02})[0]["poses"] == []


def test_parts_without_depth_replies_an_error_not_a_crash():
    reply, _ = call("parts", frame()[0], None)
    assert reply["type"] == "error"


def test_hello_and_shutdown_answer_and_an_unsupported_call_replies_an_error():
    w = worker.Worker(FakeSegmenter(), RIG)
    reply, _ = w.handle({"type": "hello", "protocol": proto.PROTOCOL_VERSION}, {})
    assert reply["type"] == "hello_ack" and reply["worker_id"]
    assert w.handle({"type": "shutdown"}, {})[0]["type"] == "shutdown_ack"
    for kind in ("marker", "wrist", "lift"):
        reply, _ = w.handle({"type": kind}, {})
        assert reply["type"] == "error" and reply["error"] == "unsupported"


def test_the_worker_module_imports_no_torch_at_top_level():
    tree = ast.parse(WORKER.read_text())
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    assert names and not [n for n in names if n.split(".")[0] in ("torch", "sam2")]
    assert "maker_arm_lab" not in WORKER.read_text()


def test_a_usage_error_prints_one_worker_fail_line():
    out = subprocess.run(
        [sys.executable, str(WORKER), "--no_such_flag"], capture_output=True, text=True, check=False
    )
    lines = [x for x in out.stdout.splitlines() if x.startswith("WORKER_")]
    assert out.returncode != 0 and len(lines) == 1 and lines[0].startswith("WORKER_FAIL ")
    assert "reason" in json.loads(lines[0].split(" ", 1)[1])


def test_the_rig_config_carries_the_box_wall_and_loads_an_older_rig_without_it(tmp_path):
    path = tmp_path / "rig.json"
    path.write_text(RigConfig(box_wall=0.0015).to_json())
    assert RigConfig.from_json(path).box_wall == 0.0015
    old = json.loads(RigConfig().to_json())
    del old["box_wall"]
    path.write_text(json.dumps(old))
    assert RigConfig.from_json(path).box_wall == pytest.approx(0.005)
