"""MS5 sensing: the tag scripts print one fail token on every exit path (no OpenCV needed)."""

import json
import math
import subprocess
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from helpers import load_script
from maker_arm_lab.real.rig import RigConfig

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _run(script, *args):
    out = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return out.returncode, out.stdout.strip().splitlines()


@pytest.mark.parametrize(
    ("script", "token"),
    [
        ("real_calibrate_overhead.py", "CALIB_FAIL"),
        ("real_screen_pose.py", "SCREEN_FAIL"),
        ("real_register_base.py", "REGISTER_FAIL"),
        ("real_tag_log.py", "TAGLOG_FAIL"),
    ],
)
def test_a_usage_error_prints_the_fail_token(script, token):
    code, lines = _run(script, "--bogus")
    assert code == 2 and len(lines) == 1 and lines[0].startswith(token + " ")


def test_register_base_refuses_seven_pairs(tmp_path):
    (tmp_path / "rig.json").write_text(RigConfig().to_json())
    pairs = [{"q": [0.0] * 6, "tag_cam": [0.0, 0.0, 0.6]}] * 7
    (tmp_path / "pairs.json").write_text(json.dumps(pairs))
    code, lines = _run(
        "real_register_base.py",
        "--rig", str(tmp_path / "rig.json"),
        "--pairs", str(tmp_path / "pairs.json"),
        "--rig_out", str(tmp_path / "out.json"),
        "--report", str(tmp_path / "report.json"),
    )  # fmt: skip
    assert code == 1 and len(lines) == 1 and lines[0].startswith("REGISTER_FAIL ")
    assert not (tmp_path / "out.json").exists()


def _screen_session(monkeypatch, info_size, screen):
    """A fake Conduit stream of ``info_size`` and fake tag reads that put the screen at
    ``screen`` (a base-frame ``Pose``); no OpenCV and no ROS."""
    import maker_arm_lab.real
    from maker_arm_lab.real import tags
    from maker_arm_lab.real.camera_model import Pose

    class Sub:
        error = None

        def __init__(self, on_jpeg=None, on_info=None, **kw):
            self.on_jpeg, self.on_info, self.stop = on_jpeg, on_info, threading.Event()

        def start(self):
            w, h = info_size
            self.on_info(
                types.SimpleNamespace(
                    width=w, height=h, k=(1450, 0, w / 2, 0, 1450, h / 2, 0, 0, 1)
                )
            )

            def feed():
                while not self.stop.wait(0.005):
                    self.on_jpeg(types.SimpleNamespace(data=b""))

            threading.Thread(target=feed, daemon=True).start()
            return self

        def close(self):
            self.stop.set()

    def screen_in_camera(camera, corners, board):
        cam = Pose.from_pos_quat(RigConfig().overhead_pos, RigConfig().overhead_quat)
        return Pose(cam.rot.T @ screen.rot, cam.rot.T @ (screen.pos - cam.pos)), 0.0

    fake = types.ModuleType("maker_arm_lab.real.ros")
    fake.ConduitSubscriber = Sub
    monkeypatch.setitem(sys.modules, "maker_arm_lab.real.ros", fake)
    monkeypatch.setattr(maker_arm_lab.real, "ros", fake, raising=False)
    monkeypatch.setattr(tags, "detect_in_jpeg", lambda camera, data: dict.fromkeys(range(6)))
    monkeypatch.setattr(tags, "screen_in_camera", screen_in_camera)


def _screen_pose(tmp_path, rig, *extra):
    script = load_script("real_screen_pose")
    (tmp_path / "rig.json").write_text(rig.to_json())
    args = ["--rig", str(tmp_path / "rig.json"), "--frames", "3", "--timeout_s", "5", *extra]
    return script._run(script.build_parser().parse_args(args))


def _moved(rig, dx_m, yaw_deg):
    from maker_arm_lab.real.camera_model import Pose
    from maker_arm_lab.real.layouts import nominal_screen_pose

    nom, th = nominal_screen_pose(rig), math.radians(yaw_deg)
    rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    return Pose(rz @ nom.rot, nom.pos + [dx_m, 0.0, 0.0])


def test_screen_pose_needs_a_calibrated_rig(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "maker_arm_lab.real.ros", types.ModuleType("ros"))
    token, code = _screen_pose(tmp_path, RigConfig(), "--out", str(tmp_path / "s.json"))
    assert code == 1 and token.startswith("SCREEN_FAIL ") and "calibrate first" in token


def test_screen_pose_refuses_a_stream_of_another_size(tmp_path, monkeypatch):
    rig = RigConfig(overhead_size=(1920, 1440))
    _screen_session(monkeypatch, (1440, 1920), _moved(rig, 0.0, 0.0))  # the phone turned
    token, code = _screen_pose(tmp_path, rig, "--out", str(tmp_path / "s.json"))
    assert code == 1 and token.startswith("SCREEN_FAIL ")
    assert "1440x1920" in token and "1920x1440" in token
    assert not (tmp_path / "s.json").exists()


def test_screen_pose_reports_its_offset_and_how_far_it_moved(tmp_path, monkeypatch):
    rig = RigConfig(overhead_size=(1920, 1440))
    first, out = tmp_path / "first.json", tmp_path / "again.json"
    _screen_session(monkeypatch, (1920, 1440), _moved(rig, 0.002, 1.0))
    token, code = _screen_pose(tmp_path, rig, "--out", str(first))
    assert code == 0 and token.startswith("SCREEN_OK "), token
    got = json.loads(token.split(" ", 1)[1])
    assert got["offset_mm"] == pytest.approx(2.0, abs=1e-6)
    assert got["yaw_deg"] == pytest.approx(1.0, abs=1e-6)
    assert "moved_mm" not in got
    token, code = _screen_pose(tmp_path, rig, "--out", str(first))  # --out exists
    assert code == 1 and token.startswith("SCREEN_FAIL ") and "exists" in token
    _screen_session(monkeypatch, (1920, 1440), _moved(rig, 0.0026, 1.5))
    token, code = _screen_pose(tmp_path, rig, "--out", str(out), "--compare", str(first))
    got = json.loads(token.split(" ", 1)[1])
    assert code == 0 and got["moved_mm"] == pytest.approx(0.6, abs=1e-6)
    assert got["moved_deg"] == pytest.approx(0.5, abs=1e-6)
