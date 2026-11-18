"""MS5 sensing: the tag scripts print one fail token on every exit path (no OpenCV needed)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

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
