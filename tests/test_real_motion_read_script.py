"""MS5 control plan task 9: ``scripts/real_motion_read.py``, its output files and its one
token line on every exit path."""

import json
import math
import os
import subprocess
import sys

import numpy as np

from helpers import ROOT
from test_real_motion import POSE, still, tag_line, write_log

SCRIPT = ROOT / "scripts" / "real_motion_read.py"
TOKENS = ("MOTION_READ_OK", "MOTION_READ_FAIL")
METRICS = (
    "tracking_rms",
    "stop_sway",
    "hold_jitter",
    "free_residual_rms",
    "yield_mm",
    "drag_resistance",
    "tip_sway",
    "flex_share",
)


def _run(args, timeout=300):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "AMENT_PREFIX_PATH")}
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )
    tokens = [line for line in out.stdout.splitlines() if line.startswith(TOKENS)]
    return out.returncode, tokens, out.stderr


def _payload(token):
    return json.loads(token.split(" ", 1)[1])


def _session(tmp_path):
    """A traj, its stop and two holds; tag frames every 7th tick (~29 Hz) seen by a camera at the identity
    pose, so the tag's camera position is its world position."""
    t, q = still(8.0)
    move = t < 2.0
    q[move, 0] += np.linspace(-0.1, 0.0, int(move.sum()))
    q[~move, 0] += 0.001 * np.exp(-3.0 * (t[~move] - 2.0)) * np.sin(22.0 * (t[~move] - 2.0))
    segs = [(0.0, 2.0, "traj", "a.npz"), (2.0, 5.0, "hold", ""), (5.0, 8.0, "hold", "")]
    log = write_log(tmp_path / "s.h5", t, q, segments=segs)
    from maker_arm_lab.core.global_ik import GlobalIK
    from maker_arm_lab.real import motion

    pts = motion.points(GlobalIK(), q, [0.0, 0.0, 0.01])
    lines = [tag_line(t[k], cam_pos=pts[k], pos=pts[k]) for k in range(10, t.shape[0], 7)]
    tag = tmp_path / "tag.jsonl"
    tag.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return log, tag


def _rig(tmp_path, **fields):
    path = tmp_path / "rig.json"
    path.write_text(json.dumps({"rt": None, **fields}))
    return path


def test_the_read_usage_error_prints_one_fail_line():
    code, tokens, _ = _run(["--no_such_flag"])
    assert code == 2 and len(tokens) == 1 and tokens[0].startswith("MOTION_READ_FAIL")
    assert _payload(tokens[0])["reason"] == "usage"


def test_a_tag_log_needs_its_latency(tmp_path):
    log, tag = _session(tmp_path)
    code, tokens, _ = _run(["--log", log, "--tag_log", tag, "--out", tmp_path / "o.json"])
    assert code == 2 and len(tokens) == 1 and tokens[0].startswith("MOTION_READ_FAIL")
    assert _payload(tokens[0])["reason"] == "usage"


def test_a_missing_log_fails_with_one_line(tmp_path):
    code, tokens, _ = _run(["--log", tmp_path / "none.h5", "--out", tmp_path / "o.json"])
    assert code == 1 and len(tokens) == 1 and _payload(tokens[0])["reason"] == "log"


def test_the_full_read_writes_every_metric(tmp_path):
    log, tag = _session(tmp_path)
    rig = _rig(
        tmp_path,
        tag_offset=[0.0, 0.0, 0.01],
        overhead_pos=[0.0, 0.0, 0.0],
        overhead_quat=[1.0, 0.0, 0.0, 0.0],
    )
    out, pairs = tmp_path / "o.json", tmp_path / "pairs.json"
    args = ["--log", log, "--tag_log", tag, "--latency_ms", 91, "--rig", rig]
    code, tokens, err = _run([*args, "--pairs_out", pairs, "--out", out])
    assert code == 0 and len(tokens) == 1 and tokens[0].startswith("MOTION_READ_OK"), err
    body = json.loads(out.read_text())
    assert body["schema"] == "maker-arm-lab/motion-read/1"
    assert set(body["metrics"]) == set(METRICS)
    for name in METRICS:
        assert set(body["metrics"][name]) == {"joints", "tip", "count"}
        assert set(body["units"][name]) == {"joints", "tip"}
    assert body["tip_side"] == {"available": True, "reason": None, "tag_world": "rig_overhead"}
    # the tag rides on FK through the tag offset: a rigid arm
    assert body["metrics"]["flex_share"]["tip"] < 1e-3
    assert body["metrics"]["tip_sway"]["count"] == 1
    assert body["metrics"]["hold_jitter"]["count"] == 2
    kinds = [s["kind"] for s in body["segments"]]
    assert kinds == ["traj", "hold", "stop", "hold"]
    assert len(json.loads(pairs.read_text())["pairs"]) == 2
    payload = _payload(tokens[0])
    assert payload["out"] == str(out) and set(payload["metrics"]) == set(METRICS)
    # an output is never overwritten
    code, tokens, _ = _run([*args, "--out", out])
    assert code == 1 and _payload(tokens[0])["reason"] == "exists"


def test_without_a_tag_offset_the_read_writes_the_pairs_and_the_motor_side(tmp_path):
    log, tag = _session(tmp_path)
    out, pairs = tmp_path / "o.json", tmp_path / "pairs.json"
    args = ["--log", log, "--tag_log", tag, "--latency_ms", 91, "--rig", _rig(tmp_path)]
    code, tokens, err = _run([*args, "--pairs_out", pairs, "--out", out])
    assert code == 0 and len(tokens) == 1 and tokens[0].startswith("MOTION_READ_OK"), err
    body = json.loads(out.read_text())
    assert body["tip_side"]["available"] is False and "tag_offset" in body["tip_side"]["reason"]
    assert body["metrics"]["flex_share"] == {"joints": None, "tip": None, "count": 0}
    assert all(math.isfinite(v) for v in body["metrics"]["stop_sway"]["joints"])
    held = json.loads(pairs.read_text())["pairs"]
    assert len(held) == 2 and np.allclose(held[0]["q"], POSE, atol=1e-3)


def test_the_motor_side_reads_with_no_tag_log(tmp_path):
    log, _ = _session(tmp_path)
    out = tmp_path / "o.json"
    code, tokens, err = _run(["--log", log, "--out", out])
    assert code == 0 and len(tokens) == 1 and tokens[0].startswith("MOTION_READ_OK"), err
    body = json.loads(out.read_text())
    assert body["metrics"]["tracking_rms"]["count"] == 1
    assert body["pairs"] is None
