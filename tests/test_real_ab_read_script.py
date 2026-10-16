"""MS5 control plan task 11: ``scripts/real_ab_read.py``, the pre-registered adoption rule on
three motion reads, and its one token line on every exit path."""

import copy
import json
import os
import subprocess
import sys

import pytest

from helpers import ROOT, load_script

SCRIPT = ROOT / "scripts" / "real_ab_read.py"
TOKENS = ("AB_OK", "AB_FAIL")
BARS = {"yield_free_max_mm": 0.5, "yield_push_min_mm": 2.0}
TARGETS = {"hold_jitter": 0.002, "stop_sway": 0.005, "free_residual_rms": 0.2, "tip_sway": 0.5}
"""In section 5's units: rad for the motor metrics, N m for the residual, mm at the tip."""


def _run(args, timeout=60):
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


def _metric(joints=None, tip=None):
    return {"joints": joints, "tip": tip, "count": 1, "excluded": 0}


def motion_read(
    *,
    tracking=0.8,
    stop=0.2,
    hold=0.03,
    residual=0.15,
    yield_free=0.1,
    yield_push=3.0,
    tip_sway=0.25,
    hold_joints=None,
    labels=("a.npz", "b.npz"),
):
    """A ``maker-arm-lab/motion-read/2`` read with the pooled tip values given (mm), the
    free residual on every joint (N m), 0.001 rad on every motor joint (``hold_joints`` six
    for ``hold_jitter``), and one traj segment per label."""
    six = [0.001] * 6
    return {
        "schema": "maker-arm-lab/motion-read/2",
        "metrics": {
            "tracking_rms": _metric(six, tracking),
            "stop_sway": _metric(six, stop),
            "hold_jitter": _metric(six if hold_joints is None else hold_joints, hold),
            "free_residual_rms": _metric(None if residual is None else [residual] * 6),
            "yield_free_mm": _metric(six, yield_free),
            "yield_push_mm": _metric(six, yield_push),
            "drag_free": _metric([0.1] * 6),
            "drag_push": _metric([0.4] * 6),
            "tip_sway": _metric(None, tip_sway),
            "flex_share": _metric(None, 0.1),
        },
        "segments": [{"kind": "traj", "label": label} for label in labels]
        + [{"kind": "hold", "label": ""}],
    }


def _write(path, data):
    path.write_text(json.dumps(data))
    return path


def _ab(tmp_path, arm1, arm2, arm3, targets=None):
    targets = {"targets": TARGETS, "bars": BARS} if targets is None else targets
    args = []
    for name, data in (("arm1", arm1), ("arm2", arm2), ("arm3", arm3)):
        args += [f"--{name}", _write(tmp_path / f"{name}.json", data)]
    args += ["--targets", _write(tmp_path / "targets.json", targets)]
    out = tmp_path / "ab.json"
    code, tokens, err = _run([*args, "--out", out])
    assert len(tokens) == 1, (tokens, err)
    return code, tokens[0], out


def _failed(body, layer):
    return {(c["check"], c["metric"]): c["reason"] for c in body["layers"][layer]["checks"]} | {
        "adopted": body["layers"][layer]["adopted"]
    }


def test_the_ab_read_adopts_a_layer_that_meets_every_rule(tmp_path):
    arm1 = motion_read(tracking=0.9, stop=0.35, hold=0.06, tip_sway=0.30, residual=0.18)
    arm2 = motion_read(tracking=0.8, stop=0.20, hold=0.03, tip_sway=0.25, residual=0.15)
    arm3 = motion_read(tracking=0.8, stop=0.20, hold=0.03, tip_sway=0.24, residual=0.10)
    code, token, out = _ab(tmp_path, arm1, arm2, arm3)
    assert code == 0 and token.startswith("AB_OK "), token
    payload = _payload(token)
    assert payload["adopted"] == {"dob": True, "mlp": True}
    assert payload["table"]["tip_sway"] == {"arm1": 0.30, "arm2": 0.25, "arm3": 0.24}
    assert payload["table"]["free_residual_rms"]["arm3"] == pytest.approx(0.10)
    body = json.loads(out.read_text())
    assert body["schema"] == "maker-arm-lab/ab-read/1"
    assert body["layers"]["dob"]["against"] == "arm1"
    assert body["layers"]["mlp"]["against"] == "arm2"
    checks = body["layers"]["mlp"]["checks"]
    assert {c["check"] for c in checks} == {
        "target",
        "free_yield",
        "no_worse",
        "push_yield",
        "lower",
    }
    assert all(c["pass"] and c["reason"] == "met" for c in checks)
    assert "lower" not in {c["check"] for c in body["layers"]["dob"]["checks"]}


def test_the_ab_read_rejects_a_layer_that_breaks_the_push_yield(tmp_path):
    arm1 = motion_read(stop=0.35, hold=0.06)
    arm2 = motion_read(yield_push=1.5)  # the compensation eats the 5 N push
    arm3 = motion_read(residual=0.10)
    code, token, out = _ab(tmp_path, arm1, arm2, arm3)
    assert code == 0, token  # a rejected layer is a result, not a failure
    assert _payload(token)["adopted"] == {"dob": False, "mlp": True}
    body = json.loads(out.read_text())
    dob = _failed(body, "dob")
    assert dob[("push_yield", "yield_push_mm")] == "failed" and not dob["adopted"]
    assert dob[("target", "hold_jitter")] == "met"


def test_a_layer_must_meet_every_target_and_be_no_worse(tmp_path):
    arm1 = motion_read(tracking=0.8, tip_sway=0.25)
    arm2 = motion_read(tracking=0.81, tip_sway=0.6)  # worse tracking, a target missed
    arm3 = motion_read(tracking=0.81, residual=0.15)  # not lower than arm 2's
    code, token, out = _ab(tmp_path, arm1, arm2, arm3)
    assert code == 0 and _payload(token)["adopted"] == {"dob": False, "mlp": False}
    body = json.loads(out.read_text())
    dob, mlp = _failed(body, "dob"), _failed(body, "mlp")
    assert dob[("no_worse", "tracking_rms")] == "failed"
    assert dob[("target", "tip_sway")] == "failed"
    assert dob[("target", "hold_jitter")] == "met"
    assert mlp[("lower", "free_residual_rms")] == "failed"
    assert mlp[("no_worse", "tracking_rms")] == "met"


def test_a_motor_target_reads_the_largest_joint_in_rad(tmp_path):
    # 9 mm of hold jitter at the tip does not matter to a rad target; one joint over it does.
    arm2 = motion_read(hold=9.0)
    arm3 = motion_read(residual=0.10, hold_joints=[0.001] * 5 + [0.003])
    code, token, out = _ab(tmp_path, motion_read(), arm2, arm3)
    assert code == 0 and _payload(token)["adopted"] == {"dob": True, "mlp": False}
    body = json.loads(out.read_text())
    (hold,) = [c for c in body["layers"]["mlp"]["checks"] if c["metric"] == "hold_jitter"]
    assert hold["value"] == pytest.approx(0.003) and hold["ref"] == 0.002
    assert not hold["pass"] and hold["reason"] == "failed"
    assert body["target_table"]["hold_jitter"]["arm3"] == pytest.approx(0.003)
    assert body["table"]["hold_jitter"]["arm2"] == 9.0  # the tip value, for the record


def test_a_missing_value_fails_its_check_and_never_passes(tmp_path):
    arm1 = motion_read(tip_sway=None)
    arm2 = motion_read()
    arm3 = motion_read(residual=None)
    code, token, out = _ab(tmp_path, arm1, arm2, arm3)
    assert code == 0 and _payload(token)["adopted"] == {"dob": False, "mlp": False}
    body = json.loads(out.read_text())
    assert _failed(body, "dob")[("no_worse", "tip_sway")] == "missing"
    mlp = _failed(body, "mlp")
    assert mlp[("lower", "free_residual_rms")] == "missing"
    assert mlp[("target", "free_residual_rms")] == "missing"
    assert body["table"]["tip_sway"]["arm1"] is None


def test_the_free_yield_bar_binds(tmp_path):
    code, token, out = _ab(tmp_path, motion_read(), motion_read(yield_free=0.6), motion_read())
    assert code == 0 and not _payload(token)["adopted"]["dob"]
    assert _failed(json.loads(out.read_text()), "dob")[("free_yield", "yield_free_mm")] == "failed"


@pytest.mark.parametrize(
    "targets, detail",
    [
        ({"targets": {"yield_push_mm": 2.0}, "bars": BARS}, "unknown target metric"),
        ({"targets": {"jitter": 0.1}, "bars": BARS}, "unknown target metric"),
        ({"targets": TARGETS, "bars": {"yield_free_max_mm": 0.5}}, "missing bar"),
        ({"targets": TARGETS, "bars": {**BARS, "extra": 1.0}}, "unknown bar"),
        ({"targets": {}, "bars": BARS}, "non-empty"),
        ({"targets": {"hold_jitter": "x"}, "bars": BARS}, "finite number"),
        ({"targets": TARGETS}, "exactly the keys"),
    ],
)
def test_a_bad_targets_file_is_refused(tmp_path, targets, detail):
    read = motion_read()
    code, token, out = _ab(tmp_path, read, read, read, targets=targets)
    assert code == 1 and token.startswith("AB_FAIL "), token
    payload = _payload(token)
    assert payload["reason"] == "targets" and detail in payload["detail"], payload
    assert not out.exists()


def test_reads_of_different_trajectory_sets_are_refused(tmp_path):
    read = motion_read()
    code, token, _ = _ab(tmp_path, read, read, motion_read(labels=("a.npz", "c.npz")))
    assert code == 1 and _payload(token)["reason"] == "trajectory_set"


def test_a_read_of_another_schema_is_refused(tmp_path):
    other = copy.deepcopy(motion_read())
    other["schema"] = "maker-arm-lab/motion-read/1"
    code, token, _ = _ab(tmp_path, motion_read(), other, motion_read())
    assert code == 1 and _payload(token)["reason"] == "arm2"


def test_an_existing_out_is_never_overwritten(tmp_path):
    read = motion_read()
    (tmp_path / "ab.json").write_text("{}")
    code, token, out = _ab(tmp_path, read, read, read)
    assert code == 1 and _payload(token)["reason"] == "exists"
    assert out.read_text() == "{}"


def test_an_argparse_error_prints_one_token(tmp_path):
    code, tokens, _ = _run(["--arm1", tmp_path / "a.json"])
    assert code == 2 and len(tokens) == 1
    assert _payload(tokens[0])["reason"] == "usage"


def test_the_metric_copies_match_the_motion_read():
    from maker_arm_lab.real import motion

    mod = load_script("real_ab_read")
    assert mod.METRICS == tuple(motion.UNITS)
    assert mod.IN_SCHEMA == motion.SCHEMA
    assert set(mod.TIP_METRICS) == {m for m, u in motion.UNITS.items() if u["tip"] is not None}
    assert set(mod.JOINT_POOL) == set(mod.METRICS) - set(mod.TIP_METRICS)
    assert set(mod.TARGET_POOL) == set(mod.TARGET_UNITS) == set(mod.METRICS) - {"yield_push_mm"}
    for metric, pool in mod.TARGET_POOL.items():
        side = "tip" if pool == "tip" else "joints"
        assert mod.TARGET_UNITS[metric] == motion.UNITS[metric][side], metric
