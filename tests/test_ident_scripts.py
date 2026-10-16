"""MS5 control plan task 7: ``scripts/real_excitation.py`` and ``scripts/real_ident.py``, their
files and their one token line on every exit path."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from helpers import ROOT, load_script
from maker_arm_lab.core import dynamics_model as dm
from maker_arm_lab.core import profile as core_profile
from maker_arm_lab.ident import excitation
from maker_arm_lab.real import session
from test_ident import _true_model, synthetic_log

EXCITE = ROOT / "scripts" / "real_excitation.py"
IDENT = ROOT / "scripts" / "real_ident.py"
TOKENS = ("EXCITE_OK", "EXCITE_FAIL", "IDENT_OK", "IDENT_FAIL")
REAL_CENTER = [1.0, -1.1, 7.5, 1.1, 1.5, 2.0]
"""A center inside the real profile's windows (still its motor-frame numbers, ruling R20) that
keeps the arm clear of the desk."""


def _run(script, args, timeout=300):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "AMENT_PREFIX_PATH")}
    out = subprocess.run(
        [sys.executable, str(script), *map(str, args)],
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


def _rig(tmp_path, **fields):
    path = tmp_path / "rig.json"
    path.write_text(json.dumps({"rt": None, **fields}))
    return path


@pytest.mark.parametrize("script,fail", [(EXCITE, "EXCITE_FAIL"), (IDENT, "IDENT_FAIL")])
def test_the_scripts_print_one_fail_line_on_a_usage_error(script, fail):
    code, tokens, _ = _run(script, ["--no_such_flag"])
    assert code == 2 and len(tokens) == 1 and tokens[0].startswith(fail)
    assert _payload(tokens[0])["reason"] == "usage"


def test_the_excitation_script_writes_ten_files_and_a_session_plan(tmp_path):
    out = tmp_path / "exc"
    args = ["--rig", _rig(tmp_path), "--seed", 5, "--count", 10, "--center", *REAL_CENTER]
    code, tokens, err = _run(EXCITE, [*args, "--out_dir", out])
    assert code == 0 and len(tokens) == 1 and tokens[0].startswith("EXCITE_OK"), err
    payload = _payload(tokens[0])
    assert 1.0 <= payload["cond"] <= 200.0
    names = sorted(p.name for p in out.glob("exc_*.npz"))
    assert names == [f"exc_{k:02d}.npz" for k in range(10)]
    plan = json.loads((out / "plan.json").read_text())
    kinds = [next(iter(seg)) for seg in plan["segments"]]
    assert kinds[0] == "hold" and kinds[-1] == "hold" and kinds.count("traj") == 10
    assert plan["excitation"]["heldout"] == ["exc_08.npz", "exc_09.npz"]
    assert plan["excitation"]["min_clearance_m"] >= 0.05
    # the plan loads under the rig's profile: every file passes check_traj there
    profile = core_profile.load_profile()
    segments = session.load_plan(out / "plan.json", profile)
    assert [s["label"] for s in segments if s["kind"] == "traj"] == names
    with np.load(out / "exc_00.npz") as data:
        assert set(data.files) >= {"t", "q", "grip", "dq", "ddq", "a", "b"}
        assert np.allclose(data["dq"][[0, -1]], 0.0)
    # the same seed draws the same files
    again = tmp_path / "again"
    _run(EXCITE, [*args, "--out_dir", again])
    with np.load(out / "exc_09.npz") as a, np.load(again / "exc_09.npz") as b:
        assert np.array_equal(a["q"], b["q"])


def test_the_excitation_script_refuses_a_poorly_conditioned_set(tmp_path, monkeypatch, capsys):
    script = load_script(EXCITE)
    monkeypatch.setattr(script, "COND_MAX", 1.0)
    argv = ["--rig", str(_rig(tmp_path)), "--seed", "5", "--center", *map(str, REAL_CENTER)]
    code = script.main([*argv, "--out_dir", str(tmp_path)])
    tokens = [x for x in capsys.readouterr().out.splitlines() if x.startswith(TOKENS)]
    assert code == 1 and len(tokens) == 1 and tokens[0].startswith("EXCITE_FAIL")
    assert _payload(tokens[0])["reason"] == "cond"


def test_the_excitation_script_refuses_a_center_outside_the_profile(tmp_path):
    # EXCITE_CENTER is in URDF joint coordinates; the real profile's j3 window is not (R20)
    out = tmp_path / "exc"
    code, tokens, _ = _run(EXCITE, ["--rig", _rig(tmp_path), "--seed", 5, "--out_dir", out])
    assert code == 1 and len(tokens) == 1
    payload = _payload(tokens[0])
    assert payload["reason"] == "design" and "j3" in payload["detail"] and not out.exists()


def test_the_excitation_script_refuses_a_swing_into_the_desk(tmp_path):
    rig = _rig(tmp_path, desk_z=0.5)  # a desk above the arm's shoulder
    args = ["--rig", rig, "--seed", 5, "--center", *REAL_CENTER, "--out_dir", tmp_path / "x"]
    code, tokens, _ = _run(EXCITE, args)
    assert code == 1 and _payload(tokens[0])["reason"] == "desk"


def test_the_excitation_script_never_overwrites_a_file(tmp_path):
    out = tmp_path / "exc"
    out.mkdir()
    (out / "plan.json").write_text("{}")
    code, tokens, _ = _run(EXCITE, ["--rig", _rig(tmp_path), "--seed", 5, "--out_dir", out])
    assert code == 1 and len(tokens) == 1 and _payload(tokens[0])["reason"] == "exists"
    assert (out / "plan.json").read_text() == "{}"


@pytest.fixture(scope="module")
def ident_logs(tmp_path_factory):
    root = tmp_path_factory.mktemp("ident_logs")
    trajs = excitation.design_excitation(
        np.random.default_rng(3),
        core_profile.sim_profile(),
        excitation.EXCITE_CENTER,
        count=3,
        amplitude_rad=None,
    )
    model = _true_model()
    logs = [
        synthetic_log(root / f"exc_{k:02d}.h5", model, tr, noise=0.05, seed=k)
        for k, tr in enumerate(trajs)
    ]
    return logs[:2], logs[2:]


def test_the_ident_script_writes_the_model_and_applies_the_held_out_rule(ident_logs, tmp_path):
    logs, held = ident_logs
    out, report = tmp_path / "ident.json", tmp_path / "report.json"
    code, tokens, err = _run(
        IDENT, ["--logs", *logs, "--heldout", *held, "--out", out, "--report", report]
    )
    assert code == 0 and len(tokens) == 1 and tokens[0].startswith("IDENT_OK"), err
    payload = _payload(tokens[0])
    assert payload["pass"] is True
    assert len(payload["rms_ident"]) == len(payload["rms_urdf"]) == 6
    assert all(i < u for i, u in zip(payload["rms_ident"], payload["rms_urdf"]))
    rep = json.loads(report.read_text())
    assert rep["pass"] is True and rep["heldout"] == [str(h) for h in held]
    assert rep["rms_ident"] == payload["rms_ident"]
    dyn = dm.from_ident(out)
    assert len(dyn.friction([0.3] * 6)) == 6
    data = json.loads(out.read_text())
    assert data["provenance"]["logs"] == [str(p) for p in logs]


def test_the_ident_script_fails_on_a_missing_log(tmp_path):
    out, report = tmp_path / "ident.json", tmp_path / "report.json"
    missing = tmp_path / "missing.h5"
    code, tokens, _ = _run(
        IDENT, ["--logs", missing, "--heldout", missing, "--out", out, "--report", report]
    )
    assert code == 1 and len(tokens) == 1 and tokens[0].startswith("IDENT_FAIL")
    assert _payload(tokens[0])["reason"] == "logs" and not out.exists()


def test_the_ident_script_never_overwrites_its_output(ident_logs, tmp_path):
    logs, held = ident_logs
    out, report = tmp_path / "ident.json", tmp_path / "report.json"
    out.write_text("{}")
    code, tokens, _ = _run(
        IDENT, ["--logs", *logs, "--heldout", *held, "--out", out, "--report", report]
    )
    assert code == 1 and _payload(tokens[0])["reason"] == "exists"
    assert out.read_text() == "{}"
