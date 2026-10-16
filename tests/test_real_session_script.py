"""MS5 control plan task 6: the session runner, its checks before enable, its hold and its
release, on the SDK simulator and on a fake arm."""

import io
import json
import math
import os
import signal
import subprocess
import sys
import time

import h5py
import numpy as np
import pytest

from helpers import ROOT, load_script
from maker_arm_lab.core import profile as core_profile
from maker_arm_lab.real import session
from maker_arm_lab.real.arm import ArmReading
from maker_arm_lab.real.telemetry import TelemetryLog

SCRIPT = ROOT / "scripts" / "real_session.py"
real_session = load_script(SCRIPT)
HZ = 200.0
PROFILE = core_profile.load_profile()
MID = [0.5 * (j["q_lo"] + j["q_hi"]) for j in PROFILE["joints"]]
"""The SDK simulator starts every motor at the middle of its range."""


def _tokens(text):
    return [line for line in text.splitlines() if line.startswith(("SESSION_OK", "SESSION_FAIL"))]


def _payload(token):
    return json.loads(token.split(" ", 1)[1])


def _run(args, stdin="", timeout=120):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "AMENT_PREFIX_PATH")}
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        input=stdin,
        check=False,
    )
    return out.returncode, _tokens(out.stdout), out.stderr


def _rig(tmp_path, **fields):
    path = tmp_path / "rig.json"
    path.write_text(json.dumps({"rt": None, **fields}))
    return path


def _traj(tmp_path, name="exc_00.npz", seconds=2.0, amp=0.1, start=None, joint=0):
    t = np.arange(round(seconds * HZ) + 1) / HZ
    q0 = np.array(MID[:6] if start is None else start, dtype=np.float64)
    q = np.tile(q0, (len(t), 1))
    q[:, joint] += amp * np.sin(2.0 * math.pi * 0.5 * t)
    grip = np.full(len(t), MID[6])
    np.savez(tmp_path / name, t=t, q=q, grip=grip)
    return name


def _plan(tmp_path, segments, name="plan.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"segments": segments}))
    return path


def _never_connect(*args, **kwargs):
    raise AssertionError("the arm was connected")


# --- check_traj ---------------------------------------------------------------------------


def _arrays(q, grip=None):
    q = np.asarray(q, dtype=np.float64)
    t = np.arange(len(q)) / HZ
    grip = np.full(len(q), MID[6]) if grip is None else np.asarray(grip, dtype=np.float64)
    return {"t": t, "q": q, "grip": grip}


def test_check_traj_passes_a_slow_motion_inside_the_limits():
    q = np.tile(MID[:6], (400, 1))
    q[:, 2] += np.linspace(0.0, 0.3, 400)
    session.check_traj(_arrays(q), PROFILE)


def test_check_traj_names_the_joint_and_the_time_past_a_soft_limit():
    q = np.tile(MID[:6], (400, 1))
    hi = PROFILE["joints"][1]["q_hi"]
    q[300:, 1] = hi - 0.05  # inside the hard limit, past the 0.1 rad margin
    with pytest.raises(ValueError, match=r"j2.*1\.5"):
        session.check_traj(_arrays(q), PROFILE)


def test_check_traj_names_the_joint_and_the_time_past_the_velocity_cap():
    q = np.tile(MID[:6], (400, 1))
    q[200:, 3] += 0.05  # one 5 ms step of 0.05 rad is 10 rad/s
    with pytest.raises(ValueError, match=r"j4.*0\.99"):
        session.check_traj(_arrays(q), PROFILE)


def test_check_traj_checks_the_gripper_column_too():
    q = np.tile(MID[:6], (10, 1))
    grip = np.full(10, PROFILE["joints"][6]["q_lo"])
    with pytest.raises(ValueError, match="gripper"):
        session.check_traj(_arrays(q, grip), PROFILE)


def test_check_traj_refuses_a_time_column_off_the_200_hz_grid():
    arrays = _arrays(np.tile(MID[:6], (10, 1)))
    arrays["t"] = arrays["t"] + 0.1
    with pytest.raises(ValueError, match="200 Hz"):
        session.check_traj(arrays, PROFILE)


def test_the_approach_is_a_minimum_jerk_walk_at_the_approach_speed():
    start = list(MID)
    goal = list(MID)
    goal[0] += 0.6
    rows = session.approach_rows(start, goal)
    assert len(rows) == pytest.approx(0.6 / session.APPROACH_SPEED_RAD_S * HZ, abs=1)
    assert rows[0] == pytest.approx(start)
    speed = np.abs(np.diff(np.asarray(rows)[:, 0])) * HZ
    assert speed.max() == pytest.approx(1.875 * session.APPROACH_SPEED_RAD_S, rel=0.02)
    assert session.approach_rows(start, start) == []


# --- the script on the SDK simulator --------------------------------------------------------


def test_a_sim_session_logs_every_tick_and_prints_ok(tmp_path):
    start = list(MID[:6])
    start[0] += 0.15  # the approach walks 0.15 rad
    name = _traj(tmp_path, seconds=2.0, start=start)
    plan = _plan(tmp_path, [{"hold": 1.0}, {"traj": name}, {"hold": 0.5}])
    rig = _rig(tmp_path, caps={"tau_max": [0.5] * 7, "kp": [0.5] * 7, "kd": [0.5] * 7})
    log = tmp_path / "s.h5"
    code, tokens, err = _run(
        ["--rig", rig, "--plan", plan, "--interface", "sim", "--log", log], stdin="RELEASE\n"
    )
    assert code == 0 and len(tokens) == 1 and tokens[0].startswith("SESSION_OK"), err
    payload = _payload(tokens[0])
    total_seconds = 1.0 + session.APPROACH_MIN_S + 2.0 + 0.5
    with h5py.File(log) as f:
        assert f["t"].shape[0] >= 0.9 * HZ * total_seconds and f.attrs["dropped"] == 0
        assert payload["ticks"] == f["t"].shape[0] and payload["dropped"] == 0
        assert payload["segments"] == 4 and payload["faults"] == 0
        kinds = [k.decode() for k in f["segments"]["kind"]]
        assert kinds == ["hold", "approach", "traj", "hold"]
        assert f["trk/tick_t"].shape[0] >= 0.9 * HZ * total_seconds
        # the reduced caps reached the arm: every gain sent is at most half the profile's
        kp_cap = np.array([0.5 * j["kp"] for j in PROFILE["joints"]])
        assert (f["cmd_kp"][:] <= kp_cap + 1e-9).all()
        assert json.loads(f.attrs["profile"])["joints"][1]["kp"] == pytest.approx(kp_cap[1])
        assert json.loads(f.attrs["rig"])["caps"]["kp"] == [0.5] * 7
        assert f.attrs["dynamics"] == "urdf" and len(f.attrs["sdk_revision"]) == 40
        assert f.attrs["lab_revision"] and json.loads(f.attrs["tracking_params"])
        pairs = f["clock_pairs"][:]
        assert pairs.shape[0] >= 8 and (np.diff(pairs[:, 0]) > 0).all()
        # the trajectory was tracked: j1 swept most of its 0.1 rad amplitude
        seg = f["segments"][2]
        t = f["t"][:]
        q = f["q"][:, 0][(t >= seg["start_t"]) & (t <= seg["end_t"])]
        assert q.max() - q.min() > 0.12


def test_a_prompt_records_what_the_operator_typed(tmp_path):
    plan = _plan(tmp_path, [{"hold": 0.3}, {"prompt": "Is the tag visible?", "record": "tag"}])
    log = tmp_path / "s.h5"
    code, tokens, err = _run(
        ["--rig", _rig(tmp_path), "--plan", plan, "--interface", "sim", "--log", log],
        stdin="yes, both corners\nRELEASE\n",
    )
    assert code == 0 and tokens[0].startswith("SESSION_OK"), err
    assert "Is the tag visible?" in err
    with h5py.File(log) as f:
        assert json.loads(f.attrs["records"]) == {"tag": "yes, both corners"}
        assert [k.decode() for k in f["segments"]["kind"]] == ["hold", "prompt"]


def test_a_trajectory_past_a_soft_limit_is_refused_before_enable(tmp_path, monkeypatch, capsys):
    hi = PROFILE["joints"][4]["q_hi"]
    start = list(MID[:6])
    start[4] = hi - 0.12
    name = _traj(tmp_path, seconds=1.0, amp=0.05, start=start, joint=4)
    plan = _plan(tmp_path, [{"hold": 0.5}, {"traj": name}])

    monkeypatch.setattr(real_session, "connect_arm", _never_connect)
    argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--interface", "sim"]
    code = real_session.main([*argv, "--log", str(tmp_path / "s.h5"), "--release", "now"])
    tokens = _tokens(capsys.readouterr().out)
    assert code == 1 and len(tokens) == 1 and tokens[0].startswith("SESSION_FAIL")
    payload = _payload(tokens[0])
    assert payload["reason"] == "plan" and "j5" in payload["detail"]
    assert not (tmp_path / "s.h5").exists()


def test_a_raised_cap_in_the_rig_prints_session_fail(tmp_path):
    plan = _plan(tmp_path, [{"hold": 0.2}])
    rig = _rig(tmp_path, caps={"kp": [1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0]})
    code, tokens, _ = _run(
        ["--rig", rig, "--plan", plan, "--interface", "sim", "--log", tmp_path / "s.h5"]
    )
    assert code == 1 and len(tokens) == 1 and tokens[0].startswith("SESSION_FAIL")
    payload = _payload(tokens[0])
    assert payload["reason"] == "profile" and "caps.kp[1]" in payload["detail"]
    assert not (tmp_path / "s.h5").exists()


def test_a_calibration_the_sdk_refuses_prints_session_fail(tmp_path):
    plan = _plan(tmp_path, [{"hold": 0.2}])
    rig = _rig(tmp_path, calibration={"direction": [1.0] * 6 + [2.0], "offset": [0.0] * 7})
    code, tokens, _ = _run(
        ["--rig", rig, "--plan", plan, "--interface", "sim", "--log", tmp_path / "s.h5"]
    )
    assert code == 1 and len(tokens) == 1
    payload = _payload(tokens[0])
    assert payload["reason"] == "profile" and "joints[6].direction" in payload["detail"]


def test_an_unknown_tracking_parameter_is_refused_before_connect(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(real_session, "connect_arm", _never_connect)
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"tau_thh": [2.0] * 6}))
    plan = _plan(tmp_path, [{"hold": 0.2}])
    argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--params", str(params)]
    code = real_session.main([*argv, "--log", str(tmp_path / "s.h5"), "--release", "now"])
    payload = _payload(_tokens(capsys.readouterr().out)[0])
    assert code == 1 and payload["reason"] == "params" and "tau_thh" in payload["detail"]


def test_a_usage_error_prints_one_session_fail_line():
    code, tokens, _ = _run([])
    assert code == 2 and len(tokens) == 1
    assert _payload(tokens[0])["reason"] == "usage"


def test_an_existing_log_is_never_overwritten(tmp_path):
    log = tmp_path / "s.h5"
    log.write_bytes(b"yesterday")
    plan = _plan(tmp_path, [{"hold": 0.2}])
    code, tokens, _ = _run(
        ["--rig", _rig(tmp_path), "--plan", plan, "--interface", "sim", "--log", log]
    )
    assert code == 1 and _payload(tokens[0])["reason"] == "log_exists"
    assert log.read_bytes() == b"yesterday"


def test_ctrl_c_holds_the_arm_logs_the_ticks_and_waits_for_release(tmp_path):
    plan = _plan(tmp_path, [{"hold": 60.0}])
    log = tmp_path / "s.h5"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "AMENT_PREFIX_PATH")}
    args = ["--rig", _rig(tmp_path), "--plan", plan, "--interface", "sim", "--log", log]
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), *map(str, args)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        deadline = time.monotonic() + 60.0
        while not log.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        time.sleep(1.5)
        proc.send_signal(signal.SIGINT)
        line = proc.stdout.readline()
        assert line.startswith("SESSION_FAIL") and _payload(line)["reason"] == "interrupted"
        time.sleep(0.3)
        assert proc.poll() is None  # the arm holds until the operator types RELEASE
        proc.stdin.write("RELEASE\n")
        proc.stdin.flush()
        assert proc.wait(timeout=30) == 1
    finally:
        if proc.poll() is None:
            proc.kill()
    assert _tokens(proc.stdout.read()) == []
    with h5py.File(log) as f:
        assert f["t"].shape[0] >= 0.9 * HZ * 1.5


# --- the session body on a fake arm --------------------------------------------------------


class FakeSdk:
    """``SdkArm``'s session surface with no motors: ticks at 200 Hz on the wall clock, a
    drain that reports ``dropped`` once, and a fault after ``fault_after_s``."""

    def __init__(self, dropped_loop=0, dropped_trk=0, fault_after_s=None):
        self.t0 = time.monotonic()
        self.drained = 0
        self.dropped = [dropped_loop, dropped_trk]
        self.fault_after_s = fault_after_s
        self.pushes, self.holds, self.stops = [], 0, 0

    def _t(self):
        return time.monotonic() - self.t0

    def _faulted(self):
        return self.fault_after_s is not None and self._t() > self.fault_after_s

    def now(self):
        return None if self._faulted() else self._t()

    def read(self):
        state, fault = ("fault", "over-temperature") if self._faulted() else ("enabled", None)
        return ArmReading(self._t(), tuple(MID[:6]), (0.0,) * 6, (0.0,) * 6, MID[6], state,
                          True, fault)  # fmt: skip

    def start(self):
        return self.read()

    def telemetry(self):
        return None

    def push_rows(self, t0, dt, rows):
        self.pushes.append((t0, dt, len(rows)))

    def drain(self):
        upto = int(self._t() * HZ)
        ks = range(self.drained, upto)
        self.drained = upto
        seven = [list(MID) for _ in ks]
        ticks = {
            "t": [k / HZ for k in ks],
            "tick": list(ks),
            "clamped": [False for _ in ks],
            "q": seven,
            "dq": seven,
            "tau": seven,
            "temp": seven,
            "fault_bits": [[0] * 7 for _ in ks],
            "feedback_age": seven,
            "cmd_pos": seven,
            "cmd_vel": seven,
            "cmd_kp": seven,
            "cmd_kd": seven,
            "cmd_tau": seven,
            "dropped": self.dropped[0],
        }
        trk = {
            "tick_t": [k / HZ for k in ks],
            "r": [[0.0] * 6 for _ in ks],
            "gate": [1.0 for _ in ks],
            "offset": [[0.0] * 6 for _ in ks],
            "q_r": [list(MID[:6]) for _ in ks],
            "tau_ff": [[0.0] * 6 for _ in ks],
            "dropped": self.dropped[1],
        }
        self.dropped = [0, 0]
        return ticks, trk

    def hold(self):
        self.holds += 1

    def stop(self):
        self.stops += 1


def test_a_drain_that_fell_behind_records_its_dropped_count(tmp_path):
    arm = FakeSdk(dropped_loop=37, dropped_trk=5)
    log = TelemetryLog(tmp_path / "s.h5", {})
    lines = session.Lines(io.StringIO(""))
    summary = session.run_session(arm, [{"kind": "hold", "seconds": 0.3}], log, lines)
    log.close()
    assert summary["dropped"] == 42 and summary["segments"] == 1
    with h5py.File(tmp_path / "s.h5") as f:
        assert f.attrs["dropped"] == 42
        assert f.attrs["dropped_loop"] == 37 and f.attrs["dropped_trk"] == 5
        assert f["t"].shape[0] == summary["ticks"] > 0


def test_the_ok_payload_reports_the_dropped_total(tmp_path, monkeypatch, capsys):
    arm = FakeSdk(dropped_loop=3, dropped_trk=2)
    monkeypatch.setattr(real_session, "connect_arm", lambda *a, **k: arm)
    plan = _plan(tmp_path, [{"hold": 0.2}])
    argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--interface", "sim"]
    code = real_session.main([*argv, "--log", str(tmp_path / "s.h5"), "--release", "now"])
    tokens = _tokens(capsys.readouterr().out)
    assert code == 0 and len(tokens) == 1 and _payload(tokens[0])["dropped"] == 5
    assert arm.holds == 0 and arm.stops == 1


def test_a_fault_holds_the_arm_and_releases_only_on_release(tmp_path, monkeypatch, capsys):
    arm = FakeSdk(fault_after_s=0.3)
    monkeypatch.setattr(real_session, "connect_arm", lambda *a, **k: arm)

    class Operator(io.StringIO):
        def readline(self):
            stops_seen.append(arm.stops)
            return super().readline()

    stops_seen = []
    monkeypatch.setattr(sys, "stdin", Operator("not yet\nRELEASE\n"))
    plan = _plan(tmp_path, [{"hold": 5.0}])
    argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--interface", "sim"]
    code = real_session.main([*argv, "--log", str(tmp_path / "s.h5")])
    tokens = _tokens(capsys.readouterr().out)
    assert code == 1 and len(tokens) == 1 and _payload(tokens[0])["reason"] == "fault"
    assert "over-temperature" in _payload(tokens[0])["detail"]
    assert arm.holds == 1 and arm.stops == 1 and stops_seen[:2] == [0, 0]
    with h5py.File(tmp_path / "s.h5") as f:
        assert f["t"].shape[0] > 0  # the ticks up to the fault stay in the log


def test_end_of_input_at_a_prompt_fails_and_holds(tmp_path):
    arm = FakeSdk()
    log = TelemetryLog(tmp_path / "s.h5", {})
    with pytest.raises(session.SessionFailure) as err:
        session.run_session(
            arm, [{"kind": "prompt", "text": "ready?", "record": "r"}], log,
            session.Lines(io.StringIO("")),
        )  # fmt: skip
    log.close()
    assert err.value.reason == "end_of_input"


def test_the_plan_loader_refuses_an_unknown_segment(tmp_path):
    plan = _plan(tmp_path, [{"hold": 1.0}, {"wait": 2.0}])
    with pytest.raises(ValueError, match="segment 1"):
        session.load_plan(plan, PROFILE)
