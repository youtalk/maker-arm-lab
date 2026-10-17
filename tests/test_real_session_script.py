"""MS5 control plan task 6: the session runner, its checks before enable, its hold and its
release, on the SDK simulator and on a fake arm."""

import io
import json
import math
import os
import signal
import subprocess
import sys
import threading
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
        # kp at half the profile's: the error clamp doubles, so kp e_max stays the profile's
        assert json.loads(f.attrs["tracking_params"])["e_max"] == pytest.approx([0.06] * 6)
        assert f.attrs["controller"] == "tracking"
        assert payload["late_ticks"] == f.attrs["late_ticks"]
        assert 0.0 < payload["max_interval_s"] == f.attrs["max_interval_s"]
        pairs = f["clock_pairs"][:]
        assert pairs.shape[0] >= 8 and (np.diff(pairs[:, 0]) > 0).all()
        # the trajectory was tracked: j1 swept most of its 0.1 rad amplitude
        seg = f["segments"][2]
        t = f["t"][:]
        q = f["q"][:, 0][(t >= seg["start_t"]) & (t <= seg["end_t"])]
        assert q.max() - q.min() > 0.12


def _answer_when_asked(proc, answers):
    """Read the script's stderr and write each answer once its question has printed."""
    pending = list(answers)
    err = []
    while pending:
        line = proc.stderr.readline()
        assert line, "".join(err)
        err.append(line)
        if pending[0][0] in line:
            proc.stdin.write(pending.pop(0)[1] + "\n")
            proc.stdin.flush()
    return "".join(err)


def test_a_prompt_records_what_the_operator_typed(tmp_path):
    plan = _plan(tmp_path, [{"hold": 0.3}, {"prompt": "Is the tag visible?", "record": "tag"}])
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
        # an answer typed before the question is dropped, not recorded
        proc.stdin.write("too early\n")
        proc.stdin.flush()
        _answer_when_asked(proc, [("then Enter", "yes, both corners"), ("type RELEASE", "RELEASE")])
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    tokens = _tokens(proc.stdout.read())
    assert len(tokens) == 1 and tokens[0].startswith("SESSION_OK")
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


def test_a_non_identity_calibration_prints_session_fail_before_connect(tmp_path):
    plan = _plan(tmp_path, [{"hold": 0.2}])
    rig = _rig(tmp_path, calibration={"direction": [1.0] * 7, "offset": [0.0] * 6 + [0.2]})
    code, tokens, _ = _run(
        ["--rig", rig, "--plan", plan, "--interface", "sim", "--log", tmp_path / "s.h5"]
    )
    assert code == 1 and len(tokens) == 1
    payload = _payload(tokens[0])
    assert payload["reason"] == "profile" and "calibration.offset[6]" in payload["detail"]
    assert "motor frame" in payload["detail"] and not (tmp_path / "s.h5").exists()


def test_a_profile_the_sdk_refuses_at_connect_prints_session_fail(tmp_path, monkeypatch, capsys):
    from maker_arm_lab.real import rig as real_rig

    loosened = core_profile.load_profile()
    loosened["joints"][3]["tau_max"] *= 2.0  # past the lab's checks: the SDK still refuses
    monkeypatch.setattr(real_rig, "reduced_profile", lambda rig: loosened)
    plan = _plan(tmp_path, [{"hold": 0.2}])
    argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--interface", "sim"]
    code = real_session.main([*argv, "--log", str(tmp_path / "s.h5"), "--release", "now"])
    payload = _payload(_tokens(capsys.readouterr().out)[0])
    assert code == 1 and payload["reason"] == "profile"
    assert "joints[3].tau_max" in payload["detail"] and not (tmp_path / "s.h5").exists()


def test_an_unknown_tracking_parameter_is_refused_before_connect(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(real_session, "connect_arm", _never_connect)
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"tau_thh": [2.0] * 6}))
    plan = _plan(tmp_path, [{"hold": 0.2}])
    argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--params", str(params)]
    argv += ["--interface", "sim"]
    code = real_session.main([*argv, "--log", str(tmp_path / "s.h5"), "--release", "now"])
    payload = _payload(_tokens(capsys.readouterr().out)[0])
    assert code == 1 and payload["reason"] == "params" and "tau_thh" in payload["detail"]


def test_an_error_clamp_past_the_scaled_one_is_refused_before_connect(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(real_session, "connect_arm", _never_connect)
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"e_max": [0.03, 0.07, 0.03, 0.03, 0.03, 0.03]}))
    plan = _plan(tmp_path, [{"hold": 0.2}])
    rig = _rig(tmp_path, caps={"kp": [0.5] * 7})
    argv = ["--rig", str(rig), "--plan", str(plan), "--params", str(params)]
    argv += ["--interface", "sim"]
    code = real_session.main([*argv, "--log", str(tmp_path / "s.h5"), "--release", "now"])
    payload = _payload(_tokens(capsys.readouterr().out)[0])
    assert code == 1 and payload["reason"] == "params" and "e_max[1]" in payload["detail"]


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
        self.pushed_rows = []

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
        self.pushed_rows.append([list(r) for r in rows])

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
    # the fake ticks every 5 ms on the grid: no interval is late
    assert _payload(tokens[0])["max_interval_s"] == pytest.approx(0.005)
    assert _payload(tokens[0])["late_ticks"] == 0


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


def test_an_interrupt_flag_ends_the_run_between_drains(tmp_path):
    arm = FakeSdk()
    log = TelemetryLog(tmp_path / "s.h5", {})
    t_stop = time.monotonic() + 0.3
    with pytest.raises(session.SessionFailure) as err:
        session.run_session(
            arm, [{"kind": "hold", "seconds": 10.0}], log, session.Lines(io.StringIO("")),
            interrupted=lambda: time.monotonic() > t_stop,
        )  # fmt: skip
    log.close()
    assert err.value.reason == "interrupted" and time.monotonic() - t_stop < 1.0


def test_ctrl_c_in_the_session_sets_a_flag_and_takes_the_hold_path(tmp_path, monkeypatch, capsys):
    arm = FakeSdk()
    monkeypatch.setattr(real_session, "connect_arm", lambda *a, **k: arm)
    drain = arm.drain

    def drain_then_ctrl_c():
        # A real SIGINT inside the drain: with a raising handler it would escape mid-append.
        out = drain()
        if arm.drained > 0.2 * HZ and not getattr(arm, "signalled", False):
            arm.signalled = True
            os.kill(os.getpid(), signal.SIGINT)
        return out

    arm.drain = drain_then_ctrl_c
    plan = _plan(tmp_path, [{"hold": 10.0}])
    argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--interface", "sim"]
    code = real_session.main([*argv, "--log", str(tmp_path / "s.h5"), "--release", "now"])
    tokens = _tokens(capsys.readouterr().out)
    assert code == 1 and len(tokens) == 1 and _payload(tokens[0])["reason"] == "interrupted"
    # the controller still runs: it keeps its gravity feed-forward and holds the current
    # target (one row), rather than swap to the SDK's pure PD hold
    assert arm.holds == 0 and arm.stops == 1
    assert arm.pushes[-1][2] == 1 and arm.pushed_rows[-1] == [list(MID)]
    with h5py.File(tmp_path / "s.h5") as f:
        assert len({f[k].shape[0] for k in ("t", "q", "cmd_tau", "fault_bits")}) == 1


def test_a_release_typed_during_the_session_does_not_release_at_its_end(
    tmp_path, monkeypatch, capsys
):
    arm = FakeSdk()
    monkeypatch.setattr(real_session, "connect_arm", lambda *a, **k: arm)
    r, w = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(r, "r"))
    writer = os.fdopen(w, "w")
    writer.write("RELEASE\n")  # typed early, during the hold
    writer.flush()
    seen = []

    def operator():
        time.sleep(1.5)  # the session (0.5 s) is over and the release prompt waits
        seen.append(arm.stops)
        writer.write("RELEASE\n")
        writer.close()

    typist = threading.Thread(target=operator)
    typist.start()
    plan = _plan(tmp_path, [{"hold": 0.5}])
    argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--interface", "sim"]
    code = real_session.main([*argv, "--log", str(tmp_path / "s.h5")])
    typist.join()
    assert code == 0 and _tokens(capsys.readouterr().out)[0].startswith("SESSION_OK")
    assert seen == [0] and arm.stops == 1


def test_a_prompt_drops_lines_typed_before_it():
    r, w = os.pipe()
    lines = session.Lines(os.fdopen(r, "r")).start()
    os.write(w, b"early\n")
    time.sleep(0.2)
    lines.clear()
    os.write(w, b"late\n")
    assert lines.get(2.0) == "late\n"
    os.close(w)


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


# --- the controller, the joint frame and the operator's terminal ---------------------------


def test_a_hold_controller_session_on_sim_logs_every_tick(tmp_path):
    plan = _plan(tmp_path, [{"hold": 1.0}, {"hold": 0.5}])
    log = tmp_path / "s.h5"
    args = ["--rig", _rig(tmp_path), "--plan", plan, "--interface", "sim", "--log", log]
    code, tokens, err = _run([*args, "--controller", "hold"], stdin="RELEASE\n")
    assert code == 0 and len(tokens) == 1 and tokens[0].startswith("SESSION_OK"), err
    payload = _payload(tokens[0])
    with h5py.File(log) as f:
        assert f.attrs["controller"] == "hold" and f.attrs["dynamics"] == "none"
        assert f["t"].shape[0] >= 0.9 * HZ * 1.5 and payload["ticks"] == f["t"].shape[0]
        assert [k.decode() for k in f["segments"]["kind"]] == ["hold", "hold"]
        assert "tick_t" not in f["trk"]  # no model, no Tracking
        # the SDK hold: the profile's gains and no feed-forward on the arm joints
        kp = np.array([j["kp"] for j in PROFILE["joints"]])
        assert np.allclose(f["cmd_kp"][:], kp) and not f["cmd_tau"][:, :6].any()
        q = f["q"][:]
        assert np.abs(q - q[0]).max() < 0.05


def test_a_trajectory_under_the_hold_controller_is_refused_before_connect(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(real_session, "connect_arm", _never_connect)
    plan = _plan(tmp_path, [{"hold": 0.5}, {"traj": _traj(tmp_path, seconds=1.0)}])
    argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--interface", "sim"]
    code = real_session.main(
        [*argv, "--log", str(tmp_path / "s.h5"), "--release", "now", "--controller", "hold"]
    )
    payload = _payload(_tokens(capsys.readouterr().out)[0])
    assert code == 1 and payload["reason"] == "plan" and "hold controller" in payload["detail"]
    assert not (tmp_path / "s.h5").exists()


def test_tracking_on_a_real_interface_is_refused_before_connect(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(real_session, "connect_arm", _never_connect)
    plan = _plan(tmp_path, [{"hold": 0.5}])
    argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--interface", "vcan0"]
    code = real_session.main([*argv, "--log", str(tmp_path / "s.h5"), "--release", "now"])
    tokens = _tokens(capsys.readouterr().out)
    assert code == 1 and len(tokens) == 1
    payload = _payload(tokens[0])
    assert payload["reason"] == "joint_frame" and "motor frame" in payload["detail"]
    assert "--controller hold" in payload["detail"] and not (tmp_path / "s.h5").exists()


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def test_a_real_session_refuses_input_that_is_not_a_terminal(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(real_session, "connect_arm", _never_connect)
    monkeypatch.setattr(sys, "stdin", io.StringIO("RELEASE\n"))
    plan = _plan(tmp_path, [{"hold": 0.5}])
    argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--interface", "vcan0"]
    code = real_session.main([*argv, "--log", str(tmp_path / "s.h5"), "--controller", "hold"])
    tokens = _tokens(capsys.readouterr().out)
    assert code == 1 and len(tokens) == 1
    payload = _payload(tokens[0])
    assert payload["reason"] == "stdin" and "terminal" in payload["detail"]


def test_a_real_hold_session_on_a_terminal_reaches_the_arm(tmp_path, monkeypatch, capsys):
    arm = FakeSdk()
    seen = {}

    def connect(interface, dynamics, profile, params, rt, controller, residual=None):
        seen.update(interface=interface, controller=controller, dynamics=dynamics)
        return arm

    monkeypatch.setattr(real_session, "connect_arm", connect)
    monkeypatch.setattr(sys, "stdin", _Terminal("RELEASE\n"))
    plan = _plan(tmp_path, [{"hold": 0.2}])
    argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--interface", "vcan0"]
    code = real_session.main([*argv, "--log", str(tmp_path / "s.h5"), "--controller", "hold"])
    assert code == 0 and _tokens(capsys.readouterr().out)[0].startswith("SESSION_OK")
    assert seen == {"interface": "vcan0", "controller": "hold", "dynamics": None}


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_sigterm_and_sighup_hold_the_arm_and_print_one_token(tmp_path, signum):
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
        proc.send_signal(signum)
        line = proc.stdout.readline()
        assert line.startswith("SESSION_FAIL") and _payload(line)["reason"] == "interrupted"
        assert signal.Signals(signum).name in _payload(line)["detail"]
        time.sleep(0.3)
        proc.send_signal(signum)  # a second one at the release prompt only says it holds
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


def _ident_file(tmp_path, armature=0.02):
    from maker_arm_lab.core import dynamics_model as dm

    path = tmp_path / "ident.json"
    data = {
        "bodies": [[m, list(c), [list(r) for r in i]] for m, c, i in dm.urdf_bodies()],
        "friction": [[0.1, 0.02, 0.01, 0.15, 0.05]] * 6,
        "armature": [armature] * 6,
        "torque_sigma": [0.1] * 6,
        "provenance": {"logs": [], "date": "2026-10-16"},
    }
    path.write_text(json.dumps(data))
    return path


def test_an_ident_file_is_the_controllers_model_and_is_logged(tmp_path, monkeypatch, capsys):
    from maker_arm_lab.core import dynamics_model as dm

    ident = _ident_file(tmp_path)
    seen = {}

    def connect(interface, dynamics, profile, params, rt, controller, residual=None):
        seen["dynamics"] = dynamics
        raise ValueError("stop here")

    monkeypatch.setattr(real_session, "connect_arm", connect)
    plan = _plan(tmp_path, [{"hold": 0.2}])
    argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--dynamics", str(ident)]
    argv += ["--interface", "sim"]
    code = real_session.main([*argv, "--log", str(tmp_path / "s.h5"), "--release", "now"])
    assert code == 1 and _payload(_tokens(capsys.readouterr().out)[0])["reason"] == "profile"
    args = ([0.1] * 6, [0.3] * 6, [1.0] * 6)
    want = dm.from_ident(ident)
    assert np.allclose(seen["dynamics"].inverse(*args), want.inverse(*args))
    assert np.allclose(seen["dynamics"].friction([0.3] * 6), want.friction([0.3] * 6))
    assert not np.allclose(
        seen["dynamics"].inverse(*args), dm.dynamics(dm.urdf_bodies()).inverse(*args)
    )


def test_a_sim_session_with_an_ident_file_records_it(tmp_path):
    ident = _ident_file(tmp_path)
    plan = _plan(tmp_path, [{"hold": 0.3}])
    log = tmp_path / "s.h5"
    code, tokens, err = _run(
        ["--rig", _rig(tmp_path), "--plan", plan, "--interface", "sim", "--log", log]
        + ["--dynamics", ident],
        stdin="RELEASE\n",
    )
    assert code == 0 and tokens[0].startswith("SESSION_OK"), err
    with h5py.File(log) as f:
        assert f.attrs["dynamics"] == str(ident)
        assert json.loads(f.attrs["dynamics_ident"]) == json.loads(ident.read_text())


def test_a_bad_ident_file_prints_session_fail_before_connect(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(real_session, "connect_arm", _never_connect)
    bad = _ident_file(tmp_path, armature=-0.1)  # the SDK refuses a negative armature
    plan = _plan(tmp_path, [{"hold": 0.2}])
    for dynamics in (str(bad), str(tmp_path / "missing.json")):
        argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--dynamics", dynamics]
        argv += ["--interface", "sim"]
        code = real_session.main([*argv, "--log", str(tmp_path / "s.h5"), "--release", "now"])
        payload = _payload(_tokens(capsys.readouterr().out)[0])
        assert code == 1 and payload["reason"] == "dynamics" and dynamics in payload["detail"]
    assert not (tmp_path / "s.h5").exists()


def test_the_hold_controller_refuses_an_ident_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(real_session, "connect_arm", _never_connect)
    plan = _plan(tmp_path, [{"hold": 0.2}])
    argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--interface", "sim"]
    argv += ["--dynamics", str(_ident_file(tmp_path)), "--controller", "hold"]
    code = real_session.main([*argv, "--log", str(tmp_path / "s.h5"), "--release", "now"])
    payload = _payload(_tokens(capsys.readouterr().out)[0])
    assert code == 1 and payload["reason"] == "dynamics" and "no model" in payload["detail"]


def _residual_file(tmp_path, name="residual.json", **change):
    """A small exported residual model (``models.residual.export``) of history 2."""
    import torch

    from maker_arm_lab.models import residual

    torch.manual_seed(0)
    n_in = residual.n_inputs(2)
    stats = {
        "x_mean": np.zeros(n_in),
        "x_std": np.full(n_in, 10.0),  # the sim's 35 degC and its q stay inside X_BOUND
        "y_mean": np.zeros(6),
        "y_std": np.full(6, 0.01),
    }
    path = tmp_path / name
    spec = residual.export(residual.ResidualMlp(n_in, hidden=(8,)), stats, 2, None, path)
    if change:
        path.unlink()
        path.write_text(json.dumps({**spec, **change}))
    return path


def _sha(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _residual_report(tmp_path, ident, model, name="residual_report.json", **change):
    """The training report's provenance as ``scripts/real_residual_train.py`` writes it."""
    path = tmp_path / name
    body = {
        "model": str(model),
        "model_sha256": _sha(model),
        "provenance": {"ident": str(ident), "ident_sha256": _sha(ident)},
    }
    body["provenance"].update(change.pop("provenance", {}))
    body.update(change)
    path.write_text(json.dumps(body))
    return path


def test_a_sim_session_with_a_residual_model_runs_it_and_records_it(tmp_path):
    model = _residual_file(tmp_path)
    ident = _ident_file(tmp_path)
    report = _residual_report(tmp_path, ident, model)
    plan = _plan(tmp_path, [{"hold": 0.3}])
    log = tmp_path / "s.h5"
    code, tokens, err = _run(
        ["--rig", _rig(tmp_path), "--plan", plan, "--interface", "sim", "--log", log]
        + ["--dynamics", ident, "--residual", model, "--residual_report", report],
        stdin="RELEASE\n",
    )
    assert code == 0 and tokens[0].startswith("SESSION_OK"), err
    with h5py.File(log) as f:
        assert f.attrs["residual"] == str(model)
        assert json.loads(f.attrs["residual_model"]) == json.loads(model.read_text())
        assert json.loads(f.attrs["residual_report"]) == json.loads(report.read_text())
        tau_res = f["trk/tau_res"][:]
        skips = f["trk/residual_skips"][:]
    # the first tick fills the history of two, then the model acts on every tick
    assert np.all(np.abs(tau_res[5:]).max(axis=1) > 0.0) and skips[-1] <= 2


def test_a_session_without_a_residual_records_none(tmp_path):
    plan = _plan(tmp_path, [{"hold": 0.2}])
    log = tmp_path / "s.h5"
    code, _, err = _run(
        ["--rig", _rig(tmp_path), "--plan", plan, "--interface", "sim", "--log", log],
        stdin="RELEASE\n",
    )
    assert code == 0, err
    with h5py.File(log) as f:
        assert f.attrs["residual"] == "" and json.loads(f.attrs["residual_model"]) is None
        assert not np.any(f["trk/tau_res"][:])


def test_a_bad_residual_model_prints_session_fail_before_connect(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(real_session, "connect_arm", _never_connect)
    unknown = _residual_file(tmp_path, "unknown.json", y_mean=[0.0] * 6)
    short = _residual_file(tmp_path, "short.json", history=3)  # 2-tick layers, 3-tick history
    plan = _plan(tmp_path, [{"hold": 0.2}])
    ident = _ident_file(tmp_path)
    for model in (unknown, short, tmp_path / "missing.json"):
        report = _residual_report(tmp_path, ident, unknown, provenance={})
        argv = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--residual", str(model)]
        argv += ["--interface", "sim", "--dynamics", str(ident), "--residual_report", str(report)]
        code = real_session.main([*argv, "--log", str(tmp_path / "s.h5"), "--release", "now"])
        payload = _payload(_tokens(capsys.readouterr().out)[0])
        assert code == 1 and payload["reason"] == "residual" and str(model) in payload["detail"]
    assert not (tmp_path / "s.h5").exists()


def test_a_residual_needs_the_report_of_its_ident_file(tmp_path, monkeypatch, capsys):
    """Ruling R35: ``--residual`` needs ``--residual_report``, whose ident sha256 must be the
    ``--dynamics`` file's and whose model sha256 the model's; each check fails before
    connect."""
    monkeypatch.setattr(real_session, "connect_arm", _never_connect)
    model = _residual_file(tmp_path)
    ident = _ident_file(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    other_ident = _ident_file(other, armature=0.03)
    plan = _plan(tmp_path, [{"hold": 0.2}])
    base = ["--rig", str(_rig(tmp_path)), "--plan", str(plan), "--interface", "sim"]
    base += ["--log", str(tmp_path / "s.h5"), "--release", "now", "--residual", str(model)]
    cases = [
        ([], "--residual_report"),
        (["--residual_report", str(_residual_report(tmp_path, ident, model))], "urdf"),
        (
            ["--dynamics", str(other_ident)]
            + ["--residual_report", str(_residual_report(tmp_path, ident, model))],
            "ident_sha256",
        ),
        (
            ["--dynamics", str(ident), "--residual_report"]
            + [str(_residual_report(tmp_path, ident, model, "r2.json", model_sha256="0" * 64))],
            "model_sha256",
        ),
        (["--dynamics", str(ident), "--residual_report", str(tmp_path / "none.json")], "none"),
    ]
    for extra, needle in cases:
        code = real_session.main([*base, *extra])
        payload = _payload(_tokens(capsys.readouterr().out)[0])
        assert code == 1 and payload["reason"] == "residual", (extra, payload)
        assert needle in payload["detail"], (needle, payload["detail"])
    assert not (tmp_path / "s.h5").exists()
