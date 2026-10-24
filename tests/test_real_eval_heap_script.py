"""MS5 design sections 7 and 8: the real eval client's token contract, its hold and release."""

import ast
import io
import json
import math
import os
import signal
import subprocess
import sys

import pytest

from helpers import ROOT, load_script
from maker_arm_lab.real.rig import RigConfig, episode_sheet
from real_fakes import FakeArm, FakeClock, make_env
from test_perceived_loop import MS7D_ON

SCRIPT = ROOT / "scripts" / "real_eval_heap.py"
real_eval_heap = load_script(SCRIPT)
ARGS = ["--sheet", "s.json", "--heap_k", "3", "--dilation", "0.5"]


def _run(*args):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "AMENT_PREFIX_PATH")}
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    return out.returncode, out.stdout.strip().splitlines()


def _tokens(text):
    return [line for line in text.splitlines() if line.startswith(("EVAL_OK", "EVAL_FAIL"))]


class _Unread:
    """An input that fails the test if anything reads it."""

    def readline(self):
        raise AssertionError("release read the input")


class _Operator:
    """The operator at the prompt: presses Ctrl-C (a real SIGINT) before typing each line, and
    records how often the arm was stopped when each line was read."""

    def __init__(self, arm, lines):
        self.arm, self.lines, self.stops_seen = arm, list(lines), []

    def readline(self):
        os.kill(os.getpid(), signal.SIGINT)
        self.stops_seen.append(self.arm.stops)
        return self.lines.pop(0) if self.lines else ""


def _fake_world(monkeypatch, rollout, closer=None):
    env, arm, _ = make_env()
    closed = []

    def build(args, world):
        world.update(
            arm=arm,
            env=env,
            rig=object(),
            policy=object(),
            budget=3,
            sheet={"class_to_box": [0, 1, 2, 3]},
            config=RigConfig(),
            worker_id="fake",
            closers=[lambda: closed.append(1), *([closer] if closer else [])],
        )

    monkeypatch.setattr(real_eval_heap, "build_world", build)
    monkeypatch.setattr("maker_arm_lab.core.loop.rollout", rollout)
    return env, arm, closed


def test_the_token_contract_is_in_the_text():
    text = SCRIPT.read_text()
    assert "EVAL_OK " in text and "EVAL_FAIL " in text and "os._exit(main())" in text
    body = ast.parse(text).body
    top = {n.module for n in body if isinstance(n, ast.ImportFrom) and n.module}
    assert not top & {"maker_arm_lab.real.env", "maker_arm_lab.real.arm", "maker_arm_lab.real.ros"}


def test_the_served_switches_are_the_ms7e_reads():
    switches = real_eval_heap.SERVED_SWITCHES
    assert set(MS7D_ON) <= set(switches) and all(switches.values())
    assert {"select_at_release", "replan_on_switch", "scripted_rake", "pick_snap"} <= set(switches)
    assert switches["thin_snap"] and switches["snap_closed"]
    assert real_eval_heap.SERVED_ORACLE == {"wall_band_cont": True}


OLD_22 = dict.fromkeys(
    [
        "select_at_release",
        "replan_on_switch",
        "squeeze_floor",
        "square_wrist",
        "scripted_rake",
        "part_failures",
        "pick_snap",
        "dwell_close",
        "adopt_held",
        "park_done",
        "rake_turn",
        "box_filter_xy",
        "release_gap",
        "rearm_hold",
        "box_walls",
        "marker_bias",
        "wrist_fix",
        "class_views",
        "hold_force",
        "force_cap",
        "thin_snap",
        "snap_closed",
    ],
    True,
)


def test_the_served_switches_are_s_without_the_image_classifier():
    script = real_eval_heap
    assert script.SERVED_SWITCHES == {
        **OLD_22,
        "carry_guard": True,
        "overhang_guard": True,
        "missed_lift": True,
        "cell_failure_cap": 5,
    }
    assert script.SERVED_RIG == {"release_lean": True, "carry_via": True, "passenger_check": True}
    assert script.SERVED_ORACLE == {"wall_band_cont": True}
    assert script.SERVED_INFERENCE == {"n_action_steps": 50, "action_warp": 1.5}
    assert not {"classifier", "cls_flip", "vote_weights", "categories"} & set(
        script.SERVED_SWITCHES
    )


class _Server:
    def __init__(self, n_action_steps=50, inference=None):
        self.n_action_steps, self.inference = n_action_steps, inference or {}


@pytest.mark.parametrize(
    "server",
    [
        _Server(inference={"action_warp": 1.0}),
        _Server(inference={}),
        _Server(1),
        _Server(inference={"action_warp": 1.5, "infer_delay": 1}),  # an emulated delay
        _Server(inference={"action_warp": 1.5, "infer_delay": 6, "rtc": 25}),
        _Server(inference={"action_warp": 1.5, "rtc": 25}),
    ],
)
def test_the_client_refuses_another_serving_or_an_emulated_delay(monkeypatch, capsys, server):
    env, arm, _ = make_env()

    def build(args, world):
        world.update(arm=arm, env=env, closers=[])
        real_eval_heap.check_served_inference(server)

    monkeypatch.setattr(real_eval_heap, "build_world", build)
    code = real_eval_heap.main([*ARGS, "--release", "now"])
    tokens = _tokens(capsys.readouterr().out)
    assert code == 1 and len(tokens) == 1 and '"served_inference"' in tokens[0]
    assert arm.holds == 1


def test_the_client_accepts_the_served_inference():
    real_eval_heap.check_served_inference(_Server(inference={"action_warp": 1.5}))


def test_build_world_checks_the_served_inference_before_it_keeps_the_policy():
    """The refusal test above stubs build_world, so the real call is pinned in its text."""
    tree = ast.parse(SCRIPT.read_text())
    build = next(n for n in tree.body if getattr(n, "name", None) == "build_world")
    body = [ast.unparse(statement) for statement in build.body]
    check = body.index("check_served_inference(server)")
    assert check < body.index("world['policy'] = TimedPolicy(server, env.ticks, env.trip)")


@pytest.mark.parametrize("args", [["--heap_k", "10"], [*ARGS[:4], "--dilation", "1.5"]])
def test_a_usage_error_prints_the_fail_token(args):
    code, lines = _run(*args)
    assert code == 2 and lines[-1].startswith("EVAL_FAIL ") and "usage" in lines[-1]


def test_release_waits_for_the_typed_word_or_eof():
    arm = FakeArm(FakeClock())
    stdin = io.StringIO("no\nrelease\nRELEASE\nlater\n")
    real_eval_heap.release(arm, "prompt", stdin=stdin)
    assert arm.stops == 1 and stdin.read() == "later\n"
    arm = FakeArm(FakeClock())
    real_eval_heap.release(arm, "prompt", stdin=io.StringIO(""))
    assert arm.stops == 1
    arm = FakeArm(FakeClock())
    real_eval_heap.release(arm, "now", stdin=_Unread())
    assert arm.stops == 1 and arm.holds == 0
    # No arm connected: nothing to disable, so no prompt reads the input.
    assert real_eval_heap.release(None, "now", stdin=_Unread()) is None
    assert real_eval_heap.release(None, "prompt", stdin=_Unread()) is None


@pytest.mark.parametrize("how", ["raised", "signalled"])
def test_an_interrupt_holds_prints_one_token_and_releases_on_typed_release(
    monkeypatch, capsys, how
):
    def interrupted(*args, **kwargs):
        if how == "signalled":
            os.kill(os.getpid(), signal.SIGINT)
        raise KeyboardInterrupt

    _, arm, closed = _fake_world(monkeypatch, interrupted)
    monkeypatch.setattr(sys, "stdin", io.StringIO("RELEASE\n"))
    code = real_eval_heap.main(ARGS)
    tokens = _tokens(capsys.readouterr().out)
    assert code == 1 and len(tokens) == 1 and '"interrupted"' in tokens[0]
    assert arm.holds == 1 and arm.stops == 1 and closed == [1]


def test_a_watchdog_trip_holds_once(monkeypatch, capsys):
    def tripped(env, *args, **kwargs):
        env.trip("tick", "a late tick")

    _, arm, _ = _fake_world(monkeypatch, tripped)
    code = real_eval_heap.main([*ARGS, "--release", "now"])
    tokens = _tokens(capsys.readouterr().out)
    assert code == 1 and len(tokens) == 1 and '"watchdog_tick"' in tokens[0]
    assert arm.holds == 1 and arm.stops == 1


def test_a_finished_episode_prints_ok_with_the_real_block_and_releases(
    monkeypatch, capsys, tmp_path
):
    def finished(env, *args, **kwargs):
        # A blind scene's marker error is NaN; the token and --out must still carry it.
        return {"episodes": [], "steps_run": 0, "perception": {"marker_error_mm": math.nan}}

    _, arm, _ = _fake_world(monkeypatch, finished)
    out = tmp_path / "e.json"
    code = real_eval_heap.main([*ARGS, "--release", "now", "--out", str(out)])
    tokens = _tokens(capsys.readouterr().out)
    assert code == 0 and len(tokens) == 1 and tokens[0].startswith("EVAL_OK ")
    payload = json.loads(tokens[0][len("EVAL_OK ") :])
    assert payload["real"]["graded"] is False and "plain_ms" in payload["real"]["ticks"]
    written = json.loads(out.read_text())
    assert math.isnan(payload["perception"].pop("marker_error_mm"))
    assert math.isnan(written["perception"].pop("marker_error_mm"))
    assert written == payload
    assert arm.holds == 0 and arm.stops == 1


class _InterruptedOnClear(list):
    """``episode``: a Ctrl-C lands as the inner ``finally`` clears it, after the EVAL_OK
    assignment (the narrow window of MS5 final review M1)."""

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        raise KeyboardInterrupt


def test_a_ctrl_c_after_the_ok_assignment_exits_1_with_the_fail_token(
    monkeypatch, capsys, tmp_path
):
    def finished(env, *args, **kwargs):
        return {"episodes": [], "steps_run": 0}

    _, arm, closed = _fake_world(monkeypatch, finished)
    out = tmp_path / "e.json"
    args = real_eval_heap.parse_args([*ARGS, "--release", "now", "--out", str(out)])
    code = real_eval_heap._run(args, {}, _InterruptedOnClear([True]))
    tokens = _tokens(capsys.readouterr().out)
    assert code == 1 and len(tokens) == 1 and '"interrupted"' in tokens[0]
    assert not out.exists() and arm.holds == 1 and arm.stops == 1 and closed == [1]


def test_the_sigint_handler_writes_its_note_with_os_write():
    """A print to sys.stderr inside the handler can re-enter an interrupted stderr write."""
    handler = next(
        n
        for n in ast.walk(ast.parse(SCRIPT.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "on_sigint"
    )
    calls = {ast.unparse(n.func) for n in ast.walk(handler) if isinstance(n, ast.Call)}
    assert "os.write" in calls and "print" not in calls


@pytest.mark.parametrize("finish", ["tripped", "finished"])
def test_a_ctrl_c_after_the_episode_keeps_the_arm_held_until_the_typed_release(
    monkeypatch, capfd, finish
):
    """Review Focus 3: a Ctrl-C in the teardown (here in a closer, then at the prompt) neither
    costs the token nor disables the motors; only the typed RELEASE does."""

    def tripped(env, *args, **kwargs):
        env.trip("tick", "a late tick")

    def finished(env, *args, **kwargs):
        return {"episodes": [], "steps_run": 0}

    def interrupting_closer():
        os.kill(os.getpid(), signal.SIGINT)
        closed.append("after the Ctrl-C")

    rollout = tripped if finish == "tripped" else finished
    _, arm, closed = _fake_world(monkeypatch, rollout, closer=interrupting_closer)
    operator = _Operator(arm, ["no\n", "RELEASE\n", "later\n"])
    monkeypatch.setattr(sys, "stdin", operator)
    before = signal.getsignal(signal.SIGINT)
    code = real_eval_heap.main(ARGS)
    captured = capfd.readouterr()
    tokens = _tokens(captured.out)
    assert len(tokens) == 1 and closed == ["after the Ctrl-C", 1]
    if finish == "tripped":
        assert code == 1 and '"watchdog_tick"' in tokens[0] and arm.holds == 1
    else:
        assert code == 0 and tokens[0].startswith("EVAL_OK ") and arm.holds == 0
    assert operator.stops_seen == [0, 0] and operator.lines == ["later\n"]
    assert arm.stops == 1 and "the arm still holds" in captured.err
    assert signal.getsignal(signal.SIGINT) is before


def test_on_the_sdk_simulator_without_a_stream_a_run_fails_with_one_token(tmp_path):
    sheet = tmp_path / "s.json"
    sheet.write_text(json.dumps(episode_sheet(20265000, 0)))
    code, lines = _run(
        "--arm",
        "sim",
        "--sheet",
        str(sheet),
        "--heap_k",
        "3",
        "--dilation",
        "0.5",
        "--release",
        "now",
    )
    assert code == 1 and lines[-1].startswith("EVAL_FAIL ")
    assert len([line for line in lines if line.startswith(("EVAL_OK", "EVAL_FAIL"))]) == 1
