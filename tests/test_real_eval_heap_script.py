"""MS5 design sections 7 and 8: the real eval client's token contract, its hold and release."""

import ast
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from maker_arm_lab.real.rig import RigConfig, episode_sheet
from real_fakes import FakeArm, FakeClock, make_env
from test_perceived_loop import MS7D_ON

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "real_eval_heap.py"
spec = importlib.util.spec_from_file_location("real_eval_heap", SCRIPT)
real_eval_heap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real_eval_heap)
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


def _fake_world(monkeypatch, rollout):
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
            closers=[lambda: closed.append(1)],
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


def test_the_served_switches_are_the_ms7d_reads():
    switches = real_eval_heap.SERVED_SWITCHES
    assert set(MS7D_ON) <= set(switches) and all(switches.values())
    assert {"select_at_release", "replan_on_switch", "scripted_rake", "pick_snap"} <= set(switches)


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


def test_an_interrupt_holds_prints_one_token_and_releases_on_typed_release(monkeypatch, capsys):
    def interrupted(*args, **kwargs):
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


def test_without_ros_a_run_on_the_sdk_simulator_fails_with_one_token(tmp_path):
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
