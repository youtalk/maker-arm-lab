"""Token contracts of the tracking bench and Thor timing scripts."""

import ast
import importlib.util
import json
import pathlib
import subprocess
import sys

import maker_arm_rs
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = {"mj_track_bench.py": "BENCH", "track_timing.py": "TIMING", "track_soak.py": "SOAK"}
THOR_ALLOWED = {
    "__future__",
    "argparse",
    "json",
    "math",
    "os",
    "platform",
    "sys",
    "time",
    "traceback",
    "pathlib",
    "statistics",
    "maker_arm_rs",
    "maker_arm_lab.core.dynamics_model",
    "maker_arm_lab.core",
}


def _run(name, *args, timeout=300, prelude=None):
    """The script's last stdout line. ``prelude`` runs first in the same interpreter."""
    script = str(ROOT / "scripts" / name)
    cmd = [sys.executable, script, *args]
    if prelude is not None:
        run = "import runpy, sys\nsys.argv = sys.argv[1:]\nrunpy.run_path(sys.argv[0], None, '__main__')"
        cmd = [sys.executable, "-c", f"{prelude}\n{run}", script, *args]
    out = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return out.stdout.strip().splitlines()[-1]


def test_every_script_prints_its_tokens_and_exits_through_os_exit():
    for name, token in SCRIPTS.items():
        text = (ROOT / "scripts" / name).read_text()
        assert f"{token}_OK " in text and f"{token}_FAIL " in text, name
        assert "os._exit(main())" in text, name


def test_usage_errors_print_the_fail_token():
    for name, token in SCRIPTS.items():
        line = _run(name, "--no_such_flag")
        assert line.startswith(f"{token}_FAIL "), (name, line)
        assert "usage" in json.loads(line.split(" ", 1)[1])["reason"]


def test_the_thor_scripts_import_only_the_stdlib_and_maker_arm_rs():
    for name in ("track_timing.py", "track_soak.py"):
        tree = ast.parse((ROOT / "scripts" / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {node.module}
            else:
                continue
            assert names <= THOR_ALLOWED, (name, names)


def test_timing_runs_a_short_measurement():
    line = _run("track_timing.py", "--ticks", "2000", "--bar_us", "100000")
    assert line.startswith("TIMING_OK "), line
    assert json.loads(line.split(" ", 1)[1])["p99_us"] > 0


def test_the_bench_smoke_is_deterministic():
    a = _run("mj_track_bench.py", "--smoke")
    b = _run("mj_track_bench.py", "--smoke")
    assert a.startswith("BENCH_OK ") and a == b, (a, b)


def test_a_short_soak_stays_enabled():
    line = _run("track_soak.py", "--minutes", "0.05", timeout=120)
    assert line.startswith("SOAK_OK "), line


def test_a_missing_wheel_prints_the_fail_token():
    block = "import sys; sys.modules['maker_arm_rs'] = None"
    for name, args in (
        ("track_timing.py", ("--ticks", "10")),
        ("track_soak.py", ("--minutes", "0.01")),
    ):
        line = _run(name, *args, prelude=block)
        token = SCRIPTS[name]
        assert line.startswith(f"{token}_FAIL "), (name, line)
        assert "maker_arm_rs" in json.loads(line.split(" ", 1)[1])["reason"], (name, line)


def test_a_bench_out_write_failure_keeps_the_verdict(tmp_path):
    line = _run("mj_track_bench.py", "--smoke", "--out", str(tmp_path / "missing" / "out.json"))
    assert line.startswith("BENCH_OK "), line
    assert "FileNotFoundError" in json.loads(line.split(" ", 1)[1])["out_error"]


def test_a_stopped_loop_ends_the_soak_and_still_stops_the_arm(monkeypatch):
    spec = importlib.util.spec_from_file_location("track_soak", ROOT / "scripts" / "track_soak.py")
    soak = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(soak)
    calls = []

    class Arm:
        def enable(self):
            pass

        def start_tracking(self, tracking, rt=None):
            pass

        def snapshot(self):
            return {"state": "enabled", "positions": [0.0] * 7}

        def stop(self):
            calls.append("stop")

    class Tracking:
        def __init__(self, *args, **kwargs):
            pass

        def now(self):
            return None  # the loop's last tick is older than 0.2 s

        def push(self, *args):
            calls.append("push")

    monkeypatch.setattr(maker_arm_rs, "Arm", type("ArmNS", (), {"sim": staticmethod(Arm)}))
    monkeypatch.setattr(maker_arm_rs, "Tracking", Tracking)
    with pytest.raises(RuntimeError, match="loop stopped"):
        soak.run_soak(0.01, None)
    assert calls == ["stop"]
