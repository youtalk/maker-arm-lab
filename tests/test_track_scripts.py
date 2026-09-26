"""Token contracts of the tracking bench and Thor timing scripts."""

import ast
import json
import pathlib
import subprocess
import sys

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


def _run(name, *args, timeout=300):
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
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
