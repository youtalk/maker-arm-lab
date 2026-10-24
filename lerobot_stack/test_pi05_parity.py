"""pi05_parity.py compare, the only command that needs no model or corpus."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("pi05_parity.py")


def compare(tmp_path, delta):
    a = np.random.default_rng(0).standard_normal((16, 50, 7)).astype(np.float32)
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", a + delta)
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "compare", str(tmp_path / "a.npy"), str(tmp_path / "b.npy")],
        check=False,
        capture_output=True,
        text=True,
    )
    (line,) = [x for x in run.stdout.splitlines() if x.startswith("PARITY_")]
    token, _, payload = line.partition(" ")
    return run.returncode, token, json.loads(payload)


def test_compare_fails_over_the_joint_bar(tmp_path):
    delta = np.zeros((16, 50, 7), np.float32)
    delta[..., 2] = 0.02
    code, token, info = compare(tmp_path, delta)
    assert (code, token) == (1, "PARITY_FAIL") and info["p99"][2] > 0.01


def test_compare_fails_over_the_jaw_bar(tmp_path):
    delta = np.zeros((16, 50, 7), np.float32)
    delta[..., 6] = 0.001
    assert compare(tmp_path, delta)[:2] == (1, "PARITY_FAIL")


def test_compare_passes_identical_chunks(tmp_path):
    code, token, info = compare(tmp_path, np.zeros((16, 50, 7), np.float32))
    assert (code, token, info["p99"]) == (0, "PARITY_OK", [0.0] * 7)


def test_a_failing_compare_prints_its_token_through_a_pipe():
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "compare", "/nonexistent/a.npy", "/nonexistent/b.npy"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    lines = [x for x in run.stdout.splitlines() if x.startswith("PARITY_")]
    assert run.returncode == 1 and len(lines) == 1 and lines[0].startswith("PARITY_FAIL ")
