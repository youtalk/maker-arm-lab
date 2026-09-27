"""MS5 design P1: the segmenter timing bench's pure parts (the model runs only on the Thor)."""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "perception_stack" / "time_segmenter.py"
spec = importlib.util.spec_from_file_location("time_segmenter", SCRIPT)
time_segmenter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(time_segmenter)


def test_grid_prompts_are_n_single_points_inside_the_image():
    pts = time_segmenter.grid_prompts(30, 1024)
    assert pts.shape == (30, 1, 2) and pts.dtype == np.float32
    assert (pts > 0).all() and (pts < 1024).all()
    assert len({tuple(p) for p in pts[:, 0].tolist()}) == 30


def test_verdict_reads_both_bars_on_the_p95():
    fast = time_segmenter.verdict([400.0] * 100, [100.0] * 100)
    assert fast["pass"] and fast["perceive_ms"]["p95"] == 400.0
    slow = time_segmenter.verdict([400.0] * 90 + [900.0] * 10, [100.0] * 100)
    assert not slow["pass"]
    assert not time_segmenter.verdict([400.0] * 100, [200.0] * 100)["pass"]


def test_the_bars_are_the_pre_registered_ones():
    assert time_segmenter.PERCEIVE_BAR_MS == 500.0 and time_segmenter.WRIST_BAR_MS == 150.0


def test_a_usage_error_prints_the_fail_token():
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--calls", "x"],
        capture_output=True,
        text=True,
        check=False,
    )
    last = out.stdout.strip().splitlines()[-1]
    assert out.returncode == 2 and last.startswith("SEGTIME_FAIL ") and "usage" in last


def test_module_scope_imports_neither_torch_nor_sam2():
    tree = ast.parse(SCRIPT.read_text())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name.split(".")[0] for n in top for a in n.names}
    names |= {n.module.split(".")[0] for n in top if isinstance(n, ast.ImportFrom) and n.module}
    assert not names & {"torch", "sam2"}
