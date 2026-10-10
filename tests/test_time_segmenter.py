"""MS5 design P1: the segmenter timing bench's pure parts (the model runs only on the Thor)."""

import ast
import importlib.util
import json
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


def test_the_auto_row_is_reported_but_does_not_enter_the_verdict():
    slow_auto = time_segmenter.verdict([400.0] * 100, [100.0] * 100, [2000.0] * 20, [30] * 20)
    assert slow_auto["pass"]
    assert slow_auto["auto_ms"]["p95"] == 2000.0 and not slow_auto["auto_ms"]["under_bar"]
    assert slow_auto["auto_ms"]["masks_p50"] == 30.0
    fast_auto = time_segmenter.verdict([900.0] * 100, [100.0] * 100, [300.0] * 20)
    assert not fast_auto["pass"] and fast_auto["auto_ms"]["under_bar"]
    assert "auto_ms" not in time_segmenter.verdict([400.0] * 10, [100.0] * 10, [])


def test_the_auto_generator_uses_its_default_grid():
    assert time_segmenter.AUTO_POINTS_PER_SIDE == 32
    assert time_segmenter.AUTO_POINTS_PER_BATCH == 64
    args = time_segmenter.build_parser().parse_args([])
    assert args.auto_points_per_side == 32 and args.auto_calls is None


def test_the_synthetic_tray_holds_parts_on_a_grey_floor():
    rng = np.random.default_rng(0)
    image = time_segmenter.synthetic_tray(1024, 30, rng)
    assert image.shape == (1024, 1024, 3) and image.dtype == np.uint8
    floor = (image == 128).all(axis=2)
    assert 0.5 < floor.mean() < 1.0
    again = time_segmenter.synthetic_tray(1024, 30, np.random.default_rng(0))
    assert (image == again).all()
    assert time_segmenter.synthetic_tray(8, 3, rng).shape == (8, 8, 3)


def test_validate_names_every_bad_argument():
    parser = time_segmenter.build_parser()
    time_segmenter.validate(parser.parse_args([]))
    bad = parser.parse_args(
        [
            "--parts",
            "0",
            "--warmup",
            "-1",
            "--tray_px",
            "0",
            "--wrist_px",
            "-5",
            "--calls",
            "0",
            "--auto_calls",
            "-1",
            "--auto_points_per_side",
            "0",
            "--models",
            ",",
        ]
    )
    try:
        time_segmenter.validate(bad)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("validate accepted bad arguments")
    for flag in (
        "--parts",
        "--warmup",
        "--tray_px",
        "--wrist_px",
        "--calls",
        "--auto_calls",
        "--auto_points_per_side",
        "--models",
    ):
        assert flag in message
    time_segmenter.validate(parser.parse_args(["--auto_calls", "0"]))


def test_a_bad_value_prints_one_fail_token_before_any_model_loads(monkeypatch, capsys):
    def never(*_args):
        raise AssertionError("a model loaded")

    monkeypatch.setattr(time_segmenter, "time_model", never)
    assert time_segmenter.main(["--parts", "0"]) == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("SEGTIME_FAIL ") and "--parts" in out[0]


def test_a_keyboard_interrupt_prints_one_fail_token(monkeypatch, capsys):
    def interrupted(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(time_segmenter, "time_model", interrupted)
    assert time_segmenter.main([]) == 130
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ['SEGTIME_FAIL {"reason": "interrupted"}']


def test_the_summary_names_the_verdict_rows_and_keeps_auto_informational(monkeypatch, capsys):
    def fake(_name, _args):
        return time_segmenter.verdict([400.0] * 5, [100.0] * 5, [900.0] * 5, [12] * 5)

    monkeypatch.setattr(time_segmenter, "time_model", fake)
    assert time_segmenter.main(["--models", "m", "--calls", "5"]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("SEGTIME_OK ")
    summary = json.loads(line.split(" ", 1)[1])
    assert summary["verdict_rows"] == ["perceive_ms", "wrist_ms"]
    assert summary["auto"]["informational"] and not summary["auto"]["any_under_bar"]
    assert summary["bars_ms"]["auto_p95"] == 500.0
