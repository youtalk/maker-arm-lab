"""The drop, settle and tray sweep: its token contract and its pure measurement function."""

import ast
import importlib.util
import json
import pathlib

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SWEEP = ROOT / "scripts" / "mj_heap_drop_sweep.py"
READ = ROOT / "scripts" / "heap_sweep_read.py"


def _load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sweep_script_prints_its_tokens_and_exits_through_os_exit():
    source = SWEEP.read_text()
    tree = ast.parse(source)
    assert '"SWEEP_OK "' in source and '"SWEEP_FAIL "' in source
    exits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_exit"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
    ]
    assert exits, "the script must leave through os._exit"
    assert "except SystemExit" in source  # an argparse error is SWEEP_FAIL, not a hang


def test_the_read_script_prints_its_token_and_exits_through_os_exit():
    source = READ.read_text()
    tree = ast.parse(source)
    assert '"SWEEP_READ "' in source
    exits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_exit"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
    ]
    assert exits, "the script must leave through os._exit"


def _sample(outside, wall_blocked, blocked, no_grasp, stacked, unreachable, speed_max):
    return {
        "outside": outside,
        "wall_blocked": wall_blocked,
        "blocked": blocked,
        "no_grasp": no_grasp,
        "stacked": stacked,
        "unreachable": unreachable,
        "speed_max": speed_max,
        "tray_w_mm": 130.0,
        "tray_h_mm": 200.0,
        "wall_height_mm": 20.0,
        "drop_min_mm": 30.0,
        "drop_per_part_mm": 1.2,
        "settle_steps": 480,
        "layout_seed": 0,
        "heap_k": 8,
    }


def test_read_groups_a_two_sample_fixture_into_one_baseline_cell(tmp_path, capsys):
    read = _load(READ)
    samples = [_sample(2, 1, 1, 1, 0, 0, 0.01), _sample(4, 3, 2, 1, 1, 1, 0.05)]
    path = tmp_path / "samples.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in samples))
    assert read.main([str(path)]) == 0
    lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("SWEEP_READ ")
    ]
    assert len(lines) == 1
    summary = json.loads(lines[0][len("SWEEP_READ ") :])
    assert summary["samples"] == 2
    assert len(summary["cells"]) == 1
    cell = summary["cells"][0]
    assert cell["n"] == 2 and cell["baseline"] is True
    assert cell["outside_mean"] == pytest.approx(3.0)
    assert cell["outside_min"] == 2 and cell["outside_max"] == 4
    assert cell["stacked_mean"] == pytest.approx(0.5)
    assert cell["unreachable_max"] == 1


def _quat() -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 0.0, 0.0])


def test_measure_heap_flags_an_outside_a_stacked_and_a_no_grasp_part():
    module = _load(SWEEP)
    brick = (0.0318, 0.0158, 0.0096)  # brick_2x4, from core.parts.footprint
    # Both horizontal sides over MAX_GRASP_WIDTH (0.045 m): candidate_grasps finds no axis.
    too_wide = (0.05, 0.05, 0.01)
    sizes = torch.tensor([brick, brick, brick, too_wide])
    quat = torch.stack([_quat(), _quat(), _quat(), _quat()])
    half_height = 0.5 * brick[2]
    pos = torch.tensor(
        [
            [0.15, 0.00, half_height],  # resting flat, inside the tray, in reach: the baseline
            [0.40, 0.05, half_height],  # outside the tray, still in reach
            [0.15, 0.10, 0.02],  # inside and in reach, but well above its own half height
            [0.15, 0.20, 0.005],  # inside, in reach, resting flat, but ungraspable on any axis
        ]
    )
    tray_bounds = ((0.0, 0.3), (0.0, 0.3))
    counts = module.measure_heap(
        sizes,
        pos,
        quat,
        tray_bounds,
        wall_footprints=[],
        reach_x_limit=0.5,
        reach_radius=0.5,
    )
    assert counts == {
        "outside": 1,
        "wall_blocked": 1,
        "blocked": 1,
        "no_grasp": 1,
        "stacked": 1,
        "unreachable": 0,
    }
