"""The heap branch of mj_generate_batch.py (as source) and heap_yield_read.py (imported)."""

import ast
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BATCH = ROOT / "scripts" / "mj_generate_batch.py"
READ = ROOT / "scripts" / "heap_yield_read.py"


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_batch_script_builds_the_heap_env_and_installs_the_heap_oracle():
    text = BATCH.read_text()
    tree = ast.parse(text)
    assert "HeapOracle" in text and "heap_parts(args)" in text
    assert "oracle_factory=" in text
    assert 'export_all=args.export_mode == "all" or parts is not None' in text
    assert "MakerArm-LegoHeap-Datagen-mujoco" in text
    assert "BATCH_OK " in text and "BATCH_FAIL" in text
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    assert any(
        isinstance(n, ast.Call) and getattr(n.func, "id", "") == "heap_parts" for n in ast.walk(run)
    )


def test_a_heap_batch_counts_written_episodes_not_only_fully_stowed_ones():
    batch = _load(BATCH)
    counters = {"exported": 0, "exported_written": 3}
    assert batch.exported_count(counters, heap=True) == 3
    # The six-brick path is unchanged: it still counts only fully-successful episodes.
    assert batch.exported_count(counters, heap=False) == 0


def _meta(path, parts, episodes):
    path.write_text(
        json.dumps(
            {
                "run_meta": {"parts": parts, "heap_k": len(parts), "num_bricks": len(parts)},
                "run_stats": {},
                "episodes": episodes,
            }
        )
    )
    return path


def test_read_meta_aggregates_parts_cycles_and_families(tmp_path):
    read = _load(READ)
    parts = ["brick_2x4", "plate_1x2", "slope_1x1"]
    a = _meta(
        tmp_path / "a.meta.json",
        parts,
        [
            {
                "bricks_delivered": 3,
                "cycles": 4,
                "pushes": 1,
                "parts_out_of_reach": 0,
                "cycle_parts": [0, 1, 1, 2],
                "cycle_outcomes": [True, False, True, True],
            }
        ],
    )
    b = _meta(
        tmp_path / "b.meta.json",
        parts,
        [
            {
                "bricks_delivered": 2,
                "cycles": 3,
                "pushes": 0,
                "parts_out_of_reach": 1,
                "cycle_parts": [0, 2, 1],
                "cycle_outcomes": [True, True],
            }
        ],
    )
    summary = read.read_meta([a, b])
    assert summary["episodes"] == 2 and summary["parts_total"] == 6 and summary["parts_stowed"] == 5
    assert summary["stowed_fraction"] == pytest.approx(5 / 6)
    assert summary["cycles"] == 7 and summary["cycles_per_stowed"] == pytest.approx(7 / 5)
    assert summary["pushes"] == 1 and summary["parts_out_of_reach"] == 1
    assert summary["families"]["plate"] == {"attempts": 3, "stowed": 1}
    assert summary["families"]["brick"] == {"attempts": 2, "stowed": 2}
    assert summary["families"]["slope"] == {"attempts": 2, "stowed": 2}
    assert read.verdict(summary, 0.90, 1.3) is False
    assert read.verdict(
        {"stowed_fraction": 0.95, "cycles_per_stowed": 1.2, "episodes": 1}, 0.90, 1.3
    )


def test_main_prints_a_token_on_every_path(tmp_path, capsys):
    read = _load(READ)
    good = _meta(
        tmp_path / "g.meta.json",
        ["brick_2x4"],
        [
            {
                "bricks_delivered": 1,
                "cycles": 1,
                "pushes": 0,
                "parts_out_of_reach": 0,
                "cycle_parts": [0],
                "cycle_outcomes": [True],
            }
        ],
    )
    out = tmp_path / "read.json"
    assert read.main(["--meta", str(good), "--out", str(out)]) == 0
    assert capsys.readouterr().out.startswith("YIELD_OK ")
    assert json.loads(out.read_text())["stowed_fraction"] == 1.0
    assert read.main(["--meta", str(tmp_path / "missing*.json")]) == 1
    assert capsys.readouterr().out.startswith("YIELD_FAIL ")
    with pytest.raises(SystemExit):
        read.main(["--nonsense"])
    assert capsys.readouterr().out.startswith("YIELD_FAIL usage")
