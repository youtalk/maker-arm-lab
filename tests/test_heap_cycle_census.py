"""scripts/heap_cycle_census.py on synthetic heap payloads (imported, stdlib only)."""

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_heap_cycle_census", Path(__file__).resolve().parents[1] / "scripts/heap_cycle_census.py"
)
census = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(census)


def _episode(**extra) -> dict:
    ep = {
        "num_bricks": 50,
        "bricks_stowed": 2,
        "cycles": 4,
        "cycle_outcomes": [True, False, True, False],
        "cycle_reasons": ["stowed", "budget", "stowed", "out_of_reach"],
        "cycle_steps": [400, 800, 500, 120],
        "cycle_targets": [7, 3, 3, 9],
        "stowed_parts": [7, 4],
        "cycle_classes": ["stowed", "no_close", "stowed", "close_miss"],
        "cycle_first_close_cm": [0.4, None, 1.5, 3.2],
    }
    ep.update(extra)
    return ep


def _write(tmp_path, name: str, episodes: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"episodes": episodes}))
    return path


def test_the_census_sums_classes_stows_closes_and_steps(tmp_path):
    two_in_one = _episode(stowed_parts=[7, 4, 5], bricks_stowed=3)
    paths = [
        _write(tmp_path, "eval_000.json", [_episode()]),
        _write(tmp_path, "eval_001.json", [two_in_one]),
    ]
    s = census.read_payloads(paths)
    assert (s["draws"], s["episodes"], s["cycles"]) == (2, 2, 8)
    assert s["rakes"] == 0  # payloads before the MS7a tray carry none
    raked = census.read_payloads([_write(tmp_path, "eval_002.json", [_episode(rakes=3)])])
    assert raked["rakes"] == 3
    assert (s["parts_total"], s["parts_stowed"], s["stowed_fraction"]) == (100, 5, 0.05)
    assert s["classes"]["stowed"] == {"cycles": 4, "share": 0.5}
    assert s["classes"]["no_close"]["cycles"] == 2 and s["classes"]["close_miss"]["cycles"] == 2
    # Cycle 3 aimed at part 3 and stowed part 4. The second episode stowed two parts in one
    # cycle, so its stows cannot be paired with its cycles.
    assert s["stows"] == {"target": 1, "other_part": 1, "episodes_not_paired": 1}
    assert s["first_close_cm"]["<1"] == {"stowed": 2, "cycles": 2}
    assert s["first_close_cm"]["none"] == {"stowed": 0, "cycles": 2}
    assert s["first_close_cm"][">=3"] == {"stowed": 0, "cycles": 2}
    assert s["cycle_steps"]["budget"] == {"cycles": 2, "median": 800, "p90": 800, "max": 800}
    assert s["cycle_steps"]["stowed"]["median"] == 450


def test_the_census_prints_one_token_on_every_path(tmp_path, capsys):
    path = _write(tmp_path, "eval_000.json", [_episode()])
    assert census.main(["--evals", str(path), "--out", str(tmp_path / "census.json")]) == 0
    assert capsys.readouterr().out.startswith("CENSUS_OK ")
    assert json.loads((tmp_path / "census.json").read_text())["cycles"] == 4
    assert census.main(["--evals", str(tmp_path / "missing_*.json")]) == 1
    assert capsys.readouterr().out.startswith("CENSUS_FAIL ")
    old = _write(
        tmp_path, "eval_002.json", [{k: v for k, v in _episode().items() if k != "cycle_classes"}]
    )
    assert census.main(["--evals", str(old)]) == 1
    assert "predates the census" in capsys.readouterr().out


def test_the_census_sums_the_desk_box_rows(tmp_path):
    rows = {
        "stowed_by_class": {"slope": {"stowed": 1, "total": 2}, "wide": {"stowed": 1, "total": 1}},
        "stowed_by_box": [{"stowed": 1, "total": 2}, {"stowed": 1, "total": 1}],
        "wrong_box_parts": 1,
    }
    boxed = _episode(cycle_classes=["stowed", "no_close", "stowed", "wrong_box"], **rows)
    paths = [_write(tmp_path, f"eval_00{k}.json", [boxed]) for k in range(2)]
    s = census.read_payloads(paths)
    assert s["classes"]["wrong_box"]["cycles"] == 2
    assert s["stowed_by_class"] == {
        "slope": {"stowed": 2, "total": 4},
        "wide": {"stowed": 2, "total": 2},
    }
    assert s["stowed_by_box"] == {"0": {"stowed": 2, "total": 4}, "1": {"stowed": 2, "total": 2}}
    assert s["wrong_box_parts"] == 2
    # A payload without desk boxes reports no box rows.
    assert "stowed_by_class" not in census.read_payloads(
        [_write(tmp_path, "six.json", [_episode()])]
    )
