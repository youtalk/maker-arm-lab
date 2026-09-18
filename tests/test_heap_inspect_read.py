"""``scripts/heap_inspect_read.py``: the tool the MS6 rework report's tables come from.

Stdlib and pytest only, following the ``_load`` pattern of ``tests/test_inspect_heap_script.py``.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READ = ROOT / "scripts" / "heap_inspect_read.py"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_log(tmp_path: Path) -> Path:
    """One LIFT that rose without a pinch hold, plus an end event with the new stats."""
    log = tmp_path / "ep" / "events.jsonl"
    log.parent.mkdir()
    events = [
        {
            "step": 5,
            "phase": "LIFT",
            "kind": "phase",
            "outcome": "converged",
            "pinch_ok": False,
            "lifted": True,
            "family": "plate",
        },
        {
            "step": 50,
            "phase": "DONE",
            "kind": "end",
            "reason": "done",
            "done_step": 50,
            "idle_steps": 0,
            "stowed": 1,
            "parts": 5,
            "extra_stats": {
                "cycles": 1,
                "pushes": 0,
                "forced_picks": 0,
                "cycle_parts": [0],
                "cycle_outcomes": [True],
                "cells_excluded": 0,
                "parts_outside_tray": 4,
                "tips": 3,
                "tips_ok": 1,
            },
        },
    ]
    log.write_text("".join(json.dumps(e) + "\n" for e in events))
    return log


def test_read_episode_returns_the_four_new_keys(tmp_path):
    module = _load(READ)
    episode = module.read_episode(str(_write_log(tmp_path)))
    assert episode["outside_last"] == 4  # renamed from outside_at_end
    assert episode["tips"] == 3
    assert episode["tips_ok"] == 1
    assert episode["rescued"] == 1


def test_a_lift_that_rises_without_a_pinch_hold_is_rescued_and_counts_ok(tmp_path):
    """The gate reads `lifted` alone now, so a rejected pinch is not a fail: it is rescued."""
    module = _load(READ)
    episode = module.read_episode(str(_write_log(tmp_path)))
    assert episode["rescued"] == 1
    assert episode["lift_outcomes"] == {"plate_ok": 1}


def test_the_script_prints_exactly_one_inspect_read_token(tmp_path):
    log = _write_log(tmp_path)
    done = subprocess.run(
        [sys.executable, str(READ), str(log)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    tokens = [line for line in done.stdout.splitlines() if line.startswith("INSPECT_READ ")]
    assert len(tokens) == 1
