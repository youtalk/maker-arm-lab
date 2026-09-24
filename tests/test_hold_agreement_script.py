"""scripts/hold_agreement.py on a tiny synthetic payload and trace: one token line per exit."""

import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from maker_arm_lab.core import hold

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hold_agreement.py"
_spec = importlib.util.spec_from_file_location("_hold_agreement", SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)


def _trace() -> list[list]:
    """Cycle 1 stalls on a 15.8 mm part and lifts it 9 cm; cycle 2 closes on nothing."""
    rows = [[0, s, -1, 0.0238, 0.0238, 0.02, -1, 0.0] for s in range(2)]
    for k, gap in enumerate([0.0171] + [0.0158] * 31):
        held = 3 if k >= 2 else -1
        z = 0.02 + max(0, k - 8) * 0.004
        rows.append([0, 2 + k, 1, gap, 0.0098, z, held, z - 0.02 if held >= 0 else 0.0])
    for k, gap in enumerate([0.0171, 0.0204] + [0.0238] * 4):  # the release, between cycles
        rows.append([0, 34 + k, -1, gap, 0.0238, 0.11, -1, 0.0])
    for k, gap in enumerate([0.0171, 0.0104, 0.0038] + [0.0018] * 10):
        rows.append([0, 40 + k, 2, gap, 0.0018, 0.02, -1, 0.0])
    return rows


def _files(tmp_path):
    trace = tmp_path / "eval_000.trace.jsonl.gz"
    with gzip.open(trace, "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in _trace())
    payload = tmp_path / "eval_000.json"
    payload.write_text(json.dumps({"episodes": [{"cycle_census": hold.cycle_rows(_trace())}]}))
    return payload, trace


def _token(capsys) -> tuple[str, dict]:
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1, lines
    token, _, body = lines[0].partition(" ")
    return token, json.loads(body)


def test_the_payload_and_its_trace_read_the_same_agreement(tmp_path, capsys):
    payload, _ = _files(tmp_path)
    assert script.main(["--payloads", str(payload)]) == 0
    token, from_payload = _token(capsys)
    assert token == "HOLD_OK" and from_payload["pass"]
    assert from_payload["hold"]["tp"] == 1 and from_payload["hold"]["tn"] == 1
    assert from_payload["lift"]["tp"] == 1 and from_payload["first_hold_offset_median"] == 3
    assert script.main(["--traces", str(tmp_path / "*.trace.jsonl.gz")]) == 0
    assert _token(capsys) == ("HOLD_OK", from_payload)


def test_a_threshold_override_can_fail_the_bar(tmp_path, capsys):
    _, trace = _files(tmp_path)
    assert script.main(["--traces", str(trace), "--min_excess", "0.01"]) == 1
    token, summary = _token(capsys)
    assert token == "HOLD_FAIL" and summary["hold"]["fn"] == 1
    assert summary["thresholds"] == {"min_excess": 0.01}


def test_a_missing_file_is_an_error_token(tmp_path, capsys):
    assert script.main(["--payloads", str(tmp_path / "missing_*.json")]) == 1
    token, body = _token(capsys)
    assert token == "HOLD_ERROR" and "FileNotFoundError" in body["detail"]


def test_argparse_errors_print_one_error_token(tmp_path):
    payload, _ = _files(tmp_path)
    for argv in ([], ["--bogus"], ["--payloads", str(payload), "--min_gap", "0.004"]):
        done = subprocess.run(
            [sys.executable, str(SCRIPT), *argv],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        lines = done.stdout.strip().splitlines()
        assert done.returncode == 2 and len(lines) == 1, (argv, done.stdout, done.stderr)
        assert lines[0].startswith("HOLD_ERROR ") and '"usage"' in lines[0]
