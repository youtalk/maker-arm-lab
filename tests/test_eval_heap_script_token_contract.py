"""scripts/mj_eval_heap.py prints EVAL_OK or EVAL_FAIL on every exit path.

Read as source: the script sets MUJOCO_GL at import and builds a MuJoCo env, so the token
contract is checked structurally, as tests/test_dagger_script_token_contract.py does."""

import ast
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mj_eval_heap.py"


def _module_scope_calls(tree):
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                func = sub.func
                found.add(func.id if isinstance(func, ast.Name) else getattr(func, "attr", ""))
    return found


def test_tokens_are_printed_inside_main_and_nothing_risky_runs_at_module_scope():
    text = SCRIPT.read_text()
    tree = ast.parse(text)
    assert "EVAL_OK " in text and "EVAL_FAIL " in text
    assert not ({"rollout", "MjLegoSortEnv", "OracleRig"} & _module_scope_calls(tree))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    handlers = [h for n in ast.walk(main) if isinstance(n, ast.Try) for h in n.handlers]
    names = {h.type.id for h in handlers if isinstance(h.type, ast.Name)}
    assert {"EvalFailure", "Exception"} <= names


def test_usage_errors_print_the_fail_token():
    text = SCRIPT.read_text()
    assert 'print("EVAL_FAIL " + json.dumps({"reason": "usage"' in text


def test_the_heap_client_takes_the_gripper_flag():
    text = SCRIPT.read_text()
    assert '"--gripper"' in text and "gripper_mode=args.gripper" in text


def test_the_heap_client_takes_the_protocol_switches():
    text = SCRIPT.read_text()
    for name in (
        "select_at_release",
        "replan_on_switch",
        "row_refresh",
        "squeeze_floor",
        "close_gate",
        "square_wrist",
        "scripted_place",
        "scripted_rake",
        "part_failures",
        "pick_snap",
        "dwell_close",
        "adopt_held",
        "park_done",
        "rake_turn",
        "perceived",
        "noise_scales",
        "contact_hold",
    ):
        assert f'"--{name}"' in text and f"{name}=args.{name}" in text


PERCEIVED = ["--perceived", "--desk_boxes", "4", "--gripper", "continuous"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--perceived", "--gripper", "continuous"],  # no desk boxes
        ["--perceived", "--desk_boxes", "4"],  # the binary gripper
        [*PERCEIVED, "--oracle_driver"],
        ["--contact_hold"],
        ["--noise_scales", "heading=2"],
        [*PERCEIVED, "--noise_scales", "bogus=1"],
        [*PERCEIVED, "--noise_scales", "heading=-1"],
    ],
)
def test_the_perceived_flags_refuse_a_bad_combination_with_the_token(argv):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "source"), "MUJOCO_GL": "egl"}
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert out.returncode == 2, out.stderr
    assert out.stdout.strip().splitlines() == [out.stdout.strip()]  # one token line
    assert out.stdout.startswith('EVAL_FAIL {"reason": "usage"'), out.stdout


def test_the_heap_client_writes_the_hold_trace():
    text = SCRIPT.read_text()
    assert '"--hold_trace"' in text and "hold_trace=trace" in text
