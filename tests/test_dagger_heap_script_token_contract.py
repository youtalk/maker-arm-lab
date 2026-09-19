"""scripts/mj_dagger_heap.py prints DAGGER_OK or DAGGER_FAIL on every exit path."""

import ast
import pathlib

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "mj_dagger_heap.py"


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
    assert "DAGGER_OK " in text and "DAGGER_FAIL " in text
    assert not ({"rollout", "MjLegoSortEnv", "OracleRig"} & _module_scope_calls(tree))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    handlers = [h for n in ast.walk(main) if isinstance(n, ast.Try) for h in n.handlers]
    names = {h.type.id for h in handlers if isinstance(h.type, ast.Name)}
    assert {"EvalFailure", "Exception"} <= names


def test_the_round_sets_the_random_interval_and_the_env_records_everything():
    text = SCRIPT.read_text()
    assert "TAKEOVER_MEAN_INTERVAL * 2 ** (args.round - 1)" in text
    assert "export_all=True" in text and "gripper_mode=args.gripper" in text
