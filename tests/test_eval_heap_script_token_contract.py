"""scripts/mj_eval_heap.py prints EVAL_OK or EVAL_FAIL on every exit path.

Read as source: the script sets MUJOCO_GL at import and builds a MuJoCo env, so the token
contract is checked structurally, as tests/test_dagger_script_token_contract.py does."""

import ast
import pathlib

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "mj_eval_heap.py"


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
