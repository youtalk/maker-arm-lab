"""scripts/mj_dagger_batch.py prints DAGGER_OK or DAGGER_FAIL on every exit path.

Read as source: the script sets MUJOCO_GL at import and builds a MuJoCo env, so the token
contract is checked structurally, as tests/test_grasp_script_token_contract.py does for the
Isaac grasp script."""

import ast
import pathlib
import types

import pytest

from maker_arm_lab.core import dagger_loop, eval_loop
from test_grasp_script_token_contract import _module_scope_calls

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "mj_dagger_batch.py"


def test_tokens_are_printed_inside_main_and_nothing_risky_runs_at_module_scope():
    text = SCRIPT.read_text()
    tree = ast.parse(text)
    assert "DAGGER_OK " in text and "DAGGER_FAIL " in text
    assert not ({"run", "rollout", "MjLegoSortEnv", "OracleRig"} & _module_scope_calls(tree))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    handlers = [h for n in ast.walk(main) if isinstance(n, ast.Try) for h in n.handlers]
    names = {h.type.id for h in handlers if isinstance(h.type, ast.Name)}
    assert {"EvalFailure", "Exception"} <= names


def test_the_round_sets_the_random_interval():
    text = SCRIPT.read_text()
    assert "TAKEOVER_MEAN_INTERVAL * 2 ** (args.round - 1)" in text


def test_dagger_ok_does_not_print_inside_the_try_clause():
    """The success token must not claim the shard is on disk before env.close() writes it.

    ``env.close()`` runs in the ``finally`` of main's try/except/finally, which flushes the
    recorder. A DAGGER_OK print added right after the payload is built -- still inside the
    ``try`` clause, before the ``finally`` runs -- would satisfy every other structural check
    in this file (the token string exists, the required except handlers exist), so this test
    reads the source of the ``try`` clause specifically (excluding its except/finally, where
    DAGGER_OK does not belong either) and requires the token to be absent from it.

    A second variant of the same mutation is a DAGGER_OK print inserted as the FIRST statement
    of the ``finally`` block itself, ahead of ``env.close()``. That also claims the shard is on
    disk before it is, and the check above alone misses it (the print sits in ``finalbody``,
    not in the ``try`` clause). So this test also locates the ``finally`` statement that calls
    ``env.close()`` and requires no DAGGER_OK print at or before it, among the statements
    preceding and including it.
    """
    text = SCRIPT.read_text()
    tree = ast.parse(text)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    try_node = next(n for n in main.body if isinstance(n, ast.Try))
    try_clause = "\n".join(ast.get_source_segment(text, stmt) or "" for stmt in try_node.body)
    assert "DAGGER_OK" not in try_clause
    close_idx = next(
        i
        for i, stmt in enumerate(try_node.finalbody)
        if "env.close()" in (ast.get_source_segment(text, stmt) or "")
    )
    finally_prefix = "\n".join(
        ast.get_source_segment(text, stmt) or "" for stmt in try_node.finalbody[: close_idx + 1]
    )
    assert "DAGGER_OK" not in finally_prefix
    assert "DAGGER_OK " in text  # still printed somewhere -- after the try/except/finally


def test_the_six_brick_rollouts_refuse_a_continuous_checkpoint():
    """Both core rollouts refuse before they touch the env, so every client (the Isaac
    eval script too) gets the refusal without its own check."""
    policy = types.SimpleNamespace(gripper="continuous")
    draw = {"run_seed": 1, "layout_seed": 2, "permutation": (0, 1, 2, 3), "max_wall_minutes": 1.0}
    with pytest.raises(eval_loop.EvalFailure) as err:
        eval_loop.rollout(None, policy, **draw)
    assert err.value.reason == "gripper"
    with pytest.raises(eval_loop.EvalFailure) as err:
        dagger_loop.rollout(None, None, policy, random_interval=None, **draw)
    assert err.value.reason == "gripper"
