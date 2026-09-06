"""The grasp script must emit GRASP_OK or GRASP_FAIL on EVERY exit path.

`scripts/scripted_grasp_check.py` is parsed by the D6/D7 supervisor for exactly one of two
tokens. A run that terminates without printing either reads to that supervisor as a HANG
rather than a failure, which is a far worse outcome than a reported failure.

The script imports isaaclab at module scope, so these tests cannot import it. They read it as
SOURCE and assert the structural properties that keep the contract, which is the level the
regression actually happened at: Task D3 fix round 1 "fixed" an unguarded module-scope
`descend_pick_offset(...)[0]` by raising a friendlier `SystemExit` from the same module-scope
expression. That changed the stderr text and fixed nothing -- the exit was still outside the
`try:` that produces the token, and `SystemExit` is a `BaseException` that the `except
Exception` would not have caught even if it had been inside.
"""

import ast
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "scripted_grasp_check.py"

# Anything whose failure must become a token rather than a bare exit. These are the calls
# that can raise on user input; a call to one of them at module scope is the bug.
MUST_NOT_RUN_AT_MODULE_SCOPE = frozenset(
    {"descend_pick_offset", "descend_pick_offset_or_raise", "nominal_offset_z"}
)


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(SCRIPT.read_text())


def _module_scope_calls(tree: ast.Module) -> set[str]:
    """Names called by top-level statements, excluding function/class BODIES.

    Definitions are skipped deliberately: a `def` that calls one of these is exactly the fix,
    since the call then happens when the function runs, inside the guarded region.
    """
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                func = sub.func
                if isinstance(func, ast.Name):
                    found.add(func.id)
                elif isinstance(func, ast.Attribute):
                    found.add(func.attr)
    return found


def test_the_pick_offset_is_not_resolved_at_module_scope(tree):
    """The exact regression: this computation must happen inside the guarded call."""
    offenders = _module_scope_calls(tree) & MUST_NOT_RUN_AT_MODULE_SCOPE
    assert not offenders, (
        f"{sorted(offenders)} run at module scope, before the try/except that turns a "
        f"failure into a GRASP_FAIL token. Move the call inside a function."
    )


def test_no_module_scope_constant_holds_the_lazy_offset(tree):
    """A module constant would drag the computation back to import time by the back door."""
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "NOMINAL_OFFSET_Z" not in assigned


def test_no_default_argument_evaluates_the_offset(tree):
    """Default arguments are evaluated when the `def` executes -- i.e. at import.

    `target_for(..., offset_z=NOMINAL_OFFSET_Z)` was the second copy of this bug, one level
    down and easy to miss: making the helper lazy does nothing if a default argument calls it
    at import anyway.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is None:
                continue
            for sub in ast.walk(default):
                assert not (isinstance(sub, ast.Name) and sub.id in MUST_NOT_RUN_AT_MODULE_SCOPE), (
                    f"{node.name} has a default argument referencing {sub.id}"
                )
                assert not (isinstance(sub, ast.Name) and sub.id == "NOMINAL_OFFSET_Z"), (
                    f"{node.name} evaluates NOMINAL_OFFSET_Z at import"
                )


def test_the_app_launcher_failure_path_emits_a_token(tree):
    """AppLauncher validates its own args (a bad --device) and raises before the try:."""
    source = SCRIPT.read_text()
    launcher = source.index("AppLauncher(args)")
    guarded = source[max(0, launcher - 400) : launcher + 400]
    assert "try:" in guarded
    assert "GRASP_FAIL" in guarded


def test_argparse_usage_errors_emit_a_token(tree):
    """argparse's error() exits 2 with no token; --contact_sensor as a bare flag hits it."""
    parsers = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(base, ast.Attribute) and base.attr == "ArgumentParser" for base in node.bases
        )
    ]
    assert parsers, "the script should subclass ArgumentParser to emit a token on usage errors"
    overrides = [n.name for p in parsers for n in p.body if isinstance(n, ast.FunctionDef)]
    assert "error" in overrides
    assert "GRASP_FAIL" in ast.get_source_segment(SCRIPT.read_text(), parsers[0])


def test_help_is_deliberately_not_covered(tree):
    """--help exits 0 through exit(), not error(); a verdict after a help text is nonsense.

    Pinned so the exclusion stays a decision rather than an oversight.
    """
    source = ast.get_source_segment(
        SCRIPT.read_text(),
        next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name.endswith("Parser")
        ),
    )
    assert "--help" in source
