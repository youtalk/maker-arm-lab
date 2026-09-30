"""The whole-stack tests of MS7f design section 12.2, on every commit of the stack: a switch whose
behaviour is not in the commit fails with its script's token and never runs as a no-op, the
supervisor forwards every MS7f switch and its cache refuses a payload read with other values,
``policy_protocol.py`` gains no import, and every new script prints exactly one terminal token
line on every exit path. The no-truth test with every ready MS7f switch on is
``test_perceived_loop.test_no_decision_reads_the_truth`` (``_ms7f_scene``)."""

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from maker_arm_lab.core import batch_loop

ROOT = Path(__file__).resolve().parents[1]

ON = {
    "categories": ["--categories", "--box_draw", "4"],
    "deep_tray": ["--deep_tray"],
    "tray_refill": ["--tray_refill"],
    "box_empty": ["--box_empty"],
    "classifier": ["--classifier", "/tmp/ms7f_no_such_classifier.pt"],
    "passenger_drop": ["--passenger_drop"],
    "transport_speed": ["--transport_speed", "6.0"],
}
"""Each MS7f switch at a value that is not its off value. A track that adds a switch adds it
here too (the first test holds the table to ``MS7F_OFF``)."""
CLIENTS = [
    ("mj_eval_heap", ["--heap_k", "3", "--oracle_driver", "--desk_boxes", "4"], "EVAL_FAIL ", ()),
    # The recorder's walk pace predates MS7f and is built there (the MS3 speed ladder).
    (
        "mj_generate_batch",
        ["--out_dir", "/tmp/x", "--heap_k", "3", "--desk_boxes", "4"],
        "BATCH_FAIL usage: ",
        ("transport_speed",),
    ),
]
NEW_SCRIPTS = [
    (ROOT / "scripts" / "mj_render_set.py", "RENDER_OK", "RENDER_FAIL"),
    (ROOT / "scripts" / "mj_stretch_run.py", "STRETCH_OK", "STRETCH_FAIL"),
    (ROOT / "lerobot_stack" / "train_part_classifier.py", "CLS_OK", "CLS_FAIL"),
]
"""The new scripts of MS7f and their tokens (the shared interface, item 5)."""
PROTOCOL_IMPORTS = {"io", "json", "numpy"}
"""``lerobot_stack/policy_protocol.py``'s imports at 29a36ef: numpy and pyzmq's wire format only,
never torch or lerobot, because both venvs import it (MS7f design section 6.5)."""


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ge = _load(ROOT / "scripts" / "run_gate_eval.py")


def test_every_ms7f_switch_has_an_on_value_here():
    assert set(ON) == set(batch_loop.MS7F_OFF) == set(ge.MS7F_OFF)
    assert ge.MS7F_OFF == batch_loop.MS7F_OFF and ge.MS7F_READY == batch_loop.MS7F_READY


@pytest.mark.parametrize("name", list(ON))
@pytest.mark.parametrize("script, argv, token, built", CLIENTS)
def test_a_client_refuses_a_switch_whose_behaviour_is_not_in_the_commit(
    name, script, argv, token, built, capsys
):
    if name in batch_loop.MS7F_READY | set(built):
        pytest.skip(f"--{name} is built on this commit")
    module = _load(ROOT / "scripts" / f"{script}.py")
    with pytest.raises(SystemExit) as exit_info:
        module.main(argv + ON[name])
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith(token), out
    assert f"--{name} is not built on this commit" in out[0]


def _gate_argv(tmp_path: Path) -> list[str]:
    return [
        "--checkpoint", "ck", "--out_dir", str(tmp_path), "--ledger", "l", "--threshold", "0.85",
        "--seed", "1", "--backend", "mujoco", "--heap_k", "50", "--desk_boxes", "4",
    ]  # fmt: skip


@pytest.mark.parametrize("name", list(ON))
def test_the_supervisor_refuses_a_switch_whose_behaviour_is_not_in_the_commit(
    name, monkeypatch, capsys, tmp_path
):
    if name in ge.MS7F_READY:
        pytest.skip(f"--{name} is built on this commit")
    monkeypatch.setattr(ge, "_start_server", lambda cfg: pytest.fail("no server on a usage error"))
    assert ge.main(_gate_argv(tmp_path) + ON[name]) == 2
    out = capsys.readouterr()
    assert out.out.strip().splitlines() == ["GATE_ERROR"]
    assert f"--{name} is not built on this commit" in out.err


def test_the_supervisor_forwards_every_ms7f_switch_and_its_cache_refuses_other_values(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(ge, "MS7F_READY", frozenset(ge.MS7F_OFF))
    draw = ge.make_eval_draws(1, 1)[0]
    argv = _gate_argv(tmp_path / "gate")
    plain = ge._build_cfg(ge._parse_args(argv))
    every = [a for name in ON for a in ON[name]]
    cfg = ge._build_cfg(ge._parse_args(argv + every))
    cmd = ge._eval_command(cfg, draw)
    block = ge.ms7f_block(cfg)
    for name, value in block.items():
        assert value != ge.MS7F_OFF[name], name
        at = cmd.index(f"--{name}")
        if value is not True:
            assert cmd[at + 1] == str(value), name
    off = ge._eval_command(plain, draw)
    assert not {f"--{name}" for name in ON} & set(off)
    assert ge.ms7f_block(plain) == ge.MS7F_OFF
    path = tmp_path / "gate" / "eval_000.json"
    path.parent.mkdir(parents=True)

    def hit(gate_cfg, **ms7f):
        payload = {
            "run_seed": draw.run_seed,
            "layout_seed": draw.layout_seed,
            "permutation": list(draw.permutation),
            "policy_id": "act:ck",
            "inference": {"n_action_steps": 50, "temporal_ensemble_coeff": None},
            "episodes": [],
            "heap_k": 50,
            "parts_split": "heldout",
            "parts_seed": 20260915,
            "cycle_budget": 65,
            "gripper": "binary",
            "desk_boxes": 4,
            "ms7e": ge.MS7E_OFF | ({"box_draw": 4} if gate_cfg.box_draw else {}),
            **ms7f,
        }
        path.write_text(json.dumps(payload))
        return ge._cached_payload(gate_cfg, draw) is not None

    assert hit(plain) and hit(plain, ms7f=ge.MS7F_OFF) and not hit(plain, ms7f=block)
    assert hit(cfg, ms7f=block) and not hit(cfg) and not hit(cfg, ms7f=ge.MS7F_OFF)
    for name in ON:
        other = block | {name: ge.MS7F_OFF[name]}
        assert not hit(cfg, ms7f=other), name
    # A payload whose block predates a switch ran with it off.
    older = {k: v for k, v in ge.MS7F_OFF.items() if k != "passenger_drop"}
    assert hit(plain, ms7f=older)


def test_policy_protocol_gains_no_import():
    tree = ast.parse((ROOT / "lerobot_stack" / "policy_protocol.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    assert names == PROTOCOL_IMPORTS


@pytest.mark.parametrize("script, ok, fail", NEW_SCRIPTS, ids=lambda v: getattr(v, "name", v))
def test_every_new_script_prints_one_token_line_on_a_usage_error(script, ok, fail):
    """The argparse path of the token contract, run as a process, and the ``os._exit`` exit
    every script takes (CLAUDE.md, the token contract); each script's own tests cover its other
    exit paths."""
    if not script.exists():
        pytest.skip(f"{script.name} is not on this commit")
    python = sys.executable
    if script.parent.name == "lerobot_stack":
        python = str(ROOT / "lerobot_stack" / ".venv" / "bin" / "python")
        if not Path(python).exists():
            pytest.skip("the lerobot_stack venv is not set up here")
    run = subprocess.run(
        [python, str(script), "--ms7f_no_such_flag"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
        check=False,
    )
    tokens = [line for line in run.stdout.splitlines() if line.startswith((ok, fail))]
    assert run.returncode != 0 and len(tokens) == 1 and tokens[0].startswith(fail), run.stdout
    assert "os._exit(" in script.read_text()
