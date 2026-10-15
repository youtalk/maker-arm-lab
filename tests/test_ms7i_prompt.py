"""MS7i ``pi05_prompt`` (rev3 design 6.2): the heap loop sends each env's subtask id with every
request, the very ids ``record_all`` records, and never a replan; the client and the supervisor
wire it. MuJoCo three-part heap with four desk boxes, cameras off."""

import json

import h5py
import numpy as np
import pytest

from maker_arm_lab.core import hold, loop
from test_ms7f_dagger import ARGV, _wiring
from test_ms7f_stack import _load, ge
from test_ms7i_own_frames import EVAL_HEAP, _scene
from test_perceived_loop import _env_and_rig, _rollout, _served
from test_run_gate_eval import PI05_HEAP, _argv, _cfg, _episode, _fake_ledger, _payload
from test_thin_snap import _same


def _run(monkeypatch, tmp_path, record_all: bool, on: bool):
    """The teacher scene's perceived rollout with ``replan_on_switch``: the payload without its
    wall clock, what env 0 was served, env 0's (replan flag, prompt ids) of every request, the
    shard path and the served width of each ``subtask_step`` call."""
    path = tmp_path / f"teacher_{int(record_all)}{int(on)}"
    env, rig, policy, budget, switches = _scene(
        "teacher",
        monkeypatch,
        export_dir=path.parent,
        shard_name=path.name,
        export_all=True,
        record_all=record_all,
    )
    sent, real = [], policy.step

    def step(*args, replan=None, prompt=None, **kwargs):
        sent.append((bool(replan[0]), None if prompt is None else prompt.copy()))
        return real(*args, replan=replan, **kwargs)  # the teacher ignores the ids

    policy.step = step
    widths = []

    def subtask_step(lifted, low, gap, prev_cmd, pick_z, width):
        widths.append(width)
        return hold.subtask_step(lifted, low, gap, prev_cmd, pick_z, width)

    monkeypatch.setattr(loop, "subtask_step", subtask_step)
    served = _served(env)
    switches |= {"replan_on_switch": True}
    payload = _rollout(
        env,
        rig,
        policy,
        budget=budget,
        perceived=True,
        record_all=record_all,
        pi05_prompt=on,
        **switches,
    )
    env.close()
    for key in ("wall_sec", "wall_sec_per_episode"):
        payload.pop(key)
    payload = json.loads(json.dumps(payload, sort_keys=True))
    return payload, served, sent, path.with_suffix(".hdf5"), widths


def test_the_ids_are_the_recorded_subtasks_and_change_no_step_and_no_replan(monkeypatch, tmp_path):
    off, off_served, off_sent, _, _ = _run(monkeypatch, tmp_path, True, False)
    on, on_served, on_sent, shard, on_widths = _run(monkeypatch, tmp_path, True, True)
    bare, _, bare_sent, _, bare_widths = _run(monkeypatch, tmp_path, False, True)  # no recorder
    assert {ids for _, ids in off_sent} == {None}  # off: the key stays off the wire
    assert all(ids.dtype == np.int16 and ids.shape == (1,) for _, ids in on_sent)
    assert [r for r, _ in on_sent] == [r for r, _ in off_sent]  # an id change never replans
    assert on["inference"]["loop"].pop("pi05_prompt") is True
    assert "pi05_prompt" not in off["inference"]["loop"]
    assert on == off and _same(on_served, off_served)
    with h5py.File(shard, "r") as f:
        recorded = f["data/demo_0/obs/subtask"][:].tolist()
    ids = [int(ids[0]) for _, ids in on_sent]
    assert ids == recorded and {0, 1, 2} <= set(ids)  # the teacher's cycles lift
    assert [int(ids[0]) for _, ids in bare_sent] == ids  # the same ids without record_all
    # and the same rule: each command read against the width its frame answered, the last one's
    assert bare_widths == on_widths and len(set(on_widths)) > 1
    assert bare["inference"]["loop"]["pi05_prompt"] is True


def test_the_rollout_refuses_the_switch_without_a_policy():
    env, rig = _env_and_rig()
    with pytest.raises(loop.EvalFailure) as err:
        _rollout(env, rig, None, pi05_prompt=True)
    env.close()
    assert err.value.reason == "pi05_prompt"


def test_the_client_wires_the_switch_and_refuses_it_with_the_oracle_driver(monkeypatch, capsys):
    assert _wiring(monkeypatch, ARGV + ["--pi05_prompt"])["rollout"]["pi05_prompt"] is True
    assert _wiring(monkeypatch, ARGV)["rollout"]["pi05_prompt"] is False
    capsys.readouterr()
    module = _load(EVAL_HEAP)
    with pytest.raises(SystemExit) as exit_info:
        module.main(ARGV + ["--oracle_driver", "--pi05_prompt"])
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("EVAL_FAIL ") and "--pi05_prompt" in out[0]


def test_the_supervisor_forwards_the_switch_splits_the_cache_and_needs_pi05(
    tmp_path, monkeypatch, capsys
):
    read = _argv(tmp_path) + PI05_HEAP + ["--pi05"]
    cfg = ge._build_cfg(ge._parse_args(read + ["--client_flags=--pi05_prompt"]))
    draw = ge.make_eval_draws(1, cfg.seed)[0]
    assert "--pi05_prompt" in ge._eval_command(cfg, draw)
    assert ge.loop_switches(cfg.client_flags)["pi05_prompt"] is True
    ledger = _fake_ledger(tmp_path)
    off = _cfg(tmp_path, ledger, processes=1, pi05=True)
    on = _cfg(tmp_path, ledger, processes=1, pi05=True, client_flags=("--pi05_prompt",))
    off.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, off.seed)[0]

    def cached(gate, **loop_block):
        payload = _payload(draw, [_episode(6)], policy_id="pi05:ck")
        payload["inference"] = {"n_action_steps": 1, "temporal_ensemble_coeff": 0.01}
        payload["inference"] |= {"loop": loop_block} if loop_block else {}
        (off.out_dir / "eval_000.json").write_text(json.dumps(payload))
        return ge._cached_payload(gate, draw) is not None

    assert cached(off) and not cached(on)  # a payload that predates it ran with it off
    assert cached(on, pi05_prompt=True) and not cached(off, pi05_prompt=True)
    monkeypatch.setattr(ge, "_start_server", lambda cfg: pytest.fail("no server"))
    argv = _argv(tmp_path) + PI05_HEAP + ["--client_flags=--pi05_prompt"]
    assert ge.main(argv) == 2
    out = capsys.readouterr()
    assert out.out.strip().splitlines()[-1] == "GATE_ERROR" and "needs --pi05" in out.err
