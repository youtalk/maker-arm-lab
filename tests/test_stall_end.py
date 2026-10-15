"""MS7f ``stall_end``: a served cycle whose jaws read closed for ``CLOSED_NO_PINCH_STEPS`` steps
in a row while the hold detector holds nothing and the cycle has lifted nothing ends as a failure,
booked as a budget end. Off, every step and payload byte is what it was."""

import json

import pytest

from helpers import load_script
from maker_arm_lab.core import dagger_loop, loop, perception
from test_ms7f_stack import _gate_argv, ge
from test_perceived_loop import _rollout, _served, _teacher_scene, _thin_snap_scene
from test_snap_closed import _Blind
from test_thin_snap import _same

STALL = dagger_loop.CLOSED_NO_PINCH_STEPS


def test_the_stall_end_rule():
    assert loop.CLOSED_NO_PINCH_STEPS is STALL == 60  # the recorder's constant, not a copy
    n = 0
    for _ in range(STALL - 1):
        n = loop.closed_empty_count(n, grip_open=False, holding=False)
    assert n == STALL - 1 and not loop.stall_ends(n, lifted=False, holding=False)
    n = loop.closed_empty_count(n, grip_open=False, holding=False)
    assert n == STALL and loop.stall_ends(n, lifted=False, holding=False)
    assert not loop.stall_ends(n, lifted=False, holding=True)  # it holds a part now
    assert not loop.stall_ends(n, lifted=True, holding=False)  # the cycle lifted one
    assert loop.closed_empty_count(n, grip_open=True, holding=False) == 0  # an open read
    assert loop.closed_empty_count(n, grip_open=False, holding=True) == 0  # a hold


def _run(scene, monkeypatch, **switch):
    """The scene's perceived rollout: the payload without its wall clock, and what env 0 was
    served on every step."""
    env, rig, policy, budget, switches = scene(monkeypatch)
    served = _served(env)
    payload = _rollout(env, rig, policy, budget=budget, perceived=True, **switches, **switch)
    env.close()
    for key in ("wall_sec", "wall_sec_per_episode"):
        payload.pop(key)
    return json.loads(json.dumps(payload, sort_keys=True)), served


def test_with_the_switch_off_every_step_and_payload_byte_is_mains(monkeypatch):
    main, main_served = _run(_thin_snap_scene, monkeypatch)
    off, off_served = _run(_thin_snap_scene, monkeypatch, stall_end=False)
    assert off == main and _same(off_served, main_served)
    assert "stall_end" not in off["inference"]["loop"]
    assert "stall_end" not in off["episodes"][0]["cycle_reasons"]


def test_a_cycle_that_holds_its_part_is_never_ended(monkeypatch):
    """The teacher scene: every cycle holds and lifts its part in the detector's eyes, so the
    switch changes no step and no payload byte but its own flag."""
    off, off_served = _run(_teacher_scene, monkeypatch)
    on, on_served = _run(_teacher_scene, monkeypatch, stall_end=True)
    assert on["inference"]["loop"].pop("stall_end") is True
    assert on == off and _same(on_served, off_served)


def test_jaws_closed_on_nothing_end_the_cycle_as_a_budget_failure(monkeypatch):
    """A detector that never reads a hold: off, every cycle of the teacher scene runs to its
    800-step budget; on, each ends ``stall_end`` after its 60th closed step, and each is booked
    in the failure memory at its target's position as a budget end is."""
    monkeypatch.setattr(loop, "HoldDetector", _Blind)
    off, _ = _run(_teacher_scene, monkeypatch)
    booked = []
    real = perception.PositionMemory.book
    monkeypatch.setattr(
        perception.PositionMemory, "book", lambda self, xy, n=1: booked.append(xy) or real(self, xy)
    )
    on, _ = _run(_teacher_scene, monkeypatch, stall_end=True)
    off_ep, on_ep = off["episodes"][0], on["episodes"][0]
    assert set(off_ep["cycle_reasons"]) == {"budget"}, off_ep["cycle_reasons"]
    assert on_ep["cycle_reasons"] and set(on_ep["cycle_reasons"]) == {"stall_end"}
    assert {row["closed_steps"] for row in on_ep["cycle_census"]} == {STALL}
    assert not any(on_ep["cycle_outcomes"]) and "stowed" not in on_ep["cycle_classes"]
    assert len(booked) == on_ep["cycles"] == len(on_ep["cycle_census"])
    assert sum(on_ep["cycle_steps"]) < sum(off_ep["cycle_steps"])
    assert on["inference"]["loop"]["stall_end"] is True


def test_the_rollout_refuses_the_switch_without_perceived(monkeypatch):
    env, rig, policy, budget, _ = _teacher_scene(monkeypatch)
    with pytest.raises(loop.EvalFailure) as err:
        _rollout(env, rig, policy, budget=budget, stall_end=True)
    env.close()
    assert err.value.reason == "stall_end"


@pytest.mark.parametrize(
    "script, argv, message",
    [
        ("mj_eval_heap", ["--oracle_driver"], "--stall_end needs --perceived"),
        ("mj_generate_batch", ["--out_dir", "/tmp/x"], "--stall_end cuts a served cycle"),
    ],
)
def test_the_clients_refuse_the_switch_where_no_serving_loop_reads_it(
    script, argv, message, capsys
):
    module = load_script(script)
    with pytest.raises(SystemExit) as exit_info:
        module.main(argv + ["--heap_k", "3", "--desk_boxes", "4", "--stall_end"])
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and "_FAIL" in out[0] and message in out[0], out


def test_the_supervisor_forwards_the_switch_and_its_cache_key_records_it(tmp_path):
    assert ge.MS7F_OFF["stall_end"] is False and "stall_end" in ge.MS7F_READY
    draw = ge.make_eval_draws(1, 1)[0]
    argv = _gate_argv(tmp_path)
    plain = ge._build_cfg(ge._parse_args(argv))
    cfg = ge._build_cfg(ge._parse_args(argv + ["--stall_end"]))
    assert ge.ms7f_block(cfg) == ge.MS7F_OFF | {"stall_end": True}
    assert ge._eval_command(cfg, draw).count("--stall_end") == 1
    assert "--stall_end" not in ge._eval_command(plain, draw)
