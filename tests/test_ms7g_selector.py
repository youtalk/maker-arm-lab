"""MS7g design 5.3, the selector switch ``clear_first``: it ranks a grasp that sweeps no standing
neighbour before the height band. Off, every step and payload byte is what it was; the supervisor
forwards it and keys its cache on it."""

import json
import math

import pytest
import torch

from maker_arm_lab.core import grasp, loop, parts
from maker_arm_lab.core.heap_oracle import HeapOracle
from test_grasp import BRICK, IDENTITY, _heap
from test_ms7f_stack import ROOT, _load
from test_perceived_loop import _env_and_rig, _rollout, _served, _Still, _wall_band_scene
from test_run_gate_eval import _heap_payload, ge
from test_thin_snap import _same

STUD = parts.make_part("brick", 1, 1)


def _old_key(g):
    return (-math.floor(g.height / grasp.HEIGHT_BAND), g.swept, -g.clearance, g.width)


def _swept_top_and_clear_desk():
    """A 2x4 brick in the top band whose two strokes each sweep a standing 1x1 brick (parts 1
    and 2, not candidates), and a lone 2x4 brick on the desk, two bands lower."""
    gap = grasp.landing_half_gap(0.0158)
    rows = (
        (BRICK, 0.19, 0.0, 0.0144, IDENTITY),
        (STUD, 0.19 + 0.0159 + 0.006, 0.0, 0.0144, IDENTITY),  # in the x stroke
        (STUD, 0.19, 0.0079 + 0.5 * (gap - 0.0079), 0.0144, IDENTITY),  # in the y stroke
        (BRICK, 0.25, -0.06, 0.0048, IDENTITY),
    )
    mask = torch.tensor([True, False, False, True])
    return (*_heap(*rows), mask)


def test_clear_first_ranks_a_clear_grasp_in_a_lower_band_before_a_swept_one():
    sizes, pos, quat, mask = _swept_top_and_clear_desk()
    off = grasp.rank_grasps(sizes, pos, quat, mask)
    on = grasp.rank_grasps(sizes, pos, quat, mask, clear_first=True)
    assert all(g.swept >= 1 for g in off if g.part == 0) and {g.part for g in off} == {0, 3}
    assert off[0].part == 0 and grasp.select_grasp(off).part == 0  # the band first
    assert on[0].part == 3 and grasp.select_grasp(on).part == 3  # no sweep first
    assert [g.swept > 0 for g in on] == sorted(g.swept > 0 for g in on)
    # Off, the order is main's key exactly; on, within each sweep class it still is.
    assert off == sorted(off, key=_old_key)
    for swept in (False, True):
        part = [g for g in on if (g.swept > 0) is swept]
        assert part == sorted(part, key=_old_key)


def test_clear_first_leaves_a_one_part_ranking_alone():
    # The served row, the close threshold and a regrasp rank the target alone: one band.
    sizes, pos, quat, _ = _swept_top_and_clear_desk()
    alone = torch.tensor([True, False, False, False])
    assert grasp.rank_grasps(sizes, pos, quat, alone, clear_first=True) == grasp.rank_grasps(
        sizes, pos, quat, alone
    )


def _run(scene, monkeypatch, **switch):
    """The scene's perceived rollout without its wall clock, and what env 0 was served."""
    env, rig, policy, budget, switches = scene(monkeypatch)
    served = _served(env)
    payload = _rollout(env, rig, policy, budget=budget, perceived=True, **switches, **switch)
    env.close()
    for key in ("wall_sec", "wall_sec_per_episode"):
        payload.pop(key)
    return payload, served


def test_the_loop_hands_clear_first_to_its_selector_and_not_to_the_teacher(monkeypatch):
    # The perceived selector knows no destination; the teacher policy's own rig ranks too.
    seen = set()
    real = HeapOracle._ranked

    def spy(self, scene, mask):
        seen.add((self.knows_destinations, self.clear_first))
        return real(self, scene, mask)

    monkeypatch.setattr(HeapOracle, "_ranked", spy)
    on, _ = _run(_wall_band_scene, monkeypatch, clear_first=True)
    assert seen == {(False, True), (True, False)}
    assert on["inference"]["loop"]["clear_first"] is True


def test_with_the_switch_off_every_step_and_payload_byte_is_the_same(monkeypatch):
    base, base_served = _run(_wall_band_scene, monkeypatch)
    off, off_served = _run(_wall_band_scene, monkeypatch, clear_first=False)
    assert json.dumps(off, sort_keys=True) == json.dumps(base, sort_keys=True)
    assert _same(off_served, base_served)
    assert "clear_first" not in off["inference"]["loop"]


def test_the_rollout_refuses_the_switch_off_the_perceived_loop(monkeypatch):
    env, rig = _env_and_rig()
    with pytest.raises(loop.EvalFailure) as err:
        _rollout(env, rig, _Still(), clear_first=True)
    env.close()
    assert err.value.reason == "clear_first"


@pytest.mark.parametrize(
    "argv, message",
    [(["--oracle_driver", "--tray", "--clear_first"], "--clear_first needs --perceived")],
)
def test_the_client_refuses_the_switches_where_no_perceived_selector_reads_them(
    argv, message, capsys
):
    module = _load(ROOT / "scripts" / "mj_eval_heap.py")
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--heap_k", "3", "--desk_boxes", "4"] + argv)
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("EVAL_FAIL") and message in out[0], out


def test_the_supervisor_forwards_the_switches_and_keys_the_cache_on_them(tmp_path):
    argv = ["--checkpoint", "ck", "--out_dir", str(tmp_path / "gate"), "--ledger", "l"]
    argv += ["--threshold", "0.7", "--seed", "1"]
    heap = ["--backend", "mujoco", "--heap_k", "50", "--tray", "--desk_boxes", "4"]
    heap += ["--gripper", "continuous", "--perceived"]
    draw = ge.make_eval_draws(1, 1)[0]
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None}
    base = {name: False for name in ge.LOOP_SWITCHES} | {"perceived": True}
    base["noise_scales"] = ge.noise_scales_key("")
    path = tmp_path / "gate" / "eval_000.json"
    path.parent.mkdir()

    def hit(cfg, loop_block):
        inference = {**served, "loop": loop_block}
        payload = _heap_payload(
            draw, tray=True, desk_boxes=4, gripper="continuous", inference=inference
        )
        path.write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    plain = ge._build_cfg(ge._parse_args(argv + heap))
    assert not {f"--{name}" for name in ge.MS7G_SWITCHES} & set(ge._eval_command(plain, draw))
    # A payload that predates the switches, or records them off by leaving them out, is off.
    old = {k: v for k, v in base.items() if k not in ge.MS7G_SWITCHES}
    assert hit(plain, base) and hit(plain, old)
    for name in ge.MS7G_SWITCHES:
        cfg = ge._build_cfg(ge._parse_args(argv + heap + [f"--{name}"]))
        assert f"--{name}" in ge._eval_command(cfg, draw)
        assert hit(cfg, old | {name: True}) and not hit(cfg, old)
        assert not hit(plain, old | {name: True})
        with pytest.raises(ValueError, match=f"--{name} needs --perceived"):
            ge._build_cfg(ge._parse_args(argv + heap[:-1] + [f"--{name}"]))
        with pytest.raises(ValueError, match=f"--{name} needs --heap_k"):
            ge._build_cfg(ge._parse_args(argv + ["--backend", "isaac", f"--{name}"]))
