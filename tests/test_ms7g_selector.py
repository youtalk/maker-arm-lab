"""MS7g design 5.3 and 5.4, the selector switches: ``clear_first`` ranks a grasp that sweeps no
standing neighbour before the height band, and ``overhang_guard`` refuses a part outside the tray
under a tray wall's top edge and keeps the carry grace for a cycle that read a lift. Off, every
step and payload byte is what it was; the supervisor forwards both and keys its cache on them."""

import json
import math

import pytest
import torch

from helpers import load_script
from maker_arm_lab.core import grasp, layout, loop, parts
from maker_arm_lab.core.heap_oracle import HeapOracle
from test_grasp import BRICK, IDENTITY, _heap
from test_mj_heap_oracle import _place
from test_ms7f_stack import _gate_argv
from test_perceived_loop import (
    CARRY_DROP,
    ZERO,
    _env_and_rig,
    _rollout,
    _run,
    _Still,
    _teacher_scene,
    _wall_band_scene,
)
from test_run_gate_eval import _heap_payload, ge
from test_thin_snap import _same

STUD = parts.make_part("brick", 1, 1)
DEEP = layout.DEEP_TRAY_WALL_HEIGHT
WALLS = grasp.tray_walls(30.0, DEEP)
(_, _), (Y0, _) = layout.tray_bounds()
EDGE = Y0 - layout.TRAY_WALL - layout.tray_rim_growth(30.0, DEEP)
"""y of the left wall's outer face at its top: the rim face of the video diagnosis, -0.131."""


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


def _reach(width, band=grasp.JAW_HOLD_BAND):
    return grasp.landing_half_gap(width, band) + grasp.TIP_THICKNESS


def test_the_overhang_band_is_the_wall_shadow_grown_by_the_landing_gap_and_the_tip():
    x, w = 0.222, 0.0078
    # The video diagnosis's part: ldraw_87087 at y -0.135, 4 mm outside the rim face.
    assert grasp.under_overhang(torch.tensor([x, -0.135, 0.0048]), w, WALLS)
    assert grasp.under_overhang(torch.tensor([x, -0.135, 0.0048]), w, WALLS, grasp.CONT_HOLD_BAND)
    for band in (grasp.JAW_HOLD_BAND, grasp.CONT_HOLD_BAND):
        out = EDGE - _reach(w, band)
        assert grasp.under_overhang(torch.tensor([x, out + 1e-4, 0.0048]), w, WALLS, band)
        assert not grasp.under_overhang(torch.tensor([x, out - 1e-4, 0.0048]), w, WALLS, band)
    # A wider grasp lands its fingers further out, so its band is wider.
    out = EDGE - _reach(w)
    assert grasp.under_overhang(torch.tensor([x, out - 0.002, 0.0048]), 0.0158, WALLS)
    # Under the overhang next to the foot, far from the top edge: inside the shadow.
    foot = Y0 - layout.TRAY_WALL - 0.002
    assert EDGE - foot < -_reach(w) and grasp.under_overhang(
        torch.tensor([x, foot, 0.0048]), w, WALLS
    )


def test_the_overhang_band_skips_the_interior_parts_over_the_top_and_box_walls():
    x, w = 0.222, 0.0078
    vertical = grasp.tray_walls(0.0, DEEP)
    inside = torch.tensor([x, Y0 + 0.001, 0.0048])  # 1 mm off the foot, inside the tray
    assert not grasp.under_overhang(inside, w, vertical)
    outside = torch.tensor([x, Y0 - layout.TRAY_WALL - 0.001, 0.0048])
    assert grasp.under_overhang(outside, w, vertical)
    assert not grasp.under_overhang(torch.tensor([x, -0.135, DEEP + 0.001]), w, WALLS)
    # A part resting on the leaned wall's inner face lies in the tray, its centre past the
    # interior's edge on the desk (the diagnostic smoke's refused part, mirrored to this wall).
    leaning = torch.tensor([x, Y0 - 0.0017, 0.013])
    assert not grasp.under_overhang(leaning, 0.0016, WALLS)
    assert grasp.under_overhang(torch.tensor([x, Y0 - 0.0017, 0.0]), 0.0016, WALLS)
    boxed = tuple(grasp.Wall(wall.footprint, wall.heading, wall.top, box=True) for wall in vertical)
    assert not grasp.under_overhang(outside, w, boxed)
    assert not grasp.under_overhang(torch.tensor([x, -0.135, 0.0048]), w, ())


def test_the_grace_holds_only_a_lifted_cycle_under_the_guard():
    for holding in (True, False):
        for lifted in (True, False):
            assert loop.grace_hold(holding, lifted, False) is holding
            assert loop.grace_hold(holding, lifted, True) is (holding and lifted)
    budget = loop.CYCLE_STEP_BUDGET
    clamp = loop.grace_hold(True, False, True)  # a wall clamp the detector reads as a hold
    assert not loop.budget_ends_cycle(budget, loop.grace_hold(True, False, False), 0)
    assert loop.budget_ends_cycle(budget, clamp, 0)
    assert not loop.budget_ends_cycle(budget, loop.grace_hold(True, True, True), 0)


def _overhang_scene(monkeypatch):
    """A 1x1 brick on the desk at the video diagnosis's place, under the 30 degree deep tray's
    left wall top, and two 2x2 bricks 40 mm apart in the open tray. The 1x1 ranks first on its
    clearance, the still policy fails every 60-step cycle, and the perception is exact."""
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 60)
    trio = [STUD, parts.make_part("brick", 2, 2), parts.make_part("brick", 2, 2)]
    env, rig = _env_and_rig(trio, budget=3, tray=True, tray_lean=30.0, deep_tray=True)
    x = layout.TRAY_CENTER[0]
    _place(env, [(0.222, -0.135, 0.0049, 0.0), (x, 0.0, 0.0049, 0.3), (x, 0.04, 0.0049, 0.0)])
    return env, rig, _Still(), 3, {"select_at_release": True, "noise_scales": ZERO}


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


def test_the_guard_refuses_the_part_under_the_overhang_as_a_target(monkeypatch):
    off, _ = _run(_overhang_scene, monkeypatch)
    on, _ = _run(_overhang_scene, monkeypatch, overhang_guard=True)
    assert off["episodes"][0]["cycle_targets"][0] == 0, off["episodes"][0]
    targets = on["episodes"][0]["cycle_targets"]
    assert targets and 0 not in targets, on["episodes"][0]
    assert on["perception"]["overhang_refusals"] == len(targets)
    assert on["inference"]["loop"]["overhang_guard"] is True
    assert "overhang_refusals" not in off["perception"]


@pytest.mark.parametrize("scene", [_wall_band_scene, _overhang_scene])
def test_with_both_switches_off_every_step_and_payload_byte_is_the_same(scene, monkeypatch):
    base, base_served = _run(scene, monkeypatch)
    off, off_served = _run(scene, monkeypatch, clear_first=False, overhang_guard=False)
    assert json.dumps(off, sort_keys=True) == json.dumps(base, sort_keys=True)
    assert _same(off_served, base_served)
    assert not {"clear_first", "overhang_guard"} & set(off["inference"]["loop"])


def test_the_rollout_refuses_the_switches_off_the_perceived_loop_or_the_tray(monkeypatch):
    for kwargs, reason in (
        ({"clear_first": True}, "clear_first"),
        ({"overhang_guard": True}, "overhang_guard"),
        ({"perceived": True, "overhang_guard": True}, "overhang_guard"),  # the bare desk
    ):
        env, rig = _env_and_rig()
        with pytest.raises(loop.EvalFailure) as err:
            _rollout(env, rig, _Still(), **kwargs)
        env.close()
        assert err.value.reason == reason


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--oracle_driver", "--tray", "--clear_first"], "--clear_first needs --perceived"),
        (["--oracle_driver", "--tray", "--overhang_guard"], "--overhang_guard needs --perceived"),
        (["--overhang_guard"], "--overhang_guard reads the tray wall's overhang: it needs --tray"),
    ],
)
def test_the_client_refuses_the_switches_where_no_perceived_selector_reads_them(
    argv, message, capsys
):
    module = load_script("mj_eval_heap")
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--heap_k", "3", "--desk_boxes", "4"] + argv)
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("EVAL_FAIL") and message in out[0], out


def test_the_supervisor_forwards_both_switches_and_keys_the_cache_on_them(tmp_path):
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
    no_tray = [a for a in heap if a != "--tray"]
    with pytest.raises(ValueError, match="--overhang_guard reads the tray wall's overhang"):
        ge._build_cfg(ge._parse_args(argv + no_tray + ["--overhang_guard"]))


def test_the_client_parses_the_three_ms7g_switches_the_supervisor_forwards(tmp_path):
    """The stack serves ``--carry_guard`` with the selector switches: the supervisor forwards
    each once and keys its cache on each, and the client parses the command it forwards."""
    trio = ["--carry_guard", "--overhang_guard", "--clear_first"]
    argv = _gate_argv(tmp_path) + ["--tray", "--gripper", "continuous", "--perceived"]
    cfg = ge._build_cfg(ge._parse_args(argv + trio))
    cmd = ge._eval_command(cfg, ge.make_eval_draws(1, 1)[0])
    assert all(cmd.count(flag) == 1 for flag in trio), cmd
    on = {name for name, value in ge.loop_switches(cfg.client_flags).items() if value is True}
    assert on == {"perceived", "carry_guard", "overhang_guard", "clear_first"}
    client = load_script("mj_eval_heap")
    args = client.parse_args(cmd[cmd.index("scripts/mj_eval_heap.py") + 1 :])
    assert args.carry_guard and args.overhang_guard and args.clear_first


def _tray_carry_scene(monkeypatch):
    """The carry guard's carry scene on the flat tray: its parts lie inside the tray interior,
    and its teacher opens over them, away from every box."""
    env, rig, policy, budget, switches = _teacher_scene(monkeypatch, tray=True)
    policy.rig.oracles[0].release_point = torch.tensor(CARRY_DROP)
    return env, rig, policy, budget, switches | {"carry_guard": True}


def test_the_overhang_guard_changes_no_step_of_a_carry_the_carry_guard_holds(monkeypatch):
    """``overhang_guard`` drops the carry grace only for a hold that read no lift (``grace_hold``),
    and ``carry_guard`` guards only a hold that did. With no part under a wall, the guard hands
    the carry to the rig and the rig places it, with the overhang guard on as off."""
    off, off_served = _run(_tray_carry_scene, monkeypatch)
    on, on_served = _run(_tray_carry_scene, monkeypatch, overhang_guard=True)
    assert "placed" in [row["handoff"] for row in on["episodes"][0]["cycle_census"]], on
    assert on["inference"]["loop"].pop("overhang_guard") is True
    assert on["perception"].pop("overhang_refusals") == 0
    assert json.dumps(on, sort_keys=True) == json.dumps(off, sort_keys=True)
    assert _same(on_served, off_served)
