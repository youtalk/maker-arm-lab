"""MS7g ``carry_guard`` (design 5.2): the guard on a lifted part's release, its margin, the rig's
perceived carry of a still guard, the refusals and the switch-off pin. The no-truth test of the
carry entry is ``test_perceived_loop.test_no_decision_reads_the_truth`` (``_carry_scene``)."""

import json
import math
from typing import ClassVar

import pytest
import torch

from maker_arm_lab.core import desk_boxes, grasp, loop, parts
from maker_arm_lab.core.hold import MIN_EXCESS
from test_classifier_loop import SWITCHES, _classifier, _scene
from test_ms7f_stack import ROOT, _gate_argv, _load, ge
from test_perceived_loop import ZERO, _carry_scene, _rollout, _served, _teacher_scene
from test_run_gate_eval import _heap_payload
from test_thin_snap import _same

BOX = (0.20, 0.21, math.radians(30.0))
SIZE = (0.09, 0.07, 0.04)


def _at(u: float, v: float) -> tuple[float, float]:
    """The world xy of the point (u, v) in BOX's own frame."""
    c, s = math.cos(BOX[2]), math.sin(BOX[2])
    return BOX[0] + c * u - s * v, BOX[1] + s * u + c * v


def test_the_guard_margin_is_read_in_the_yawed_box_frame():
    margin = loop.GUARD_BOX_MARGIN_M
    assert margin == 0.010
    for u, v, inside, deep in (
        (0.0, 0.0, True, True),
        (0.045 - 0.011, 0.0, True, True),  # 11 mm off the long wall
        (0.045 - 0.009, 0.0, True, False),  # 9 mm off it: the rim band
        (0.0, 0.035 - 0.009, True, False),  # the short wall's band
        (0.046, 0.0, False, False),  # over the wall
        (0.0, 0.040, False, False),  # past the short wall, within the long one's reach
    ):
        xy = _at(u, v)
        assert loop.in_box_interior(xy, BOX, SIZE) is inside, (u, v)
        assert loop.in_box_interior(xy, BOX, SIZE, margin) is deep, (u, v)


def test_the_guard_serves_no_command_that_would_end_a_lifted_hold_away_from_the_box():
    measured, last = 0.0158, 0.0120  # the last command looser than the squeeze
    squeeze = grasp.close_gap(measured)
    assert squeeze == pytest.approx(measured - 0.006)
    closing = measured - MIN_EXCESS  # the widest command that keeps the hold
    assert loop.guarded_gap(closing, measured, last, None) == (closing, None)
    assert loop.guarded_gap(0.001, measured, last, None) == (0.001, None)  # a tighter squeeze
    for opening in (closing + 1e-4, measured, 0.1048):
        # The run's gap is the tighter of the last command and the teacher's squeeze.
        assert loop.guarded_gap(opening, measured, last, None) == (squeeze, squeeze)
        assert loop.guarded_gap(opening, measured, 0.004, None) == (0.004, 0.004)
    assert loop.guarded_gap(0.1048, 0.004, 0.003, None) == (0.0, 0.0)  # a narrow part: shut
    # The run's gap stays fixed while the jaws close on a part that slipped out, so the measured
    # gap reaches it and the detector's hold ends; a command that keeps the hold ends the run.
    held = squeeze
    for slipping in (0.0150, 0.0120, 0.0100, held):
        assert loop.guarded_gap(0.1048, slipping, held, held) == (held, held)
    assert loop.guarded_gap(0.002, 0.0158, held, held) == (0.002, None)


def test_the_carry_release_height_clears_the_rim_from_the_holds_start():
    deep, shallow = (0.09, 0.09, 0.05), (0.09, 0.09, 0.03)
    assert desk_boxes.rim_z(deep) == pytest.approx(0.055)
    assert loop.carry_release_z(0.006, deep) == desk_boxes.DESK_BOX_RELEASE_Z == 0.10
    assert loop.carry_release_z(0.035, deep) == pytest.approx(0.10)
    assert loop.carry_release_z(0.05, deep) == pytest.approx(0.115)
    assert loop.carry_release_z(0.05, shallow) == pytest.approx(0.10)
    assert loop.carry_release_z(0.30, deep) == loop.CARRY_RELEASE_MAX_Z == 0.12


def _run(scene, monkeypatch, **switch):
    """The scene's perceived rollout: the payload without its wall clock, and what env 0 was
    served on every step."""
    env, rig, policy, budget, switches = scene(monkeypatch)
    switches |= switch
    served = _served(env)
    payload = _rollout(env, rig, policy, budget=budget, perceived=True, **switches)
    env.close()
    for key in ("wall_sec", "wall_sec_per_episode"):
        payload.pop(key)
    return json.loads(json.dumps(payload, sort_keys=True)), served


def test_with_the_switch_off_every_step_and_payload_byte_is_the_same(monkeypatch):
    main, main_served = _run(_teacher_scene, monkeypatch)
    off, off_served = _run(_teacher_scene, monkeypatch, carry_guard=False)
    assert off == main and _same(off_served, main_served)
    assert "carry_guard" not in off["inference"]["loop"]
    rows = off["episodes"][0]["cycle_census"]
    assert rows and not set(loop.CARRY_GUARD_CENSUS) & set(rows[0])


def test_a_teacher_that_releases_inside_its_box_is_never_guarded(monkeypatch):
    """At zero perception noise the served box is the teacher's own, and its RELEASE opens over
    the box centre: the switch changes no step and no payload byte but its flag and its fields."""
    off, off_served = _run(_teacher_scene, monkeypatch, noise_scales=ZERO)
    on, on_served = _run(_teacher_scene, monkeypatch, noise_scales=ZERO, carry_guard=True)
    assert on["inference"]["loop"].pop("carry_guard") is True
    rows = on["episodes"][0]["cycle_census"]
    assert rows and all(row.pop("guard_steps") == 0 for row in rows)
    assert all(row.pop("guard_rim_steps") == 0 and row.pop("handoff") is None for row in rows)
    assert all(row.pop("stall_open") is False and row.pop("guard_vote_steps") == 0 for row in rows)
    assert on == off and _same(on_served, off_served)


def test_a_drop_away_from_the_box_is_held_and_the_rigs_carry_places_it(monkeypatch):
    """The carry scene's teacher opens over the parts it picks from. Off, every lifted cycle
    ends ``dropped``; on, the guard holds each open for GUARD_STALL_STEPS, the rig carries the
    part to the served box and the cycle ends ``placed`` with handoff "placed"."""
    off, _ = _run(_carry_scene, monkeypatch, carry_guard=False)
    on, _ = _run(_carry_scene, monkeypatch)
    off_ep, on_ep = off["episodes"][0], on["episodes"][0]
    assert "dropped" in off_ep["cycle_reasons"] and "placed" not in off_ep["cycle_reasons"]
    rows = on_ep["cycle_census"]
    handed = [row for row in rows if row["handoff"] is not None]
    assert handed and {row["handoff"] for row in handed} == {"placed"}, rows
    assert all(row["guard_steps"] >= loop.GUARD_STALL_STEPS for row in handed)
    reasons = [on_ep["cycle_reasons"][j] for j, row in enumerate(rows) if row["handoff"]]
    assert set(reasons) == {"placed"}
    assert on_ep["bricks_stowed"] > off_ep["bricks_stowed"]
    assert on["inference"]["loop"]["carry_guard"] is True


class _Stalled(loop.AirStall):
    """The census's air stall, stalled on every step (the scenes' arm is never over a tray)."""

    def update(self, pick, over_tray):
        self.stalled = True
        return True


def test_an_air_stall_with_nothing_held_opens_the_jaws_and_fails_the_cycle(monkeypatch):
    """A policy that holds still with open jaws: off, every cycle runs to its step budget; on,
    each ends ``stall_open`` once the stall has lasted AIR_STALL_STEPS steps of the cycle, the
    jaws are served fully open until the next selection, and the target is booked."""
    monkeypatch.setattr(loop, "AirStall", _Stalled)
    off, _ = _run(_still_scene, monkeypatch, carry_guard=False)
    on, _ = _run(_still_scene, monkeypatch)
    off_ep, on_ep = off["episodes"][0], on["episodes"][0]
    assert set(off_ep["cycle_reasons"]) == {"budget"}
    assert on_ep["cycle_reasons"] and set(on_ep["cycle_reasons"]) == {"stall_open"}
    assert set(on_ep["cycle_steps"]) == {loop.AIR_STALL_STEPS}
    assert all(row["stall_open"] and row["handoff"] is None for row in on_ep["cycle_census"])
    assert on_ep["cycle_classes"] and set(on_ep["cycle_classes"]) == {"no_close"}


def _still_scene(monkeypatch):
    env, rig, _, budget, switches = _teacher_scene(monkeypatch)
    from test_perceived_loop import _Still

    return env, rig, _Still(), budget, switches | {"carry_guard": True}


class _StalledWhileLifted(loop.AirStall):
    """Stalled while the loop's hold detector holds a part its hold lifted."""

    def update(self, pick, over_tray):
        d = _Detector.made[0]
        self.stalled = d.holding and d.lifted
        return self.stalled


class _Detector(loop.HoldDetector):
    made: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _Detector.made.append(self)


def test_an_air_stall_with_a_held_part_hands_it_to_the_rig(monkeypatch):
    """A stall read while the detector holds a lifted part hands the part to the rig with no
    guard step. At zero noise the first carry is the teacher's own and the rig places it; a
    later one starts beside box 2 bound for box 1 across the base, where only the leg by way of
    ``CARRY_VIA_POINT`` plans (it was declined without it), and the jaws it takes are already
    empty there, so it ends lifted_lost."""
    _Detector.made = []
    monkeypatch.setattr(loop, "HoldDetector", _Detector)
    monkeypatch.setattr(loop, "AirStall", _StalledWhileLifted)
    on, _ = _run(_teacher_scene, monkeypatch, noise_scales=ZERO, carry_guard=True)
    rows = on["episodes"][0]["cycle_census"]
    handed = [row for row in rows if row["handoff"] is not None]
    assert [row["handoff"] for row in handed] == ["placed", "lifted_lost"], rows
    assert all(row["guard_steps"] == 0 for row in handed)


class _Hovering(loop.AirStall):
    """An air stall whose run grows by one each step from the first, stalled from
    AIR_STALL_STEPS on (the scenes' arm is never over a tray)."""

    def update(self, pick, over_tray):
        self.run += 1
        self.stalled = self.run >= loop.AIR_STALL_STEPS
        return self.stalled


class _HoveringWhileHeld(loop.AirStall):
    """The same run, but only while the loop's hold detector holds a part."""

    def update(self, pick, over_tray):
        self.run = self.run + 1 if _Detector.made[-1].holding else 0
        self.stalled = self.run >= loop.AIR_STALL_STEPS
        return self.stalled


def _asks(scene):
    """The scene with ``replan_on_switch`` on, and env 0's replan flag, retry flag (None off the
    wire) and the last ``_Detector``'s hold on every policy step; the policy never sees the retry
    flag."""

    def build(monkeypatch):
        env, rig, policy, budget, switches = scene(monkeypatch)
        asks, retries, held, real = [], [], [], policy.step

        def step(*args, replan=None, retry=None, **kwargs):
            asks.append(bool(replan[0]))
            retries.append(None if retry is None else bool(retry[0]))
            held.append(bool(_Detector.made) and _Detector.made[-1].holding)
            return real(*args, replan=replan, **kwargs)

        policy.step = step
        build.asks, build.retries, build.held = asks, retries, held
        return env, rig, policy, budget, switches | {"replan_on_switch": True}

    return build


def test_retry_noise_is_due_once_the_run_reaches_n_with_nothing_held():
    due = [run for run in range(100) if loop.retry_noise_due(run, 20, holding=False)]
    assert due == list(range(20, 100))  # a hover already past N at a cycle start enters at once
    assert not any(loop.retry_noise_due(run, 20, holding=True) for run in range(100))
    assert not any(loop.retry_noise_due(run, 0, holding=False) for run in range(100))


def _retry_run(scene, monkeypatch, **switch):
    """The payload, env 0's replan and retry flags and its holds of a perceived rollout of
    ``_asks(scene)``."""
    _Detector.made = []
    build = _asks(scene)
    payload, _ = _run(build, monkeypatch, **switch)
    return payload, build.asks, build.retries, build.held


def _true_runs(flags: list) -> list[tuple[int, int]]:
    """(first index, length) of each run of True in ``flags``."""
    runs, start = [], None
    for t, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            runs.append((start, t - start))
            start = None
    return runs


def test_a_hover_with_nothing_held_flags_the_row_for_retry_noise_to_its_cycle_end(monkeypatch):
    """A policy that holds still with open jaws over the tray, its air stall's run never reset:
    on, the first cycle enters retry mode at run N (cycle step 20), and each later cycle, which
    starts out of retry mode, enters it on its first step. The loop sends the retry flag from the
    step after the entry to the cycle's last step, and sets the replan flag on that first step
    only; off, the retry flag stays off the wire. The policy ignores both flags, so every other
    step and payload byte is the same, and the census records each entry's cycle step."""
    monkeypatch.setattr(loop, "AirStall", _Hovering)
    off, off_asks, off_retries, _ = _retry_run(_still_scene, monkeypatch)
    on, on_asks, on_retries, _ = _retry_run(_still_scene, monkeypatch, retry_noise_at=20)
    assert set(off_retries) == {None} and None not in on_retries
    assert len(on_asks) == len(off_asks) == len(on_retries)
    assert on["inference"]["loop"].pop("retry_noise_at") == 20
    assert "retry_noise_at" not in off["inference"]["loop"]
    ep = on["episodes"][0]
    entries = [row.pop("retry_at") for row in ep["cycle_census"]]
    assert len(entries) > 2 and entries == [20] + [0] * (len(entries) - 1)
    assert set(ep["cycle_reasons"]) == {"stall_open"}  # the stall still opens the jaws
    runs = _true_runs(on_retries)
    assert [n for _, n in runs] == [steps - at for steps, at in zip(ep["cycle_steps"], entries)]
    assert runs[0][0] == 21  # the entry at the step whose run read 20, flagged from the next
    assert [s for s, (a, b) in enumerate(zip(on_asks, off_asks)) if a != b] == [t for t, _ in runs]
    assert on == off and "retry_at" not in off["episodes"][0]["cycle_census"][0]


def test_a_hover_while_a_part_is_held_enters_retry_mode_only_once_the_hold_ends(monkeypatch):
    """The same run while the detector holds a part: the stall still hands the part to the rig,
    and those cycles never enter retry mode, though their run passed N. A hold that ends while
    the run is past N leaves the arm hovering with nothing held, so that cycle enters it on the
    step the hold ended. The teacher policy ignores the flags: no payload byte changes."""
    monkeypatch.setattr(loop, "HoldDetector", _Detector)
    monkeypatch.setattr(loop, "AirStall", _HoveringWhileHeld)
    switches = {"noise_scales": ZERO, "carry_guard": True}
    off, off_asks, _, _ = _retry_run(_teacher_scene, monkeypatch, **switches)
    on, on_asks, on_retries, held = _retry_run(
        _teacher_scene, monkeypatch, **switches, retry_noise_at=20
    )
    rows = on["episodes"][0]["cycle_census"]
    handed = [row for row in rows if row["handoff"]]
    assert handed and all(row["retry_at"] is None for row in handed), rows  # the held stalls ran
    entries = [row.pop("retry_at") for row in rows]
    runs = _true_runs(on_retries)
    assert len(runs) == len([at for at in entries if at is not None])
    assert all(held[s - 2] and not held[s - 1] for s, _ in runs)  # entered as the hold ended
    assert on["inference"]["loop"].pop("retry_noise_at") == 20
    assert [s for s, (a, b) in enumerate(zip(on_asks, off_asks)) if a != b] == [s for s, _ in runs]
    assert on == off


def test_the_rollout_refuses_a_retry_noise_at_without_its_guard_or_its_flag(monkeypatch):
    both = {"carry_guard": True, "replan_on_switch": True}
    for kwargs in (
        {"retry_noise_at": 20, "replan_on_switch": True},
        {"retry_noise_at": 20, "carry_guard": True},
        {"retry_noise_at": -1, **both},
        {"retry_noise_at": loop.AIR_STALL_STEPS, **both},  # the stall opens the jaws there
    ):
        env, rig, policy, budget, _ = _teacher_scene(monkeypatch)
        with pytest.raises(loop.EvalFailure) as err:
            _rollout(env, rig, policy, budget=budget, perceived=True, **kwargs)
        env.close()
        assert err.value.reason == "retry_noise_at", kwargs


def test_a_margin_past_the_boxs_half_width_guards_the_teachers_centre_release(monkeypatch):
    """At zero noise the teacher opens over its box centre, so the default margin, passed or not,
    guards nothing and changes no payload byte. A 0.2 m margin leaves no point deep enough: the
    guard holds each open, its steps over the box count as rim steps, and the payload records
    the margin."""
    switches = {"noise_scales": ZERO, "carry_guard": True}
    off, off_served = _run(_teacher_scene, monkeypatch, **switches)
    same, same_served = _run(
        _teacher_scene, monkeypatch, **switches, guard_margin=loop.GUARD_BOX_MARGIN_M
    )
    assert same == off and _same(same_served, off_served)
    wide, _ = _run(_teacher_scene, monkeypatch, **switches, guard_margin=0.2)
    assert wide["inference"]["loop"]["guard_margin"] == 0.2
    assert "guard_margin" not in off["inference"]["loop"]
    rows = wide["episodes"][0]["cycle_census"]
    assert any(row["guard_rim_steps"] for row in rows), rows
    for kwargs in ({"guard_margin": 0.02}, {"guard_margin": -0.01, "carry_guard": True}):
        env, rig, policy, budget, _ = _teacher_scene(monkeypatch)
        with pytest.raises(loop.EvalFailure) as err:
            _rollout(env, rig, policy, budget=budget, perceived=True, **kwargs)
        env.close()
        assert err.value.reason == "guard_margin", kwargs


def test_only_a_guard_step_on_the_rim_band_hands_the_carry_off_at_once():
    margin = 0.025
    rim, deep, out = _at(0.045 - 0.020, 0.0), _at(0.0, 0.0), _at(0.0, 0.040)
    assert loop.rim_handoff_due(True, True, rim, BOX, SIZE, margin)
    for on, guarded, xy in (
        (True, True, deep),  # deep enough: the guard lets the jaws open
        (True, True, out),  # off the interior: the guard holds the jaws to its stall
        (True, False, rim),  # no guard step, or a pending vote's
        (False, True, rim),  # the switch off
    ):
        assert loop.rim_handoff_due(on, guarded, xy, BOX, SIZE, margin) is False, (on, guarded)


def _without_rim(payload: dict) -> dict:
    """The payload less the switch's flag and census field."""
    assert payload["inference"]["loop"].pop("release_handoff") is True
    for row in payload["episodes"][0]["cycle_census"]:
        row.pop("release_handoff")
    return payload


def test_a_release_on_the_rim_band_goes_to_the_rig_at_once(monkeypatch):
    """At zero noise the teacher opens over its box centre. Under a 0.2 m margin each such open is
    a rim step: off, the guard holds it GUARD_STALL_STEPS steps before the stall hands the part to
    the rig; on, the first rim step hands it over, the rig places it, and the episode ends
    all_stowed during the last carry. At the default margin the centre is deep, so the switch
    changes no step and no payload byte but its own two."""
    switches = {"noise_scales": ZERO, "carry_guard": True}
    off, _ = _run(_teacher_scene, monkeypatch, **switches, guard_margin=0.2)
    on, _ = _run(_teacher_scene, monkeypatch, **switches, guard_margin=0.2, release_handoff=True)
    assert "release_handoff" not in off["inference"]["loop"]
    off_handed = [row for row in off["episodes"][0]["cycle_census"] if row["handoff"]]
    assert off_handed and all(row["guard_steps"] >= loop.GUARD_STALL_STEPS for row in off_handed)
    assert "release_handoff" not in off_handed[0]
    ep = on["episodes"][0]
    rows = ep["cycle_census"]
    assert all(row["release_handoff"] and row["guard_steps"] == 1 for row in rows), rows
    assert all(row["guard_rim_steps"] == 1 for row in rows)
    assert [row["handoff"] for row in rows] == ["placed"] * (len(rows) - 1) + ["carrying"]
    assert ep["done_reason"] == "all_stowed"
    assert ep["bricks_stowed"] > off["episodes"][0]["bricks_stowed"]
    assert on["inference"]["loop"]["release_handoff"] is True
    base, base_served = _run(_teacher_scene, monkeypatch, **switches)
    same, same_served = _run(_teacher_scene, monkeypatch, **switches, release_handoff=True)
    assert not any(row["release_handoff"] for row in same["episodes"][0]["cycle_census"])
    assert _without_rim(same) == base and _same(same_served, base_served)


def test_a_drop_off_every_box_keeps_the_stall_with_the_switch_on(monkeypatch):
    """The carry scene's teacher opens away from every box: no rim step, so the guard holds each
    open to its stall as without the switch, and only the switch's two payload entries differ."""
    off, off_served = _run(_carry_scene, monkeypatch)
    on, on_served = _run(_carry_scene, monkeypatch, release_handoff=True)
    assert not any(row["release_handoff"] for row in on["episodes"][0]["cycle_census"])
    assert _without_rim(on) == off and _same(on_served, off_served)


def test_the_rollout_and_the_client_refuse_the_release_handoff_without_the_guard(
    monkeypatch, capsys
):
    env, rig, policy, budget, _ = _teacher_scene(monkeypatch)
    with pytest.raises(loop.EvalFailure) as err:
        _rollout(env, rig, policy, budget=budget, perceived=True, release_handoff=True)
    env.close()
    assert err.value.reason == "release_handoff"
    module = _load(ROOT / "scripts" / "mj_eval_heap.py")
    with pytest.raises(SystemExit) as exit_info:
        module.main(HEAP + ["--perceived", "--release_handoff"])
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("EVAL_FAIL ") and "--carry_guard" in out[0], out


def test_a_pending_vote_serves_no_box_so_the_part_goes_where_the_vote_moves_it(
    tmp_path, monkeypatch
):
    """The overhead view reads slope and both wrist views read wide, the bricks' class, and the
    teacher opens over the slope box after its lift but before the carry view decides
    (CARRY_VIEW_STEPS 100 here). Off, the first part falls into the slope box: the g_cg_scr
    M_late case. On, the guard holds the open while the vote is pending, holds it again once the
    vote has moved the box to the wide box, and the still guard hands the part to the rig,
    which places it there."""
    monkeypatch.setattr(loop, "CARRY_VIEW_STEPS", 100)
    stub = _classifier(tmp_path, "slope", "wide")
    episodes = []
    for guard in (False, True):
        env, rig, policy = _scene()
        slope = int(env.class_to_box[0][parts.BASE_CLASSES.index("slope")])
        x, y = (float(v) for v in env.desk_box_poses[0, slope, :2])
        release = (x, y, desk_boxes.DESK_BOX_RELEASE_Z)
        policy.rig.oracles[0].release_point = torch.tensor(release)
        payload = _rollout(
            env,
            rig,
            policy,
            perceived=True,
            classifier=stub,
            class_views=True,
            carry_guard=guard,
            **SWITCHES,
        )
        env.close()
        episodes.append(payload["episodes"][0])
    off, on = episodes
    assert off["cycle_classes"][0] == "wrong_box"
    assert off["cycle_census"][0]["release_place"] == "wrong_box"
    row = on["cycle_census"][0]
    assert on["cycle_classes"][0] == "stowed" and row["handoff"] == "placed", row
    assert row["read_class"] == "slope" and row["taken_class"] == "wide"
    assert row["guard_vote_steps"] > 0 and row["guard_rim_steps"] == 0
    assert row["guard_steps"] == row["guard_vote_steps"] + loop.GUARD_STALL_STEPS


def test_the_rollout_refuses_the_switch_without_perceived_or_with_the_passenger_drop(
    monkeypatch,
):
    for kwargs in ({}, {"perceived": True, "passenger_drop": True}):
        env, rig, policy, budget, _ = _teacher_scene(monkeypatch)
        with pytest.raises(loop.EvalFailure) as err:
            _rollout(env, rig, policy, budget=budget, carry_guard=True, **kwargs)
        env.close()
        assert err.value.reason == "carry_guard", kwargs


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--oracle_driver"], "--carry_guard needs --perceived"),
        (["--perceived", "--passenger_drop"], "--carry_guard refuses --passenger_drop"),
    ],
)
def test_the_client_refuses_the_switch_with_its_token(argv, message, capsys):
    module = _load(ROOT / "scripts" / "mj_eval_heap.py")
    with pytest.raises(SystemExit) as exit_info:
        module.main(argv + ["--heap_k", "3", "--desk_boxes", "4", "--carry_guard"])
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("EVAL_FAIL ") and message in out[0], out


HEAP = ["--heap_k", "3", "--desk_boxes", "4", "--gripper", "continuous", "--num_envs", "1"]


class _Policy:
    def close(self):
        pass


RETRY = ["--perceived", "--carry_guard", "--replan_on_switch", "--retry_noise_at", "20"]


@pytest.mark.parametrize(
    "argv, message",
    [
        (RETRY[:1] + RETRY[2:], "--retry_noise_at needs --carry_guard and"),
        (RETRY[:2] + RETRY[3:], "--retry_noise_at needs --carry_guard and"),
        (RETRY[:-1] + ["-1"], "--retry_noise_at must be 0 (off) or more, under 60"),
        (RETRY[:-1] + ["60"], "--retry_noise_at must be 0 (off) or more, under 60"),
    ],
)
def test_the_client_refuses_a_bad_retry_noise_at_with_its_token(argv, message, capsys):
    module = _load(ROOT / "scripts" / "mj_eval_heap.py")
    with pytest.raises(SystemExit) as exit_info:
        module.main(HEAP + argv)
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("EVAL_FAIL ") and message in out[0], out


def test_the_client_hands_the_retry_noise_at_to_the_rollout(monkeypatch, capsys):
    module = _load(ROOT / "scripts" / "mj_eval_heap.py")
    monkeypatch.setattr(module.policy_client, "ServerPolicy", lambda server: _Policy())
    real_env = module.MjLegoSortEnv
    monkeypatch.setattr(
        module, "MjLegoSortEnv", lambda *a, **k: real_env(*a, **{**k, "cameras": False})
    )
    seen = []

    def run(env, rig, policy, **kwargs):
        seen.append(kwargs)
        return {}

    for argv, at in ((RETRY, 20), (RETRY[:-2], 0)):
        assert module.main(HEAP + argv, run=run) == 0
        assert capsys.readouterr().out.strip().splitlines()[-1].startswith("EVAL_OK ")
        assert seen.pop()["retry_noise_at"] == at


def test_the_supervisor_forwards_the_retry_noise_at_and_splits_the_cache_on_it(tmp_path):
    draw = ge.make_eval_draws(1, 1)[0]
    argv = _gate_argv(tmp_path) + ["--gripper", "continuous", "--perceived", "--carry_guard"]
    plain = ge._build_cfg(ge._parse_args(argv + ["--client_flags=--replan_on_switch"]))
    flags = ["--client_flags=--replan_on_switch --retry_noise_at 20"]
    cfg = ge._build_cfg(ge._parse_args(argv + flags))
    cmd = ge._eval_command(cfg, draw)
    assert cmd[cmd.index("--retry_noise_at") + 1] == "20"
    assert "--retry_noise_at" not in ge._eval_command(plain, draw)
    found = ge.loop_switches(cfg.client_flags)
    assert found["retry_noise_at"] == 20
    assert ge.loop_switches(plain.client_flags)["retry_noise_at"] == 0
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None}
    path = tmp_path / "eval_000.json"

    def hit(gate, loop_block):
        inference = {**served, "loop": loop_block}
        payload = _heap_payload(draw, desk_boxes=4, gripper="continuous", inference=inference)
        path.write_text(json.dumps(payload))
        return ge._cached_payload(gate, draw) is not None

    recorded = {k: v for k, v in found.items() if k != "retry_noise_at"}  # off: not recorded
    assert hit(plain, recorded) and not hit(cfg, recorded)
    on = recorded | {"retry_noise_at": 20}
    assert hit(cfg, on) and not hit(plain, on)
    assert not hit(cfg, recorded | {"retry_noise_at": 30})


def test_the_client_hands_the_guard_margin_to_the_rollout(monkeypatch, capsys):
    module = _load(ROOT / "scripts" / "mj_eval_heap.py")
    with pytest.raises(SystemExit):
        module.main(HEAP + ["--perceived", "--guard_margin", "0.02"])
    assert "--guard_margin needs --carry_guard" in capsys.readouterr().out
    monkeypatch.setattr(module.policy_client, "ServerPolicy", lambda server: _Policy())
    real_env = module.MjLegoSortEnv
    monkeypatch.setattr(
        module, "MjLegoSortEnv", lambda *a, **k: real_env(*a, **{**k, "cameras": False})
    )
    seen = []

    def run(env, rig, policy, **kwargs):
        seen.append(kwargs)
        return {}

    guard = ["--perceived", "--carry_guard"]
    for argv, margin in ((guard + ["--guard_margin", "0.02"], 0.02), (guard, 0.010)):
        assert module.main(HEAP + argv, run=run) == 0
        assert capsys.readouterr().out.strip().splitlines()[-1].startswith("EVAL_OK ")
        assert seen.pop()["guard_margin"] == margin


def test_the_supervisor_forwards_the_guard_margin_and_splits_the_cache_on_it(tmp_path):
    assert ge.LOOP_MARGINS == {"guard_margin": loop.GUARD_BOX_MARGIN_M}
    draw = ge.make_eval_draws(1, 1)[0]
    argv = _gate_argv(tmp_path) + ["--gripper", "continuous", "--perceived", "--carry_guard"]
    plain = ge._build_cfg(ge._parse_args(argv))
    cfg = ge._build_cfg(ge._parse_args(argv + ["--client_flags=--guard_margin 0.02"]))
    same = ge._build_cfg(ge._parse_args(argv + ["--client_flags=--guard_margin 0.01"]))
    cmd = ge._eval_command(cfg, draw)
    assert cmd[cmd.index("--guard_margin") + 1] == "0.02"
    assert "--guard_margin" not in ge._eval_command(plain, draw)
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None}
    path = tmp_path / "eval_000.json"

    def hit(gate, loop_block):
        inference = {**served, "loop": loop_block}
        payload = _heap_payload(draw, desk_boxes=4, gripper="continuous", inference=inference)
        path.write_text(json.dumps(payload))
        return ge._cached_payload(gate, draw) is not None

    # The client records the margin only off its default, so a default read leaves it out.
    recorded = ge.loop_switches(plain.client_flags)
    del recorded["guard_margin"]
    assert hit(plain, recorded) and hit(same, recorded) and not hit(cfg, recorded)
    on = recorded | {"guard_margin": 0.02}
    assert hit(cfg, on) and not hit(plain, on)
    assert not hit(cfg, recorded | {"guard_margin": 0.03})


def test_the_release_handoff_reaches_the_rollout_and_splits_the_supervisors_cache(
    tmp_path, monkeypatch, capsys
):
    module = _load(ROOT / "scripts" / "mj_eval_heap.py")
    monkeypatch.setattr(module.policy_client, "ServerPolicy", lambda server: _Policy())
    real_env = module.MjLegoSortEnv
    monkeypatch.setattr(
        module, "MjLegoSortEnv", lambda *a, **k: real_env(*a, **{**k, "cameras": False})
    )
    seen = []

    def run(env, rig, policy, **kwargs):
        seen.append(kwargs)
        return {}

    guard = ["--perceived", "--carry_guard", "--guard_margin", "0.025"]
    for argv, on in ((guard + ["--release_handoff"], True), (guard, False)):
        assert module.main(HEAP + argv, run=run) == 0
        assert capsys.readouterr().out.strip().splitlines()[-1].startswith("EVAL_OK ")
        assert seen.pop()["release_handoff"] is on
    draw = ge.make_eval_draws(1, 1)[0]
    argv = _gate_argv(tmp_path) + ["--gripper", "continuous", "--perceived", "--carry_guard"]
    plain = ge._build_cfg(ge._parse_args(argv + ["--client_flags=--guard_margin 0.025"]))
    flags = "--client_flags=--guard_margin 0.025 --release_handoff"
    cfg = ge._build_cfg(ge._parse_args(argv + [flags]))
    cmd = ge._eval_command(cfg, draw)
    assert cmd[cmd.index("--guard_margin") + 2] == "--release_handoff"
    assert "--release_handoff" not in ge._eval_command(plain, draw)
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None}
    path = tmp_path / "eval_000.json"

    def hit(gate, loop_block):
        inference = {**served, "loop": loop_block}
        payload = _heap_payload(draw, desk_boxes=4, gripper="continuous", inference=inference)
        path.write_text(json.dumps(payload))
        return ge._cached_payload(gate, draw) is not None

    recorded = ge.loop_switches(plain.client_flags)
    assert recorded.pop("release_handoff") is False  # off: not recorded
    assert hit(plain, recorded) and not hit(cfg, recorded)
    on = recorded | {"release_handoff": True}
    assert hit(cfg, on) and not hit(plain, on)


def test_the_supervisor_forwards_the_switch_and_its_cache_key_records_it(tmp_path):
    draw = ge.make_eval_draws(1, 1)[0]
    argv = _gate_argv(tmp_path) + ["--gripper", "continuous", "--perceived"]
    plain = ge._build_cfg(ge._parse_args(argv))
    cfg = ge._build_cfg(ge._parse_args(argv + ["--carry_guard"]))
    assert ge._eval_command(cfg, draw).count("--carry_guard") == 1
    assert "--carry_guard" not in ge._eval_command(plain, draw)
    base = {name: False for name in ge.LOOP_SWITCHES} | {"perceived": True}
    base["noise_scales"] = ge.noise_scales_key("")
    assert ge.loop_switches(cfg.client_flags) == base | {
        "carry_guard": True,
        "rake_cap": 0,
        "retry_noise_at": 0,
        "guard_margin": 0.010,
    }
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None}
    path = tmp_path / "eval_000.json"

    def hit(gate, loop_block):
        inference = {**served, "loop": loop_block}
        payload = _heap_payload(draw, desk_boxes=4, gripper="continuous", inference=inference)
        path.write_text(json.dumps(payload))
        return ge._cached_payload(gate, draw) is not None

    recorded = {k: v for k, v in base.items() if k != "carry_guard"}  # off: not recorded
    assert hit(plain, recorded) and not hit(cfg, recorded)
    assert hit(cfg, recorded | {"carry_guard": True}) and not hit(
        plain, recorded | {"carry_guard": True}
    )
    for bad, message in (
        (_gate_argv(tmp_path) + ["--carry_guard"], "--carry_guard needs --perceived"),
        (argv + ["--carry_guard", "--passenger_drop"], "--carry_guard refuses --passenger_drop"),
    ):
        with pytest.raises(ValueError, match=message):
            ge._build_cfg(ge._parse_args(bad))
