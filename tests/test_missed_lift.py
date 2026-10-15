"""MS7f riders, ``missed_lift``: a closed command that carries the FK pick point off the tray is a
lift the hold detector missed, the wrist read runs at it, and the jaws opening over a perceived
box then end the cycle ``placed``. Off, every step and payload byte is what it was."""

import json

import numpy as np
import pytest

from helpers import load_script
from maker_arm_lab.core import loop, oracle, parts
from test_classifier_loop import BRIGHT, _classifier
from test_perceived_loop import _rollout, _served, _teacher_scene, _wall_band_scene
from test_snap_closed import _Blind
from test_thin_snap import _same

RISE = oracle.LIFT_MIN_RISE


def _cycle(**fields) -> loop.Cycle:
    return loop.Cycle(ordinal=1, target=0, start_step=0, stowed_before=set(), **fields)


def test_a_carry_starts_off_the_tray_above_the_lowest_closed_point_in_it():
    c = _cycle()
    assert not loop.carry_started(c, True, (0.2, 0.0, 0.03), in_tray=True)
    assert not loop.carry_started(c, True, (0.2, 0.0, 0.01), in_tray=True)
    assert c.carry_z == 0.01  # the lowest closed point over the tray
    assert not loop.carry_started(c, True, (0.2, 0.0, 0.01 + RISE + 0.01), in_tray=True)
    assert not loop.carry_started(c, True, (0.3, 0.0, 0.01 + RISE - 0.001), in_tray=False)
    assert not loop.carry_started(c, False, (0.3, 0.0, 0.01 + RISE + 0.01), in_tray=False)
    assert loop.carry_started(c, True, (0.3, 0.0, 0.01 + RISE), in_tray=False)


def test_no_closed_command_over_the_tray_starts_no_carry():
    c = _cycle()
    assert not loop.carry_started(c, False, (0.2, 0.0, 0.01), in_tray=True)  # open over it
    assert not loop.carry_started(c, True, (0.3, 0.0, 0.2), in_tray=False)  # closed off it
    assert c.carry_z is None


def test_a_carry_is_lost_when_the_jaws_open_wider_than_they_held_it():
    wide = loop.RELEASE_OPEN_M
    assert not loop.carry_lost(_cycle(), 0.05, lifted=False)  # it never carried
    for fields in ({"carried": True}, {"lift_detected": True}):
        c = _cycle(**fields)
        assert not loop.carry_lost(c, 0.012, lifted=False)
        assert not loop.carry_lost(c, 0.010, lifted=False) and c.carry_gap == 0.010
        assert not loop.carry_lost(c, 0.010 + wide - 1e-4, lifted=False)
        assert not loop.carry_lost(c, 0.010 + wide, lifted=True)  # the detector's own end
        assert loop.carry_lost(c, 0.010 + wide, lifted=False)


def _run(scene, monkeypatch, **switch):
    """The scene's perceived rollout: the payload without its wall clock, and what env 0 was
    served on every step."""
    env, rig, policy, budget, switches = scene(monkeypatch)
    served = _served(env)
    payload = _rollout(env, rig, policy, budget=budget, perceived=True, **switches, **switch)
    env.close()
    for key in ("wall_sec", "wall_sec_per_episode"):
        payload.pop(key)
    return payload, served


def test_with_the_switch_off_every_step_and_payload_byte_is_mains(monkeypatch):
    main, main_served = _run(_wall_band_scene, monkeypatch)
    off, off_served = _run(_wall_band_scene, monkeypatch, missed_lift=False)
    assert json.dumps(off, sort_keys=True) == json.dumps(main, sort_keys=True)
    assert _same(off_served, main_served)
    assert "missed_lift" not in off["inference"]["loop"]
    assert "missed_lifts" not in off["perception"]


def test_a_carry_the_detector_misses_is_read_and_ends_at_the_release(monkeypatch):
    """The tray scene with a detector that never holds: off, no lift is read and every cycle
    runs to its step budget; on, the carry off the tray is read, the jaws opening over the box
    end that cycle ``placed`` and the steps fall."""
    monkeypatch.setattr(loop, "HoldDetector", _Blind)
    off, _ = _run(_wall_band_scene, monkeypatch, missed_lift=False)
    on, _ = _run(_wall_band_scene, monkeypatch, missed_lift=True)
    off_ep, on_ep, seen = off["episodes"][0], on["episodes"][0], on["perception"]
    assert off["perception"]["lifts_classified"] == 0
    assert set(off_ep["cycle_reasons"]) == {"budget"}, off_ep["cycle_reasons"]
    assert "placed" in on_ep["cycle_reasons"], on_ep["cycle_reasons"]
    assert (
        seen["missed_lifts"] >= seen["carried_releases"] == on_ep["cycle_reasons"].count("placed")
    )
    assert seen["lifts_classified"] >= 1
    assert on_ep["bricks_stowed"] >= off_ep["bricks_stowed"]
    assert sum(on_ep["cycle_steps"][:2]) < sum(off_ep["cycle_steps"][:2])
    assert on["inference"]["loop"]["missed_lift"] is True


def test_a_missed_carry_reads_the_classifier_view_lift_view_steps_after_it(monkeypatch, tmp_path):
    """With ``lift_view_steps`` on too, every carry the blind detector missed reads its wrist
    view N steps later, as a detector lift does: no view is lost."""

    def scene(mp):
        env, *rest = _wall_band_scene(mp)
        env.narrow_overhead = lambda i, point: np.full((224, 224, 3), BRIGHT, dtype=np.uint8)
        return env, *rest

    monkeypatch.setattr(loop, "HoldDetector", _Blind)
    stub = _classifier(tmp_path, names=parts.PART_CLASSES)
    on, _ = _run(scene, monkeypatch, missed_lift=True, classifier=stub, lift_view_steps=6)
    seen = on["perception"]
    assert seen["missed_lifts"] >= 1 and seen["lift_views_lost"] == 0, seen
    assert seen["lifts_classified"] + seen["lifts_empty"] == seen["missed_lifts"], seen


def test_the_rollout_refuses_the_switch_without_the_tray_or_perceived(monkeypatch):
    for perceived in (True, False):
        env, rig, policy, budget, _ = _teacher_scene(monkeypatch)  # the bare desk
        with pytest.raises(loop.EvalFailure) as err:
            _rollout(env, rig, policy, budget=budget, perceived=perceived, missed_lift=True)
        env.close()
        assert err.value.reason == "missed_lift"


@pytest.mark.parametrize(
    "script, argv, message",
    [
        ("mj_eval_heap", ["--oracle_driver", "--tray"], "--missed_lift needs --perceived"),
        ("mj_eval_heap", [], "--missed_lift reads a carry off the tray: it needs --tray"),
        ("mj_generate_batch", ["--out_dir", "/tmp/x"], "--missed_lift is not built on this"),
    ],
)
def test_the_clients_refuse_the_switch_where_no_serving_loop_reads_it(
    script, argv, message, capsys
):
    module = load_script(script)
    base = ["--heap_k", "3", "--desk_boxes", "4", "--missed_lift"]
    with pytest.raises(SystemExit) as exit_info:
        module.main(argv + base)
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and "_FAIL" in out[0] and message in out[0], out
