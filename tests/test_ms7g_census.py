"""The MS7g census (MS7g design 5.1 and 6.2, MS7g interface section 3): the air stall detector,
the release place, the census fields on every cycle row, and ledger rule G4's pin: with every new
switch off, a pinned draw gives main 56c42b6's payload except for the new census fields."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from maker_arm_lab.core import desk_boxes, heap_oracle, layout, loop, parts
from test_classifier_loop import _classifier
from test_perceived_loop import MS7D_ON, _rollout, _run, _Still, _teacher_scene
from test_snap_closed import _Blind

PIN = Path(__file__).parent / "data" / "ms7g_census_pin_56c42b6.json"
"""The pinned payloads of main 56c42b6, ``pinned_payloads`` run on main's ``source`` there."""
CANDIDATE_6 = {
    "replan_on_switch": True,
    "scripted_rake": True,
    "rake_turn": True,
    "thin_snap": True,
    "snap_closed": True,
    "box_empty": True,
    "missed_lift": True,
    "cell_failure_cap": 5,
} | MS7D_ON
"""Candidate 6's loop switches (``read_f2.sh`` with its EXTRA) beyond ``_teacher_scene``'s own,
frozen here: a later track that turns its switch on in ``test_perceived_loop``'s scenes leaves
this pin alone."""


def _tray_scene(monkeypatch):
    """Candidate 6's switches with a stub classifier on the MS7f tray scene of
    ``test_perceived_loop._ms7f_scene``: three parts, two of them in the leaned deep tray, six
    categories over the drawn boxes, the refill and an emptying between cycles. The stub reads
    special on the bright stand-in overhead view and brick on the black wrist frames, so every
    lift splits the vote and waits for the carry view."""
    monkeypatch.setattr(layout, "TRAY_REFILL_START", 2)
    monkeypatch.setattr(layout, "TRAY_REFILL_AT", 1)
    monkeypatch.setattr(desk_boxes, "BOX_VOLUME_PER_PART_VOLUME", 1e4)
    env, rig, policy, budget, switches = _teacher_scene(
        monkeypatch,
        parts.active_categories(),
        box_draw=6,
        categories=True,
        tray=True,
        tray_lean=30.0,
        deep_tray=True,
        tray_refill=True,
    )
    env.narrow_overhead = lambda i, point: np.full((224, 224, 3), 255, dtype=np.uint8)
    stub = _classifier(Path(tempfile.mkdtemp()), "special", "brick", names=parts.PART_CATEGORIES)
    return env, rig, policy, budget, switches | CANDIDATE_6 | {"classifier": stub}


def _desk_scene(monkeypatch):
    """The perceived teacher scene on the bare desk with the MS7d levers: three bricks, each
    lifted and placed."""
    env, rig, policy, budget, switches = _teacher_scene(monkeypatch)
    return env, rig, policy, budget, switches | MS7D_ON


def _payload(scene, monkeypatch) -> dict:
    torch.manual_seed(0)  # the teacher's command noise draws from torch's global generator
    payload, _ = _run(scene, monkeypatch)
    switches = payload["inference"]["loop"]
    if "classifier" in switches:  # the stub's file lies in a fresh temporary directory
        switches["classifier"] = Path(switches["classifier"]).name
    return payload


def pinned_payloads(monkeypatch) -> dict:
    """The two pinned draws' payloads. ``PIN`` was written from a worktree of main 56c42b6 with
    this module on the path (``PYTHONPATH=$MAIN/source:$TIP/tests``)::

        import json, pytest, test_ms7g_census as t
        pinned = t.pinned_payloads(pytest.MonkeyPatch())
        t.PIN.write_text(json.dumps(pinned, indent=1, sort_keys=True) + "\n")
    """
    return {name: _payload(scene, monkeypatch) for name, scene in SCENES.items()}


SCENES = {"tray": _tray_scene, "desk": _desk_scene}


def _without_census(payload: dict) -> dict:
    """The payload without what the MS7g census and the MS7i release census add, floats to 9
    decimals."""
    payload = json.loads(json.dumps(payload), parse_float=lambda s: round(float(s), 9))
    payload["perception"].pop("logp_order", None)
    for ep in payload["episodes"]:
        ep.pop("wrong_box_part_ids", None)
        for row in ep["cycle_census"]:
            for name in loop.MS7G_CENSUS + loop.RELEASE_CENSUS:
                row.pop(name, None)
    return payload


# -- the air stall ------------------------------------------------------------------------------


def _stall_run(points, over=True) -> list[tuple[bool, bool]]:
    stall = loop.AirStall()
    return [(stall.update(p, over), stall.entered) for p in points]


def test_an_air_stall_needs_60_steps_still_over_the_tray_at_lift_height():
    assert (loop.AIR_STALL_STEPS, loop.AIR_STALL_MIN_Z, loop.AIR_STALL_SPEED) == (60, 0.06, 0.0003)
    window = loop.AIR_STALL_WINDOW
    # The speed's window fills on the 31st reading; the 60th reading in a row after it stalls.
    still = _stall_run([(0.2, 0.0, 0.10)] * 200)
    first = window + loop.AIR_STALL_STEPS - 1
    assert [s for s, _ in still].index(True) == first and all(s for s, _ in still[first:])
    assert [e for _, e in still].count(True) == 1 and still[first][1]
    assert not any(s for s, _ in _stall_run([(0.2, 0.0, 0.10)] * 200, over=False))
    assert not any(s for s, _ in _stall_run([(0.2, 0.0, 0.059)] * 200))  # under lift height


def test_the_stall_reads_the_mean_speed_over_30_steps():
    slow = [(0.2 + 0.00029 * t, 0.0, 0.10) for t in range(200)]
    fast = [(0.2 + 0.00031 * t, 0.0, 0.10) for t in range(200)]
    assert any(s for s, _ in _stall_run(slow)) and not any(s for s, _ in _stall_run(fast))
    # One 9.5 mm jump, past the window's whole allowance of 9 mm, ends a stall and keeps the next
    # one out until it leaves the window, then the run counts 60 steps again.
    points = [(0.2, 0.0, 0.10)] * 100 + [(0.2095, 0.0, 0.10)] * 200
    run = [s for s, _ in _stall_run(points)]
    assert run[99] and not any(run[100 : 100 + loop.AIR_STALL_WINDOW + 59])
    assert run[100 + loop.AIR_STALL_WINDOW + 59]
    entries = [e for _, e in _stall_run(points)]
    assert entries.count(True) == 2


# -- the release place --------------------------------------------------------------------------

BOXES = [((0.10, 0.20, 0.0), (0.08, 0.07, 0.04)), ((0.25, -0.18, 0.3), (0.09, 0.08, 0.04))]


def test_the_release_place_reads_the_box_interiors_their_walls_the_tray_and_the_desk():
    (x0, x1), (y0, _) = layout.tray_bounds()
    grow = layout.tray_rim_growth(30.0, layout.DEEP_TRAY_WALL_HEIGHT)
    wall = desk_boxes.DESK_BOX_WALL

    def place(xy, lean=30.0, right=(0,)):
        return loop.release_place(xy, BOXES, right, lean, layout.DEEP_TRAY_WALL_HEIGHT)

    assert place((0.10, 0.20)) == "right_box" and place((0.25, -0.18)) == "wrong_box"
    assert place((0.25, -0.18), right=(0, 1)) == "right_box"  # a category with two boxes
    assert place((0.10 + 0.04 - 0.001, 0.20)) == "right_box"  # just inside the interior
    assert place((0.10 + 0.04 + wall / 2, 0.20)) == "rim"  # on the wall
    assert place((0.10 + 0.04 + wall + 0.001, 0.20)) == "desk"  # just outside it
    assert place(layout.TRAY_CENTER) == "tray"
    assert place((x1 + grow - 0.001, 0.0)) == "tray"  # over the leaned wall, inside the rim
    assert place((x1 + grow + layout.TRAY_WALL / 2, 0.0)) == "rim"
    assert place((x0, y0 - grow - layout.TRAY_WALL - 0.001)) == "desk"
    assert place(layout.TRAY_CENTER, lean=None) == "desk"  # no tray
    assert loop.RELEASE_PLACES == ("tray", "right_box", "wrong_box", "desk", "rim")


def test_the_release_motion_reads_the_point_its_velocity_and_the_served_box_frame():
    pick = torch.tensor([0.12, 0.21, 0.08], dtype=torch.float64)
    xyz, vel, box_xy = loop.release_motion(pick, [0.11, 0.20, 0.08], (0.10, 0.20, math.pi / 2))
    assert xyz == pytest.approx([0.12, 0.21, 0.08])
    assert vel == pytest.approx([0.3, 0.3, 0.0])  # 1 cm in one 30 Hz control step
    assert box_xy == pytest.approx([0.01, -0.02])  # at 90 degrees the box's x axis is world y
    assert loop.release_motion(pick, None, None)[1:] == (None, None)


# -- the census on a rollout --------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SCENES))
def test_a_pinned_draw_reads_mains_payload_and_every_cycle_row_carries_the_census(
    name, monkeypatch
):
    """Ledger rule G4, with every new switch off: the census only adds fields. Every other
    payload byte of the pinned draw, every cycle's end, step count and first-close geometry
    among them, is main's. Every row has every field; a release place only after a lift, with
    the held part's true category; the log probabilities in the order the perception block
    names; and the swept count of the grasp each selection chose."""
    chosen = []
    real = heap_oracle.HeapOracle.select_target

    def spy(self, scene):
        target = real(self, scene)
        chosen.append(None if target is None else self.grasp.swept)
        return target

    monkeypatch.setattr(heap_oracle.HeapOracle, "select_target", spy)
    payload = _payload(SCENES[name], monkeypatch)
    pinned = json.loads(PIN.read_text())[name]
    assert _without_census(payload) == _without_census(pinned)
    rows = [row for ep in payload["episodes"] for row in ep["cycle_census"]]
    assert rows and all(set(loop.MS7G_CENSUS) <= set(row) for row in rows)
    assert [row["swept"] for row in rows] == [s for s in chosen if s is not None]
    order = payload["perception"].get("logp_order")
    names = parts.PART_CATEGORIES if name == "tray" else parts.BASE_CLASSES
    assert order == (list(parts.active_categories()) if name == "tray" else None)
    for row in rows:
        lifted = row["lift_detected"] or row["held_class"] is not None
        assert row["release_place"] in (*loop.RELEASE_PLACES, None)
        assert row["release_place"] is None or lifted, row
        release = [row[k] for k in loop.RELEASE_CENSUS]
        if row["release_place"] is None:
            assert release == [None] * 4, row
        else:  # the desk boxes serve every cycle a box
            xyz, vel, box, box_xy = release
            assert len(xyz) == len(vel) == 3 and box >= 0 and len(box_xy) == 2, row
        assert row["held_class"] is None or row["held_class"] in names
        for key in ("read_logp", "lift_logp", "carry_logp"):
            assert row[key] is None or len(row[key]) == len(order), (key, row[key])
        assert min(row["air_stall_steps"], row["air_stall_empty_steps"], row["air_stalls"]) >= 0
    placed = [row for row in rows if row["release_place"] is not None]
    assert placed, rows
    if name == "tray":  # the stub's split vote reads all three views
        assert all(row["read_logp"] for row in rows)
        assert any(row["lift_logp"] and row["carry_logp"] for row in placed), placed
    else:
        assert all(row[k] is None for row in rows for k in ("read_logp", "lift_logp"))


class _Holding(_Blind):
    """A hold detector that reads a hold on every step and never a lift."""

    def __init__(self):
        super().__init__()
        self.holding = True


@pytest.mark.parametrize("holding", [False, True])
def test_an_arm_held_still_over_the_tray_is_an_air_stall(holding, monkeypatch):
    """A policy that holds the arm still at home with the jaws open, the home pick point read as
    over the tray (it lies 11.5 cm up, 1.5 cm past the tray's far rim): the stall begins once the
    speed window and the 60-step run have filled, and every cycle step from then on counts,
    under the hold detector's hold or with nothing held. One entry in the episode."""
    if holding:
        monkeypatch.setattr(loop, "HoldDetector", _Holding)
    monkeypatch.setattr(loop, "over_tray", lambda xy, lean, height=0.0: True)
    env, rig, _, budget, switches = _teacher_scene(monkeypatch, tray=True)
    assert float(rig.pick_point(0)[2]) >= loop.AIR_STALL_MIN_Z
    payload = _rollout(env, rig, _Still(), budget=budget, perceived=True, **switches)
    env.close()
    ep = payload["episodes"][0]
    rows = ep["cycle_census"]
    held, empty = (sum(r[k] for r in rows) for k in ("air_stall_steps", "air_stall_empty_steps"))
    stalled = sum(ep["cycle_steps"]) - (loop.AIR_STALL_WINDOW + loop.AIR_STALL_STEPS - 1)
    assert (held, empty) == ((stalled, 0) if holding else (0, stalled)), (held, empty)
    assert [r["air_stalls"] for r in rows] == [int(holding)] + [0] * (len(rows) - 1)
    assert all(r["release_place"] is None and r["held_class"] is None for r in rows)
