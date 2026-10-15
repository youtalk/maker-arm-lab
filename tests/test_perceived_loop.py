"""The heap loop under ``perceived`` (MS7c design section 6): the pure rules, the served row,
box and frozen marker, the failure memory by position, the class at lift, and the no-truth
test (MS7c design section 10), on the three-part MuJoCo heap with four desk boxes, cameras off.
"""

import json
import math
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from maker_arm_lab.core import (
    batch_loop,
    desk_boxes,
    grasp,
    heap_oracle,
    layout,
    loop,
    oracle,
    parts,
    perception,
)
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.hold import HoldDetector
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort import oracle_runtime
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig
from test_heap_loop import SPREAD
from test_mj_desk_boxes import FIXED, _TeacherPolicy
from test_mj_heap_oracle import PARTS, _place

ZERO = perception.parse_scales("all=0")


def _env_and_rig(
    part_list=PARTS,
    budget=4,
    tray=False,
    class_names=parts.BASE_CLASSES,
    box_draw=0,
    tray_lean=0.0,
    wall_band_cont=False,
    categories=False,
    **ms7f,
):
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1,
        draw,
        parts=part_list,
        cameras=False,
        desk_boxes=4,
        tray=tray,
        tray_lean_deg=tray_lean,
        gripper_mode="continuous",
        context="camera_box_target",
        episode_length_s=loop.episode_length_s(budget),
        class_names=class_names,
        box_draw=box_draw,
        categories=categories,
        **ms7f,
    )
    env.reset(seed=11)
    if not box_draw:
        env.place_desk_boxes(0, FIXED)
    rng = random.Random(1)
    loads = [[0.0] * env.desk_boxes]  # MS7f --categories: the teacher's release record

    def factory(i, bl, episode_draw):
        return HeapOracle(
            bl,
            env.parts,
            env.destination_bins[i],
            episode_draw,
            rng=rng,
            cycle_budget=budget,
            boxes=env.desk_box_poses[i],
            tray=env.tray,
            box_sizes=env.desk_box_sizes[i],
            tray_lean_deg=env.tray_lean_deg,
            wall_band_cont=wall_band_cont,
            box_sets=env.part_box_sets(i),
            box_load=loads[i],
            tray_wall_height=env.tray_wall_height,
        )

    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory)
    return env, rig


class _Still:
    """Holds the arm still with the jaws open: every cycle ends on its step budget."""

    def __init__(self):
        self.context_dim = 16
        self.policy_id = "fake:still"
        self.n_action_steps = 1
        self.inference = {}
        self.action_space = "absolute"
        self.gripper = "continuous"

    def step(self, state, wrist, overhead, context=None, anchor=None, replan=None):
        action = np.zeros((state.shape[0], 7), dtype=np.float32)
        action[:, :6] = state[:, :6]
        action[:, 6] = 1.0
        return action


class _Home(_Still):
    """Opens the jaws, then commands the joint pose of its first call: after a takeover it lets
    go of what the jaws press and takes the arm back home."""

    def step(self, state, wrist, overhead, context=None, anchor=None, replan=None):
        self.home = getattr(self, "home", state[:, :6].copy())
        action = super().step(state, wrist, overhead, context, anchor, replan)
        opened = state[:, 6] > 0.95
        action[opened, :6] = self.home[opened]
        return action


class _Teacher(_TeacherPolicy):
    """The teacher as the policy (a second rig on the same env), served the 16-wide row."""

    def __init__(self, env, factory):
        super().__init__(env, factory)
        self.context_dim = 16

    def step(self, state, wrist, overhead, context=None, anchor=None, replan=None):
        return super().step(state, wrist, overhead, context, anchor)


def _rollout(env, rig, policy, budget=4, **kwargs):
    return loop.rollout(
        env,
        rig,
        policy,
        run_seed=11,
        layout_seed=5,
        permutation=(0, 1, 2, 3),
        cycle_budget=budget,
        max_wall_minutes=10.0,
        **kwargs,
    )


def _served(env):
    """What env 0 is served on every step: the target row, the box row, the marker point, the
    loop's target_part and the action, read as the loop calls ``env.step``."""
    served = []
    stepped = env.step

    def step(action):
        served.append(
            (
                env.context_target[0].copy(),
                env.context_box[0].copy(),
                env.marker_points[0].clone(),
                int(env.target_part[0]),
                action[0].clone(),
            )
        )
        return stepped(action)

    env.step = step
    return served


def _run(scene, monkeypatch, **switch):
    """The scene's perceived rollout with ``switch`` over the scene's switches: the payload
    without its wall clock, through JSON, and what env 0 was served on every step."""
    env, rig, policy, budget, switches = scene(monkeypatch)
    served = _served(env)
    payload = _rollout(env, rig, policy, budget=budget, perceived=True, **(switches | switch))
    env.close()
    for key in ("wall_sec", "wall_sec_per_episode"):
        payload.pop(key)
    return json.loads(json.dumps(payload, sort_keys=True)), served


def _selections(monkeypatch):
    """Every perceived state a selection ranked, in order."""
    states = []
    real = loop._perceived_scene

    def spy(st, env, rig, i, k):
        state, scene = real(st, env, rig, i, k)
        states.append(state)
        return state, scene

    monkeypatch.setattr(loop, "_perceived_scene", spy)
    return states


# -- the pure rules ---------------------------------------------------------------------------


def test_the_perceived_cycle_end_is_the_release_over_a_box_or_off_it_then_the_budget():
    cycle = loop.Cycle(ordinal=1, target=0, start_step=100, stowed_before=set())
    assert loop.perceived_cycle_end(cycle, 101, False, False) is None
    assert loop.perceived_cycle_end(cycle, 101, True, True) == "placed"
    assert loop.perceived_cycle_end(cycle, 101, True, False) == "dropped"
    budget = 100 + loop.CYCLE_STEP_BUDGET
    assert loop.perceived_cycle_end(cycle, budget, False, True) == "budget"  # over, not released
    assert loop.perceived_cycle_end(cycle, budget, True, True) == "placed"  # the release first
    cycle.takeover_end = "takeover_failed"
    assert loop.perceived_cycle_end(cycle, 101, False, False) == "takeover_failed"
    assert {"placed", "dropped"} <= set(loop.RELEASE_ENDS)


def test_a_perceived_release_completes_only_once_the_jaws_open():
    cycle = loop.Cycle(ordinal=1, target=0, start_step=100, stowed_before=set())
    wide = loop.SELECT_OPEN_FRACTION
    assert not loop.perceived_release(cycle, False, False, False, wide)  # no lifted hold ended
    assert not loop.perceived_release(cycle, True, False, True, 0.3)  # the hold ends, jaws shut
    assert not loop.perceived_release(cycle, False, False, False, 0.5)
    assert loop.perceived_release(cycle, False, False, False, wide)  # over kept from the end
    assert cycle.release_over is True
    cycle.release_over = None
    loop.perceived_release(cycle, True, False, False, 0.3)
    assert not loop.perceived_release(cycle, False, True, False, wide)  # a new hold cancels it
    assert cycle.release_over is None


def test_the_release_gap_completes_a_release_10_mm_past_the_gap_at_the_holds_end():
    """MS7d L2: the MS7c rule still completes a release, a new hold still cancels it, and with
    the switch off (``gap`` None) the rule is MS7c's."""
    cycle = loop.Cycle(ordinal=1, target=0, start_step=100, stowed_before=set())
    end = 0.016
    assert not loop.perceived_release(cycle, True, False, True, 0.3, end)  # the hold ends
    assert cycle.release_gap0 == end
    assert not loop.perceived_release(cycle, False, False, False, 0.3, end + 0.0099)
    assert loop.perceived_release(cycle, False, False, False, 0.3, end + 0.0101)
    assert cycle.release_over is True
    loop.perceived_release(cycle, True, False, False, 0.3, end)
    assert not loop.perceived_release(cycle, False, True, False, 0.3, 0.04)  # a new hold
    assert cycle.release_over is None
    loop.perceived_release(cycle, True, False, False, 0.3, end)
    assert loop.perceived_release(cycle, False, False, False, loop.SELECT_OPEN_FRACTION, end)
    loop.perceived_release(cycle, True, False, False, 0.3)  # the switch off
    assert not loop.perceived_release(cycle, False, False, False, 0.3)
    assert loop.perceived_release(cycle, False, False, False, loop.SELECT_OPEN_FRACTION)


def test_rearm_stop_clears_after_a_hold_with_no_lift_and_keeps_a_lifted_one():
    """MS7d L8 on the real detector. Jaws stalled 10 mm wide on a shut command read a hold, and
    jaws shut on the command end it. A hold with no lift turns the stop flag on and its end
    clears it once, with a spent ``dwell_close`` armed again from a dwell of 0. A new hold turns
    it on again, and the end of a lifted hold keeps it and the spent close."""
    d = HoldDetector()
    c = loop.Cycle(ordinal=1, target=0, start_step=0, stowed_before=set())
    c.latched, c.dwell = -1, loop.DWELL_STEPS  # dwell_close fired and is spent

    def feed(gap, z=0.05):
        cleared = []
        for _ in range(8):
            d.update(gap, 0.0, z)
            cleared.append(loop.rearm_stop(c, d))
        return sum(cleared)

    assert feed(0.010) == 0 and d.holding and c.stop
    assert feed(0.0) == 1 and not c.stop and (c.rearms, c.latched, c.dwell) == (1, 0, 0)
    assert feed(0.010) == 0 and c.stop
    feed(0.010, z=0.05 + oracle.LIFT_MIN_RISE)
    assert d.lifted
    c.latched = -1  # the second close is spent too
    assert feed(0.0) == 0 and not d.holding and c.stop and (c.rearms, c.latched) == (1, -1)


def test_over_box_reads_each_boxs_yawed_footprint():
    _, (h, _, _) = desk_boxes.stow_region(desk_boxes.interior(2))  # the footprint is square
    boxes = [(0.10, 0.20, 0.0), (0.10, -0.20, math.pi / 4)]
    SIZES = desk_boxes.interiors(2)
    assert loop.over_box((0.10, 0.20), boxes, SIZES) and loop.over_box((0.10, -0.20), boxes, SIZES)
    assert loop.over_box((0.10 + 0.9 * h, 0.20 + 0.9 * h), boxes, SIZES)  # box 0's corner
    assert not loop.over_box((0.10 + 1.1 * h, 0.20), boxes, SIZES)
    # Box 1 turned 45 degrees: a corner points along world y, and the square's own corner
    # direction now crosses an edge.
    assert loop.over_box((0.10, -0.20 + 1.25 * h), boxes, SIZES)
    assert not loop.over_box((0.10 + 0.9 * h, -0.20 + 0.9 * h), boxes, SIZES)
    assert not loop.over_box((0.30, 0.0), boxes, SIZES)


def test_the_class_at_lift_serves_the_box_of_the_class_drawn_for_the_part_in_the_jaws():
    boxes = {"slope": 2, "tall": 3, "narrow": 1, "wide": 0}
    draws = []

    def classify(cls):
        draws.append(cls)
        return "slope" if cls == "tall" else cls  # a confusion on tall parts

    # Nothing between the jaws (a false lift): no draw, the row stays.
    assert loop.lift_class_box(None, classify, boxes, 0) == (None, None, None) and draws == []
    # The served box already fits the drawn class: no switch.
    assert loop.lift_class_box("wide", classify, boxes, 0) == ("wide", "wide", None)
    # Another class, or a confused one: the box of the class drawn, not of the true one.
    assert loop.lift_class_box("narrow", classify, boxes, 0) == ("narrow", "narrow", 1)
    assert loop.lift_class_box("tall", classify, boxes, 0) == ("slope", "slope", 2)
    assert draws == ["wide", "narrow", "tall"]


def test_class_views_draws_a_third_view_only_when_the_wrist_disagrees_with_the_overhead():
    """MS7d L5: a wrist class equal to the overhead one is taken with no third view; otherwise
    the vote of the three views is taken and serves its box."""
    boxes = {"slope": 2, "tall": 3, "narrow": 1, "wide": 0}
    thirds = []

    def third(cls):
        thirds.append(cls)
        return "tall"

    def box(seen, drawn):
        return loop.lift_class_box("tall", lambda cls: drawn, boxes, 0, seen=seen, third=third)

    assert box("slope", "slope") == ("slope", "slope", 2) and thirds == []
    assert box("tall", "slope") == ("slope", "tall", 3) and thirds == ["tall"]  # sides with seen
    assert box("wide", "slope") == ("slope", "tall", 3)  # a split: the third view
    assert loop.lift_class_box(None, None, boxes, 0, seen="wide", third=third) == (None,) * 3


def test_the_vote_takes_the_class_two_views_share_and_the_third_on_a_split():
    assert loop.voted_class(("wide", "slope", "wide")) == "wide"
    assert loop.voted_class(("wide", "slope", "slope")) == "slope"
    assert loop.voted_class(("wide", "wide", "tall")) == "wide"
    assert loop.voted_class(("wide", "slope", "tall")) == "tall"


def test_the_tool_is_clear_of_the_tray_40_mm_off_its_footprint_at_the_rim():
    (x0, x1), (y0, y1) = layout.tray_bounds()
    radius = perception.TOOL_OCCLUSION_RADIUS_M
    rim = layout.TRAY_WALL_HEIGHT * math.tan(math.radians(30.0))
    assert not loop.tool_clear_of_tray(((x0 + x1) / 2, 0.0), 0.0)  # over the tray
    assert not loop.tool_clear_of_tray((x1 + radius - 1e-4, 0.0), 0.0)
    assert loop.tool_clear_of_tray((x1 + radius + 1e-4, 0.0), 0.0)
    # A 30 degree lean widens the rim by 11.5 mm: the same point is not clear any more.
    assert not loop.tool_clear_of_tray((x1 + radius + 1e-4, 0.0), 30.0)
    assert loop.tool_clear_of_tray((x1 + rim + radius + 1e-4, 0.0), 30.0)
    corner = radius / math.sqrt(2.0)
    assert not loop.tool_clear_of_tray((x0 - corner + 1e-4, y0 - corner + 1e-4), 0.0)
    assert loop.tool_clear_of_tray((x0 - corner - 1e-4, y1 + corner + 1e-4), 0.0)
    cap = int(loop.SELECT_WAIT_FRACTION * loop.CYCLE_STEP_BUDGET)
    over = ((x0 + x1) / 2, 0.0)
    assert loop.look_again(over, 0.0, 0) and loop.look_again(over, 0.0, cap - 1)
    assert not loop.look_again(over, 0.0, cap)  # the streak reached the cap
    assert not loop.look_again((x1 + radius + 0.01, 0.0), 0.0, 0)  # clear: end at once


def test_the_selector_skips_the_destination_grade_when_it_knows_no_destination():
    bl = layout.DEFAULT_BIN_LAYOUT
    sel = HeapOracle(bl, PARTS, [0, 0, 0], oracle.nominal_draw(), boxes=FIXED)
    z = parts.make_part("brick", 2, 4).size[2] / 2
    scene = oracle.SceneSnapshot(
        brick_pos=torch.tensor([[0.2, -0.05, z], [0.2, 0.0, z], [0.2, 0.05, z]]),
        brick_yaw=torch.zeros(3),
        brick_lin_vel=torch.zeros(3, 3),
        pick_point=torch.tensor([0.3, 0.0, 0.1]),
        brick_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3),
    )
    graded = []
    sel._stowed = lambda scene: graded.append(1) or set()
    assert sel._candidates(scene).all() and graded == [1]
    sel.knows_destinations = False
    assert sel._candidates(scene).all() and graded == [1]


def _in_box(pose, u, v, z):
    """A world point at (u, v) in the box frame, height z."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + c * u - s * v, y + s * u + c * v, z)


def test_box_filter_xy_tests_the_outer_footprint_plus_the_margin_and_not_z():
    """MS7d L1. Part 0 rests on box 2's floor, seen 5 mm low: under the grader's region, whose
    lower face is the floor, so the MS7c filter lets it back in. Parts 1 and 2 lie 20 mm and
    4 mm off the outer wall of box 3, yawed 0.6 rad: only the first clears the 5 mm margin.
    Part 3 lies on the open desk."""
    z = parts.make_part("brick", 2, 4).size[2] / 2
    _, hy = desk_boxes.outer_half(desk_boxes.interior(len(FIXED)))
    positions = [
        _in_box(FIXED[2], 0.0, -0.035, desk_boxes.DESK_BOX_WALL + z - 0.005),
        _in_box(FIXED[3], 0.0, hy + 0.020, z),
        _in_box(FIXED[3], 0.0, hy + 0.004, z),
        (0.2, 0.0, z),
    ]
    scene = oracle.SceneSnapshot(
        brick_pos=torch.tensor(positions),
        brick_yaw=torch.zeros(4),
        brick_lin_vel=torch.zeros(4, 3),
        pick_point=torch.tensor([0.3, 0.0, 0.1]),
        brick_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4),
    )
    sel = HeapOracle(
        layout.DEFAULT_BIN_LAYOUT, [PARTS[0]] * 4, 0, oracle.nominal_draw(), boxes=FIXED
    )
    sel.knows_destinations = False
    assert sel._candidates(scene).tolist() == [True, True, True, True]
    sel.box_filter_xy = True
    assert sel._candidates(scene).tolist() == [False, True, False, True]


ROW = [0.2, 0.0, 0.005, math.cos(0.3), math.sin(0.3), 0.016, 0.005]
"""A served target row (``target_context``): centre, closing axis at 0.3 rad, width, half
height."""


def test_a_wrist_look_is_due_on_entering_the_snap_window_and_again_only_after_leaving_it():
    """MS7d L4: the window is the snap window, and a look fires once per stay in it."""
    cycle = loop.Cycle(ordinal=1, target=0, start_step=0, stowed_before=set())
    top = ROW[2] + ROW[6]
    high = [0.2, 0.0, top + loop.SQUARE_ABOVE + 0.001]
    inside = [0.2 + loop.SNAP_RADIUS - 0.001, 0.0, top + 0.02]
    wide = [0.2 + loop.SNAP_RADIUS + 0.001, 0.0, top]
    path = [high, inside, inside, ROW[:3], wide, inside, high, inside]
    due = [loop.look_due(cycle, pick, ROW) for pick in path]
    assert due == [False, True, False, False, False, True, False, True]
    cycle.in_window = False
    assert not loop.look_due(cycle, ROW[:3], [*ROW[:5], 0.0, ROW[6]])  # no live row


def test_the_wrist_row_moves_the_centre_and_turns_the_heading_by_the_folded_yaw_error():
    """MS7d L4: the serving row takes the look's centre and turns by the look's yaw minus the
    detection's, folded into a quarter turn; z, width and half height stay, and so does the
    selected row it was built from."""
    row = np.array(ROW, dtype=np.float32)
    kept = row.copy()
    quarter = math.pi / 4
    for err, turn in (
        (0.05, 0.05),
        (-0.05, -0.05),
        (math.pi + 0.05, 0.05),
        (quarter + 0.01, -quarter + 0.01),
    ):
        out = loop.wrist_row(row, (0.203, -0.002, 1.0 + err), 1.0)
        assert out[:2] == [0.203, -0.002]
        assert [out[2], *out[5:]] == [float(v) for v in (row[2], row[5], row[6])]
        assert math.atan2(out[4], out[3]) == pytest.approx(0.3 + turn, abs=1e-6)
    assert np.array_equal(row, kept)


def test_the_force_window_reads_the_wrist_looks_serving_row():
    """MS7d L4 with L6 and L7: after a wrist look the force sensor's snap window reads the
    serving row, as ``pick_snap`` does, and not the row served at selection."""
    windows = []

    class Sensor:
        def update(self, window):
            windows.append(window)
            return 0.0

    class Env:
        def arm_torques(self, i):
            return ()

    st = loop.EnvState(selector=None)
    st.cycle = loop.Cycle(ordinal=1, target=0, start_step=0, stowed_before=set())
    moved = [ROW[0] + 2 * loop.SNAP_RADIUS, *ROW[1:]]
    for serving in (None, moved):
        st.cycle.serving_row = serving
        loop._pick_forces(Env(), [Sensor()], [st], [ROW], [moved[:3]], [False])
    assert windows == [False, True]


# -- the rollout --------------------------------------------------------------------------------


def test_perceived_needs_a_policy_desk_boxes_and_a_row_and_refuses_the_truth_switches():
    env, rig = _env_and_rig()
    cases = [
        (None, {}, "perceived"),
        (_Still(), {"row_refresh": True}, "perceived"),
        (_Still(), {"scripted_place": True}, "perceived"),
    ]  # ``takeover`` under perceived is the MS7f wave 3 DAgger mode (test_ms7f_dagger.py)
    for policy, extra, reason in cases:
        with pytest.raises(loop.EvalFailure) as err:
            _rollout(env, rig, policy, perceived=True, **extra)
        assert err.value.reason == reason, extra
    with pytest.raises(loop.EvalFailure) as err:
        _rollout(env, rig, _Still(), contact_hold=True)
    assert err.value.reason == "contact_hold"
    env.close()
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    bare = MjLegoSortEnv(1, draw, parts=PARTS, cameras=False, gripper_mode="continuous")
    bare.reset(seed=11)
    rig = OracleRig(bare, bare.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    still = _Still()
    still.context_dim = 0
    with pytest.raises(loop.EvalFailure) as err:
        _rollout(bare, rig, still, perceived=True)
    bare.close()
    assert err.value.reason == "perceived"


PERCEIVED_ONLY = (
    "box_filter_xy",
    "release_gap",
    "rearm_hold",
    "box_walls",
    "marker_bias",
    "wrist_fix",
    "class_views",
)
"""The MS7d switches that need ``perceived``; ``hold_force`` and ``force_cap`` do not."""


def test_the_ms7d_switches_need_perceived_before_a_step():
    env, rig = _env_and_rig()
    served = _served(env)
    for name in PERCEIVED_ONLY:
        with pytest.raises(loop.EvalFailure) as err:
            _rollout(env, rig, _Still(), **{name: True})
        assert err.value.reason == name
    env.close()
    assert served == []


def test_cell_failure_cap_sets_the_failed_cycles_that_exclude_a_position(monkeypatch, capsys):
    """One brick the still policy never lifts: each cycle ends on its step budget and books a
    failure at the brick's perceived position, so the episode ends ``no_candidate`` after as
    many cycles as the cap, the constant's 3 or the switch's 5. The recorder's teacher keeps
    the constant, so the recorder refuses the switch."""
    from test_classifier_loop import _usage  # it imports this module

    heap = ["--heap_k", "3", "--desk_boxes", "4", "--gripper", "continuous"]
    line = _usage("mj_eval_heap", heap + ["--perceived", "--cell_failure_cap", "11"], capsys)
    assert line.startswith("EVAL_FAIL ") and "must lie in [1, 10]" in line
    line = _usage("mj_eval_heap", heap + ["--cell_failure_cap", "5"], capsys)
    assert line.startswith("EVAL_FAIL ") and "it needs --perceived" in line
    line = _usage("mj_generate_batch", ["--out_dir", "/tmp/x", "--cell_failure_cap", "5"], capsys)
    assert line.startswith("BATCH_FAIL usage: ") and "--cell_failure_cap is not built" in line
    env, rig = _env_and_rig()
    for bad in ({"cell_failure_cap": 5}, {"perceived": True, "cell_failure_cap": 0}):
        with pytest.raises(loop.EvalFailure, match="cell_failure_cap"):
            _rollout(env, rig, _Still(), **bad)
    env.close()
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 60)
    ends = []
    for cap in (heap_oracle.CELL_FAILURE_CAP, 5):
        env, rig = _env_and_rig(part_list=PARTS[:1], budget=8)
        _place(env, SPREAD[:1])
        payload = _rollout(env, rig, _Still(), budget=8, perceived=True, cell_failure_cap=cap)
        env.close()
        ep = payload["episodes"][0]
        on = payload["inference"]["loop"].get("cell_failure_cap")
        ends.append((ep["cycle_reasons"], ep["done_reason"], on))
    assert ends == [(["budget"] * 3, "no_candidate", None), (["budget"] * 5, "no_candidate", 5)]


def test_the_release_gap_selects_at_once_after_a_perceived_release(monkeypatch):
    """MS7d L2: after a cycle that ended ``placed`` or ``dropped`` the loop selects at once, as
    after a non-release end; with the switch off it waits for the full gap as in MS7c. The
    teacher drives as the policy and places its parts. The census records that no target lay
    in a box when it was selected."""
    real, after_stow = loop.release_ready, []

    def spy(opening, waited, after):
        after_stow.append(after)
        return real(opening, waited, after)

    monkeypatch.setattr(loop, "release_ready", spy)
    for gap in (False, True):
        after_stow.clear()
        env, rig = _env_and_rig()
        _place(env, SPREAD)
        payload = _rollout(
            env,
            rig,
            _Teacher(env, rig.oracle_factory),
            perceived=True,
            select_at_release=True,
            release_gap=gap,
        )
        env.close()
        ep = payload["episodes"][0]
        assert ep["cycle_reasons"][:2] == ["placed", "placed"] and ep["bricks_stowed"] == 3, ep
        assert after_stow and set(after_stow) == {not gap}, after_stow
        assert [row["target_in_box"] for row in ep["cycle_census"]] == [False] * 3
        assert payload["inference"]["loop"]["release_gap"] is gap


class _ScriptedHold:
    """A hold detector that reads nothing: it holds on the rollout's steps 10 to 14 and never
    lifts, so its hold ends with no lift on step 15."""

    def __init__(self):
        self.t = -1
        self.holding = self.lifted = self.started = self.lift_started = False
        self.ended = self.lost = self.blocked = False

    def update(self, measured_gap, commanded_gap, pick_z, entry=True):
        self.t += 1
        was, self.holding = self.holding, 10 <= self.t < 15
        self.started, self.ended = self.holding and not was, was and not self.holding


SERVING_RULES = ("squared_roll", "snap_shift", "snap_hold", "gated_gap")


def test_rearm_hold_serves_the_rules_again_after_a_hold_with_no_lift(monkeypatch):
    """MS7d L8 in the rollout, on the still policy and a scripted detector. ``dwell_close``
    fires after a dwell of 5 steps (at any pose here), and its close is spent 3 steps later.
    The hold on steps 10 to 14 turns the stop flag on from step 11. Latched (the switch off),
    it stays on to the budget. With the switch on, the hold's end clears it: the four serving
    rules read no stop from step 16 on, and ``dwell_close`` fires once more, at step 20, and
    not a third time. The census's hold fields are the latched run's. The switch refuses
    ``contact_hold``. Every other MS7d switch on, ``wrist_fix``'s serving row included, leaves
    the rules' stop flag as it is."""
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 60)
    monkeypatch.setattr(loop, "DWELL_STEPS", 5)
    monkeypatch.setattr(loop, "CLOSE_LATCH_STEPS", 3)
    monkeypatch.setattr(loop, "HoldDetector", _ScriptedHold)
    monkeypatch.setattr(loop, "dwell_count", lambda n, *args, held=False: 0 if held else n + 1)
    stops = []

    def spy(name, real):
        def rule(*args, held=False, **kwargs):
            stops.append((name, held))
            return real(*args, held=held, **kwargs)

        return rule

    for name in SERVING_RULES:
        monkeypatch.setattr(loop, name, spy(name, getattr(loop, name)))
    serving = dict.fromkeys(("square_wrist", "pick_snap", "close_gate", "dwell_close"), True)
    others = {name: True for name in MS7D_ON if name != "rearm_hold"}
    for extra, rearm in [(e, r) for e in ({}, others) for r in (False, True)]:
        stops.clear()
        env, rig = _env_and_rig(budget=1)
        _place(env, SPREAD)
        if rearm:
            with pytest.raises(loop.EvalFailure) as err:
                _rollout(env, rig, _Still(), perceived=True, rearm_hold=True, contact_hold=True)
            assert err.value.reason == "rearm_hold"
        payload = _rollout(
            env, rig, _Still(), budget=1, perceived=True, rearm_hold=rearm, **serving, **extra
        )
        env.close()
        want = [False] * 11 + [True] * 5 + [not rearm] * 44
        for name in SERVING_RULES:
            assert [held for n, held in stops if n == name] == want, (rearm, name)
        ep = payload["episodes"][0]
        row = ep["cycle_census"][0]
        assert ep["cycle_reasons"] == ["budget"], ep
        hold = (row["hold_detected"], row["lift_detected"], row["first_hold_at"])
        assert hold == (True, False, 10)
        assert (row["dwell_closes"], row["rearms"]) == (1 + rearm, int(rearm))
        assert payload["perception"]["rearms"] == int(rearm)


def test_the_box_walls_join_the_selectors_walls_only_with_the_switch(monkeypatch):
    """MS7d L9: with the switch off the selector ranks against the tray walls alone and the
    payload has no box-wall counts; with it on the walls of the perceived boxes follow the
    tray's. On three parts in the middle of the tray, far from every box, nothing else changes."""
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 40)
    walls, runs = [], {}
    real = loop._perceived_scene

    def spy(st, env, rig, i, k):
        walls.append(st.selector.walls)
        return real(st, env, rig, i, k)

    monkeypatch.setattr(loop, "_perceived_scene", spy)
    for on in (False, True):
        walls.clear()
        env, rig = _env_and_rig(budget=3, tray=True)
        _place(env, SPREAD)
        runs[on] = _rollout(env, rig, _Still(), budget=3, perceived=True, box_walls=on)
        boxes = env.perceived_box_poses[0].copy()
        env.close()
        expected = grasp.tray_walls() + (
            desk_boxes.box_walls(boxes, desk_boxes.interiors(4)) if on else ()
        )
        assert walls and all(w == expected for w in walls)
    off, on = runs[False], runs[True]
    assert off["episodes"][0]["cycles"] == 3 and on["episodes"] == off["episodes"]
    assert on["perception"] == {**off["perception"], "box_wall_rakes": 0, "box_wall_refusals": 0}
    assert [p["inference"]["loop"]["box_walls"] for p in (off, on)] == [False, True]


def test_the_census_reads_the_graders_any_box_region():
    """``TruthReads.boxed``, the read behind ``target_in_box``: a part on box 0's floor and one
    on box 3's are in a box, whoever's box it is, and a part on the desk is not."""
    env, _ = _env_and_rig()
    floor = desk_boxes.DESK_BOX_WALL + 0.006
    _place(env, [(*FIXED[0][:2], floor, 0.0), (*FIXED[3][:2], floor, 0.3), (0.2, 0.0, 0.006, 0.0)])
    boxed = loop.truth_reads(env, True).boxed
    assert loop.truth_reads(env, False).boxed is None
    env.close()
    assert boxed.tolist() == [[True, True, False]]


def test_the_row_marker_and_box_are_the_detections_at_selection_and_stay_frozen(monkeypatch):
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 40)
    env, rig = _env_and_rig(budget=12)
    _place(env, SPREAD)
    served, states = _served(env), _selections(monkeypatch)
    booked = []
    real_book = perception.PositionMemory.book
    monkeypatch.setattr(
        perception.PositionMemory,
        "book",
        lambda self, xy, n=1: (booked.append(tuple(xy)), real_book(self, xy, n))[1],
    )
    payload = _rollout(env, rig, _Still(), budget=12, perceived=True, select_at_release=True)
    boxes = env.perceived_box_poses[0].copy()
    class_to_box = env.class_to_box[0].tolist()
    true_boxes = env.desk_box_poses[0].copy()
    env.close()
    ep = payload["episodes"][0]
    assert ep["cycle_reasons"] == ["budget"] * 9 and ep["cycle_steps"] == [40] * 9, ep
    assert not np.allclose(boxes, true_boxes)  # the box knob's episode bias
    for j, target in enumerate(ep["cycle_targets"]):
        seen = next(d for d in states[j].detections if d.sim_index == target)
        box = boxes[class_to_box[parts.BASE_CLASSES.index(seen.cls)]]
        for row, box_row, marker, marked, _ in served[40 * j : 40 * (j + 1)]:
            assert np.allclose(row[:3], seen.centre, atol=1e-6)
            assert np.allclose(marker.numpy(), seen.centre, atol=1e-6) and marked == -1
            assert np.allclose(box_row, box, atol=1e-6)
        # A budget end books one failure where the target was perceived.
        assert booked[j] == pytest.approx(seen.centre[:2])
    # Three failures exclude a part: nine cycles, three a part, and then no candidate is left
    # under a budget of twelve.
    assert sorted(ep["cycle_targets"]) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert ep["done_reason"] == "no_candidate"
    assert payload["perception"] == {
        "perceived": True,
        "noise_scales": perception.format_scales(perception.parse_scales("")),
        "contact_hold": False,
        "selections": 9,
        "wrong_overhead_class": payload["perception"]["wrong_overhead_class"],
        "lifts_classified": 0,
        "wrong_wrist_class": 0,
        "third_views": 0,
        "votes_changed": 0,
        "wrong_voted_class": 0,
        "lifts_empty": 0,
        # The still tool stands 15 mm off the tray footprint: the empty selection looks again
        # for the retreat wait's cap, 15 steps at this budget, before it ends the episode.
        "look_again_waits": 1,
        "look_again_steps": int(loop.SELECT_WAIT_FRACTION * 40),
        "look_again_found": 0,
        "rearms": 0,
    }
    recorded = payload["inference"]["loop"]
    assert (recorded["perceived"], recorded["contact_hold"]) == (True, False)
    assert recorded["noise_scales"] == payload["perception"]["noise_scales"]


def test_the_marker_bias_correction_reaches_the_row_and_the_marker(monkeypatch):
    """MS7d L3. With ``tool`` at 0 the first look reads the episode bias, so each selection's
    target is the ``Perceiver``'s detection minus that bias, and so are the row and the frozen
    marker served for its cycle. Every perceive call looks once, and the estimate error reads
    0. With the switch off the loop builds no ``MarkerBias`` and the payload carries no marker
    count."""
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 40)
    raw, perceive = [], perception.Perceiver.perceive

    def spy(self, truth):
        raw.append((self.bias, perceive(self, truth)))
        return raw[-1][1]

    monkeypatch.setattr(perception.Perceiver, "perceive", spy)
    env, rig = _env_and_rig(budget=3)
    _place(env, SPREAD)
    served, states = _served(env), _selections(monkeypatch)
    payload = _rollout(
        env,
        rig,
        _Still(),
        budget=3,
        perceived=True,
        select_at_release=True,
        marker_bias=True,
        noise_scales={"tool": 0.0},
    )
    env.close()
    ep, seen = payload["episodes"][0], payload["perception"]
    assert ep["cycle_reasons"] == ["budget"] * 3, ep
    assert len(states) == len(raw) == seen["marker_looks"] > 3
    assert seen["marker_error_mm"] == pytest.approx(0.0, abs=1e-9)
    assert payload["inference"]["loop"]["marker_bias"] is True
    for j, target in enumerate(ep["cycle_targets"]):
        bias, before = raw[j]
        assert math.hypot(*bias[:2]) > 1e-4
        fixed = next(d for d in states[j].detections if d.sim_index == target)
        was = next(d for d in before.detections if d.sim_index == target)
        assert fixed.centre == pytest.approx([c - b for c, b in zip(was.centre, bias)], abs=1e-12)
        for row, _, marker, _, _ in served[40 * j : 40 * (j + 1)]:
            assert np.allclose(row[:3], fixed.centre, atol=1e-6)
            assert np.allclose(marker.numpy(), fixed.centre, atol=1e-6)
    monkeypatch.setattr(perception, "MarkerBias", _raises("MarkerBias"))
    env, rig = _env_and_rig(budget=1)
    _place(env, SPREAD)
    payload = _rollout(env, rig, _Still(), budget=1, perceived=True, select_at_release=True)
    env.close()
    assert set(loop.MARKER_COUNTS).isdisjoint(payload["perception"])


def test_a_part_missed_at_every_call_or_under_the_tool_is_never_selected(monkeypatch):
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 40)
    env, rig = _env_and_rig(budget=9)
    _place(env, SPREAD)
    payload = _rollout(env, rig, _Still(), budget=9, perceived=True, noise_scales={"miss": 1000.0})
    env.close()
    ep = payload["episodes"][0]
    assert ep["cycles"] == 0 and ep["done_reason"] == "no_candidate", ep
    # Part 1 lies under the still tool: the other two take three failures each, then no
    # candidate is left.
    env, rig = _env_and_rig(budget=9)
    tool = rig.pick_point(0).tolist()
    _place(env, [SPREAD[0], (tool[0], tool[1], 0.006, 0.0), SPREAD[2]])
    payload = _rollout(env, rig, _Still(), budget=9, perceived=True, noise_scales=ZERO)
    env.close()
    ep = payload["episodes"][0]
    assert sorted(ep["cycle_targets"]) == [0, 0, 0, 2, 2, 2], ep
    assert ep["done_reason"] == "no_candidate"


def _one_stuck_brick(monkeypatch):
    """One 2x4 brick at yaw 45 degrees 25 mm off the tray's left wall: under a 20 degree band
    it ranks no grasp, so the selector plans a rake."""
    monkeypatch.setattr(grasp, "WALL_BAND_RAD", math.radians(20.0))
    env, rig = _env_and_rig(part_list=[parts.make_part("brick", 2, 4)], budget=3, tray=True)
    (_, _), (y0, _) = layout.tray_bounds()
    _place(env, [(layout.TRAY_CENTER[0], y0 + 0.025, 0.0049, math.radians(45.0))])
    return env, rig


def test_a_rake_that_hands_back_over_its_part_looks_again_and_finds_it(monkeypatch):
    """The rake hands the env back with the tool over the part it raked, which the tool hides.
    Without the look-again the loop saw an empty tray and ended the episode; with it the loop
    perceives again while the policy takes the arm home, finds the part and opens a cycle."""
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 200)
    found = {}
    for looks in (False, True):
        env, rig = _one_stuck_brick(monkeypatch)
        if not looks:
            monkeypatch.setattr(loop, "look_again", lambda *args: False)
        payload = _rollout(
            env, rig, _Home(), budget=3, perceived=True, noise_scales=ZERO, scripted_rake=True
        )
        env.close()
        monkeypatch.undo()
        monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 200)
        found[looks] = (payload["episodes"][0], payload["perception"])
    old, _ = found[False]
    assert old["rakes"] >= 1 and old["cycles"] == 0 and old["done_reason"] == "no_candidate"
    ep, seen = found[True]
    assert ep["rakes"] >= 1 and ep["cycles"] >= 1 and ep["cycle_targets"][0] == 0, ep
    assert seen["look_again_waits"] >= 1 and seen["look_again_found"] >= 1, seen
    assert 0 < seen["look_again_steps"] < int(loop.SELECT_WAIT_FRACTION * 200), seen


def test_an_empty_tray_ends_at_once_with_the_tool_clear_and_after_the_cap_without(monkeypatch):
    """Three parts out of reach, none in the tray. With the tool swung clear of the tray the
    first empty selection ends the episode; at the home pose, 15 mm off the tray, the loop
    looks again for the retreat wait's cap first."""
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 40)
    out_of_reach = [(0.45, y, 0.006, 0.0) for y in (-0.05, 0.0, 0.05)]
    for swing, steps in ((1.0, 1), (0.0, int(loop.SELECT_WAIT_FRACTION * 40) + 1)):
        env, rig = _env_and_rig(budget=3)
        d, h = env.datas[0], env.handles
        d.qpos[h.joint_qpos[0]] += swing  # the base turns the tool off the tray
        d.ctrl[list(h.arm_actuators)] = d.qpos[list(h.joint_qpos[:6])]
        _place(env, out_of_reach)
        clear = loop.tool_clear_of_tray(rig.pick_point(0)[:2].tolist(), 0.0)
        payload = _rollout(env, rig, _Still(), budget=3, perceived=True, noise_scales=ZERO)
        env.close()
        ep, seen = payload["episodes"][0], payload["perception"]
        assert clear is bool(swing)
        assert ep["cycles"] == 0 and ep["done_reason"] == "no_candidate", ep
        assert payload["steps_run"] == steps
        assert seen["look_again_steps"] == steps - 1 and seen["look_again_found"] == 0, seen
        assert seen["look_again_waits"] == int(not swing)


def test_at_zero_scales_the_loop_serves_the_truth_paths_rows_and_boxes(monkeypatch):
    """With every knob at 0 the perception is the truth minus the structural rules, and on
    three spread parts nothing hides: the selections, rows, boxes and cycle ends are the truth
    path's (with --part_failures, the scope perceived forces). Only the marker differs: a frozen
    point at the selected centre, and no live target_part."""
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 40)
    runs = {}
    for perceived in (False, True):
        env, rig = _env_and_rig(budget=9)
        _place(env, SPREAD)
        centres = [env.scene[f"brick_{b}"].data.root_pos_w[0].tolist() for b in range(3)]
        served = _served(env)
        extra = {"perceived": True, "noise_scales": ZERO} if perceived else {}
        payload = _rollout(
            env, rig, _Still(), budget=9, select_at_release=True, part_failures=True, **extra
        )
        env.close()
        runs[perceived] = (payload["episodes"][0], served)
    (truth_ep, truth_served), (seen_ep, seen_served) = runs[False], runs[True]
    for key in ("cycle_targets", "cycle_reasons", "cycle_steps", "done_reason"):
        assert seen_ep[key] == truth_ep[key], key
    assert seen_ep["cycle_targets"] == [0, 0, 0, 2, 2, 2, 1, 1, 1]
    assert len(seen_served) == len(truth_served)
    for (row, box, marker, marked, _), (t_row, t_box, _, t_marked, _) in zip(
        seen_served, truth_served
    ):
        assert np.array_equal(row, t_row) and np.array_equal(box, t_box)
        assert marked == -1 and t_marked >= 0
        assert np.allclose(marker.numpy(), centres[t_marked], atol=1e-6)


def test_the_class_at_lift_classifies_each_lift_and_a_placed_stow_counts(monkeypatch):
    """The teacher drives as the policy: each lift draws the class of the part between the
    jaws, each release over a box ends its cycle placed once the jaws open, and at the
    episode's end every placed part rests in its own box, so each placed cycle counts as a
    stow. The last part lands before the jaws open, so the env's success ends that cycle
    stowed. At a class scale that always confuses, every overhead and wrist class is wrong and
    the teacher still stows. Under ``class_views`` at a persistent share of 1 (rho 2) every part
    is persistent: all its views name one wrong class, so no vote changes the wrist class. At a
    share of 0 (rho 0) every view is wrong on its own draws, so the wrist and the overhead
    disagree on some parts, and the vote draws a third view on those."""
    rows = ((0.0, 1.0, False), (1000.0, 1.0, False), (1000.0, 2.0, True), (1000.0, 0.0, True))
    for scale, rho, views in rows:
        wrong = 3 if scale else 0
        env, rig = _env_and_rig()
        _place(env, SPREAD)
        payload = _rollout(
            env,
            rig,
            _Teacher(env, rig.oracle_factory),
            perceived=True,
            noise_scales={"class": scale, "rho": rho},
            select_at_release=True,
            class_views=views,
        )
        env.close()
        ep, seen = payload["episodes"][0], payload["perception"]
        assert ep["cycle_reasons"] == ["placed", "placed", "stowed"], ep
        assert ep["cycle_outcomes"] == [True] * 3, ep
        assert ep["cycle_classes"] == ["stowed"] * 3 and sorted(ep["stowed_parts"]) == [0, 1, 2]
        assert ep["bricks_stowed"] == 3 and payload["cycles_per_stowed"] == pytest.approx(1.0)
        assert (seen["lifts_classified"], seen["lifts_empty"]) == (3, 0), seen
        assert seen["wrong_wrist_class"] == wrong and seen["wrong_overhead_class"] == wrong
        thirds, changed, wrong_voted = (
            seen[name] for name in ("third_views", "votes_changed", "wrong_voted_class")
        )
        assert wrong_voted == (wrong if views else 0), seen
        if rho == 0.0:
            assert 0 < thirds <= 3 and changed <= thirds, seen
        else:
            assert thirds == changed == 0, seen


def test_credit_placed_credits_a_stowed_part_once_to_its_last_placed_cycle():
    st = loop.EnvState(selector=None)
    rows = [(0, "placed"), (1, "placed"), (0, "placed"), (2, "dropped"), (1, "stowed")]
    for b, reason in rows:
        st.reasons.append(reason)
        st.outcomes.append(reason == "stowed")
        st.classes.append("stowed" if reason == "stowed" else "lifted_lost")
        st.census.append({"last_held": b})
    st.stowed_parts = [1]  # the env's success credited part 1 to the last cycle
    loop._credit_placed(st, [True, True, True])
    assert st.outcomes == [False, False, True, False, True]
    assert st.classes == ["lifted_lost", "lifted_lost", "stowed", "lifted_lost", "stowed"]
    assert st.stowed_parts == [1, 0]


def _wrist_rollout(wrist_fix, scales, spies=()):
    """The teacher drives the three spread parts under ``pick_snap`` and ``square_wrist``, with
    ``wrist_fix`` on or off. ``spies`` are (name, wrapper) pairs set on ``loop`` for the run.
    Returns the payload and ``_served``'s rows."""
    env, rig = _env_and_rig()
    _place(env, SPREAD)
    served = _served(env)
    with pytest.MonkeyPatch.context() as mp:
        for name, wrap in spies:
            mp.setattr(loop, name, wrap(env, getattr(loop, name)))
        payload = _rollout(
            env,
            rig,
            _Teacher(env, rig.oracle_factory),
            perceived=True,
            pick_snap=True,
            square_wrist=True,
            wrist_fix=wrist_fix,
            noise_scales=scales,
        )
    env.close()
    return payload, served


def test_a_wrist_look_that_misses_keeps_the_overhead_row_as_with_the_switch_off():
    """MS7d L4: when every look misses, the loop serves exactly what it serves with the switch
    off, the payload adds only the wrist counts, and the Perceiver's draws are untouched."""
    scales = {"wrist_miss": 1000.0}
    (off, off_served), (on, on_served) = (_wrist_rollout(w, scales) for w in (False, True))
    assert len(on_served) == len(off_served) > 0
    for (row, box, marker, _, action), b in zip(off_served, on_served):
        assert np.array_equal(row, b[0]) and np.array_equal(box, b[1]) and torch.equal(action, b[4])
        assert torch.equal(marker.nan_to_num(), b[2].nan_to_num())
    assert on["episodes"][0]["cycle_reasons"] == off["episodes"][0]["cycle_reasons"]
    seen = on["perception"]
    assert seen["wrist_looks"] >= 3 and seen["wrist_misses"] == seen["wrist_looks"], seen
    assert seen["wrist_covered"] == seen["wrist_shift_mm"] == seen["wrist_turn_deg"] == 0
    assert {k: v for k, v in seen.items() if k not in loop.WRIST_COUNTS} == off["perception"]
    assert not set(loop.WRIST_COUNTS) & set(off["perception"])


def test_a_wrist_look_at_zero_noise_snaps_to_the_true_centre_and_leaves_the_policy_row():
    """MS7d L4, with ``wrist`` and ``wrist_bias`` at 0: after a look the row ``pick_snap``
    reads is centred on the target's true centre at the look, and nothing else moves; the
    policy's row stays the detection at selection, where the frozen marker stands."""
    events = []

    def look_spy(env, real):
        def spy(env_, rig, i, b):
            got = real(env_, rig, i, b)
            true = env.scene[f"brick_{b}"].data.root_pos_w[i] - env.scene.env_origins[i]
            events.append(("look", got, true[:2].tolist()))
            return got

        return spy

    def snap_spy(env, real):
        def spy(cmd, pick, row, *args, **kwargs):
            events.append(("snap", [float(v) for v in row], env.context_target[0].tolist()))
            return real(cmd, pick, row, *args, **kwargs)

        return spy

    scales = {"wrist": 0.0, "wrist_bias": 0.0, "wrist_miss": 0.0}
    payload, served = _wrist_rollout(
        True, scales, [("wrist_look_truth", look_spy), ("snap_shift", snap_spy)]
    )
    seen = payload["perception"]
    assert seen["wrist_looks"] >= 3 and seen["wrist_misses"] == seen["wrist_covered"] == 0, seen
    assert seen["wrist_shift_mm"] > 0.0
    last, snapped = None, 0
    for kind, got, other in events:
        if kind == "look":
            assert got[:2] == pytest.approx(other, abs=1e-6) and got[3] is False
            last = got[:2]
        elif got[:2] != other[:2]:  # the serving row moved off the policy's row: a look's
            assert tuple(got[:2]) == last and [got[2], *got[5:]] == [other[2], *other[5:]]
            snapped += 1
    assert snapped > 0
    for row, _, marker, _, _ in served:
        if not marker.isnan().any():
            assert np.allclose(row[:3], marker.numpy(), atol=1e-6)


# -- the no-truth test ------------------------------------------------------------------------


class _TruthlessEnv:
    """The env as the loop sees it in the no-truth run: every member passes through to the
    real env, except that ``destination_bins`` and ``desk_box_poses`` raise once the loop has
    written ``perceived_box_poses``, its last setup step. The real env, the rig and the teacher
    policy still read them: the physics and the grader are not the loop's decisions."""

    def __init__(self, env):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_strict", False)

    def __getattr__(self, name):
        strict = ("destination_bins", "desk_box_poses", "desk_box_sizes", "destination_mask")
        if self._strict and name in strict + ("class_boxes",):
            raise AssertionError(f"the loop read env.{name} after its setup")
        return getattr(self._env, name)

    def __setattr__(self, name, value):
        setattr(self._env, name, value)
        if name == "perceived_box_poses":
            object.__setattr__(self, "_strict", True)


def _garbage_reads(env, per_part, gen):
    """A ``TruthReads`` of noise with the right shapes and ranges: what the metrics get in the
    no-truth run. Drawn from its own generator: the teacher's command noise draws from torch's
    global one, and sharing it would change the teacher's actions."""
    n, k = env.num_envs, env.cfg.num_bricks
    quat = torch.randn(n, k, 4, generator=gen)
    return loop.TruthReads(
        per_brick=torch.rand(n, k, generator=gen) < 0.5,
        in_bin=torch.rand(n, k, generator=gen) < 0.5,
        pos=torch.rand(n, k, 3, generator=gen) * 0.3,
        speed=torch.rand(n, k, generator=gen),
        quat=quat / quat.norm(dim=-1, keepdim=True),
        held=torch.randint(-1, k, (n,), generator=gen),
        resting_box=torch.randint(-1, 4, (n, k), generator=gen) if per_part else None,
        boxed=torch.rand(n, k, generator=gen) < 0.5 if per_part else None,
    )


def _raises(name):
    def read(*args, **kwargs):
        raise AssertionError(f"a decision read the truth through {name}")

    return read


TRUTH_DECISION_READS = (
    "per_brick_stowed",
    "parts_in_desk_boxes",
    "parts_resting_box",
    "_brick_states",
    "held_brick_from_forces",
    "resting_out_of_reach",
    "cycle_end",
    "released_after_lift",
)
"""The loop's truth reads besides the three accessors: what its decisions read before MS7c."""


def _teacher_scene(
    monkeypatch, class_names=parts.BASE_CLASSES, box_draw=0, categories=False, **env_kwargs
):
    env, rig = _env_and_rig(
        class_names=class_names, box_draw=box_draw, categories=categories, **env_kwargs
    )
    _place(env, SPREAD)
    switches = {
        "select_at_release": True,
        "squeeze_floor": True,
        "square_wrist": True,
        "pick_snap": True,
        "dwell_close": True,
        "part_failures": True,
        "adopt_held": True,
        "park_done": True,
    }
    return env, rig, _Teacher(env, rig.oracle_factory), 4, switches


def _eight_class_scene(monkeypatch):
    """The teacher scene under the eight MS7e classes, the one MS7e switch this commit builds
    into the loop (MS7e design section 10.2)."""
    return _teacher_scene(monkeypatch, parts.PART_CLASSES)


def _box_draw_scene(monkeypatch):
    """The teacher scene under the MS7e box draw (``--box_draw 6``): the reset draws the count,
    each box's interior and pose, and the loop perceives each box's size (design section 4.3)."""
    return _teacher_scene(monkeypatch, parts.PART_CLASSES, box_draw=6)


def _ms7f_scene(monkeypatch):
    """The teacher scene with every MS7f switch the commit builds (``batch_loop.MS7F_READY``):
    the six categories over the drawn boxes, a category's second boxes chosen by the loop's
    own release record (MS7f design section 6.2). A track that readies a loop switch adds it
    here, so the no-truth test runs with every ready switch on (design section 12.2).
    ``classifier`` and ``passenger_drop`` (track C) read a stub TorchScript model on a stand-in
    narrow view (the cameras are off). It reads no passenger: every lift of this tray scene is
    over the tray interior, where a passenger read ends the cycle, and the scene's one lifting
    cycle must place. ``test_classifier_loop`` runs the drop on its own tray scene."""
    from test_classifier_loop import _classifier  # it imports this module

    # Track S's teacher pace and waits drive the rig, which serves no loop decision, and
    # end_at_done ends the recorder's episodes.
    teacher = {
        "transport_speed",
        "descend_speed",
        "close_steps",
        "lift_hold_steps",
        "release_steps",
        "close_guard",
        "quiet_landing",
        "landing_steps",
        "end_at_done",
        "net_stall",
    }
    ready = {"categories", "deep_tray", "tray_refill", "box_empty", "book_miss"}
    # abandon_cycle is the recorder's and near_tracking the rig's watchdog: no loop decision
    ready |= {"lift_budget", "abandon_cycle", "near_tracking", "classifier", "passenger_drop"}
    ready |= {"lift_view_steps"}
    ready |= {"missed_lift"}  # the riders track: a carry the hold detector missed
    ready |= {"cell_failure_cap", "carry_votes", "stall_end"}
    ready |= {"pile_pgs"}  # the env's solver: no loop decision either
    # MS7g's fused vote refuses carry_votes, which this scene turns on; it changes only how the
    # classifier's log probabilities are summed (test_vote_weights runs it).
    ready |= {"vote_weights"}
    ready |= {"cls_flip"}  # MS7g D10: the scene's classifier reads every frame mirrored too
    assert batch_loop.MS7F_READY == ready | teacher, "turn it on in this scene"
    # Two of the three bricks start in the tray and one part fills a box, so the refill and
    # an emptying both run between cycles. Set before the build: the box draw reads the ratio.
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
        pile_pgs=True,
    )
    env.narrow_overhead = lambda i, point: np.full((224, 224, 3), 255, dtype=np.uint8)
    stub = _classifier(Path(tempfile.mkdtemp()), "special", "brick", names=parts.PART_CATEGORIES)
    stub.flip = True
    # A lift budget under the 800-step cycle budget: the tray scene's first three cycles lift
    # nothing and end on it, and the fourth places its part at step 487 (150 cuts it too).
    on = {"box_empty": True, "book_miss": True, "lift_budget": 750}
    on |= {"classifier": stub, "passenger_drop": True, "lift_view_steps": 6, "missed_lift": True}
    # carry_votes adds reads to the class_views vote, so the scene turns the vote on too.
    on |= {"cell_failure_cap": 5, "carry_votes": 2, "class_views": True, "stall_end": True}
    # At the recorder's 60-step span every cycle of this tray scene closes on nothing and ends
    # on it, so none places; at 220 the fourth still places. ``_stall_end_scene`` fires it at 60.
    monkeypatch.setattr(loop, "CLOSED_NO_PINCH_STEPS", 220)
    return env, rig, policy, budget, switches | on


def _ms7g_selector_scene(monkeypatch):
    """The MS7f scene with the two MS7g selector switches on (MS7g design 5.3 and 5.4): the
    clear grasps rank first and the tray wall's overhang refuses a part, on perceived reads."""
    env, rig, policy, budget, switches = _ms7f_scene(monkeypatch)
    return env, rig, policy, budget, switches | {"clear_first": True, "overhang_guard": True}


def _thin_snap_scene(monkeypatch):
    """The teacher scene with a 2x4 plate and a 2x2 tile for two of its bricks, under the eight
    MS7e classes and ``thin_snap`` (MS7e ledger R13 S2): the loop serves flat rows."""
    flat = [
        parts.make_part("brick", 2, 4),
        parts.make_part("plate", 2, 4),
        parts.make_part("tile", 2, 2),
    ]
    env, rig = _env_and_rig(part_list=flat, class_names=parts.PART_CLASSES)
    _place(env, [SPREAD[0], (0.195, 0.0, 0.0026, -0.8), (0.195, 0.045, 0.0026, 1.2)])
    switches = {
        "select_at_release": True,
        "squeeze_floor": True,
        "square_wrist": True,
        "pick_snap": True,
        "dwell_close": True,
        "part_failures": True,
        "adopt_held": True,
        "park_done": True,
        "thin_snap": True,
    }
    return env, rig, _Teacher(env, rig.oracle_factory), 4, switches


def _snap_closed_scene(monkeypatch):
    """The flat scene under ``snap_closed`` too (MS7e flat study, 2026-09-30): ``snap_hold``
    reads the measured jaw gap, a proprioceptive read."""
    env, rig, policy, budget, switches = _thin_snap_scene(monkeypatch)
    return env, rig, policy, budget, switches | {"snap_closed": True}


def _stall_end_scene(monkeypatch):
    """The flat scene under ``stall_end`` (MS7f): two of its four cycles close on nothing and end
    on it, and the other two place."""
    env, rig, policy, budget, switches = _thin_snap_scene(monkeypatch)
    return env, rig, policy, budget, switches | {"stall_end": True}


CARRY_DROP = (0.195, 0.0, 0.10)
"""m: where the carry scene's teacher releases every part: over the parts it picks from."""


def _carry_scene(monkeypatch):
    """The teacher scene under ``carry_guard`` (MS7g), its teacher releasing every part over the
    parts it picks from (``EnvOracle.release_point``), away from every box: the guard holds the
    jaws, and each still guard hands the part to the rig's perceived carry."""
    env, rig, policy, budget, switches = _teacher_scene(monkeypatch)
    policy.rig.oracles[0].release_point = torch.tensor(CARRY_DROP)
    return env, rig, policy, budget, switches | {"carry_guard": True}


def test_a_box_draw_episode_records_its_boxes(monkeypatch):
    """The census rebuilds each drawn box from the episode record (MS7e ledger R11)."""
    env, rig, policy, budget, switches = _box_draw_scene(monkeypatch)
    n = int(env.box_count[0])
    poses, sizes = env.desk_box_poses[0, :n].tolist(), env.desk_box_sizes[0, :n].tolist()
    ep = _rollout(env, rig, policy, budget=budget, perceived=True, **switches)["episodes"][0]
    env.close()
    assert ep["box_count"] == n and ep["box_draw_rounds"] == 0
    assert np.allclose(ep["box_poses"], poses, atol=1e-4)
    assert np.allclose(ep["box_sizes"], sizes, atol=1e-4)


def _rake_scene(monkeypatch):
    """A brick stuck 25 mm off the tray's left wall at 45 degrees, which ranks no grasp under a
    20 degree band so the selector plans a rake, and a 2x2 brick in the open tray."""
    monkeypatch.setattr(grasp, "WALL_BAND_RAD", math.radians(20.0))
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 60)
    pair = [parts.make_part("brick", 2, 4), parts.make_part("brick", 2, 2)]
    env, rig = _env_and_rig(part_list=pair, budget=4, tray=True)
    (_, _), (y0, _) = layout.tray_bounds()
    x = layout.TRAY_CENTER[0]
    _place(env, [(x, y0 + 0.025, 0.0049, math.radians(45.0)), (x, y0 + 0.09, 0.0049, 0.3)])
    switches = {
        "select_at_release": True,
        "scripted_rake": True,
        "rake_turn": True,
        "square_wrist": True,
    }
    return env, rig, _Still(), 4, switches


def _wall_band_scene(monkeypatch):
    """MS7e ledger R13 S3 (``--wall_band_cont``): a 1x2 plate at 30 degrees, 13 mm off the left
    wall of the 30 degree tray, which ranks a grasp only at the continuous band, and a 2x2
    brick in the open tray. The selector and the teacher policy both rank at that band."""
    pair = [parts.by_name(["plate_1x2"])[0], parts.make_part("brick", 2, 2)]
    env, rig = _env_and_rig(
        part_list=pair,
        tray=True,
        class_names=parts.PART_CLASSES,
        tray_lean=30.0,
        wall_band_cont=True,
    )
    (_, _), (y0, _) = layout.tray_bounds()
    x = layout.TRAY_CENTER[0]
    _place(env, [(x, y0 + 0.013, 0.0016, math.radians(30.0)), (x, y0 + 0.09, 0.0049, 0.3)])
    switches = {"select_at_release": True, "square_wrist": True, "pick_snap": True}
    return env, rig, _Teacher(env, rig.oracle_factory), 4, switches


MS7D_ON = {
    "box_filter_xy": True,
    "release_gap": True,
    "rearm_hold": True,
    "box_walls": True,
    "marker_bias": True,
    "wrist_fix": True,
    "class_views": True,
    "hold_force": True,
    "force_cap": True,
}
"""All nine MS7d switches: the no-truth test runs with them off and on."""


def test_every_ms7d_switch_at_once_runs_the_teacher_scene_to_its_end(monkeypatch):
    """The nine MS7d levers together at the nominal knobs on the perceived teacher scene: the
    teacher stows every part, one classified lift a cycle, the marker, the wrist and the force
    sensor act, and the payload records every switch, each lever's counts and the torque
    block."""
    env, rig, policy, budget, switches = _teacher_scene(monkeypatch)
    payload = _rollout(env, rig, policy, budget=budget, perceived=True, **switches, **MS7D_ON)
    env.close()
    ep, seen, torque = payload["episodes"][0], payload["perception"], payload["torque"]
    assert ep["done_reason"] == "all_stowed" and seen["lifts_classified"] == ep["cycles"], ep
    assert {name: payload["inference"]["loop"][name] for name in MS7D_ON} == MS7D_ON
    assert set(loop.BOX_WALL_COUNTS + loop.MARKER_COUNTS + loop.WRIST_COUNTS) <= set(seen)
    assert seen["marker_looks"] > 0 and seen["wrist_looks"] > 0, seen
    assert torque["hold_force"] and torque["force_cap"] and torque["approach_force"]["steps"] > 0


@pytest.mark.parametrize("ms7d", [{}, MS7D_ON])
@pytest.mark.parametrize(
    "scene",
    [
        _teacher_scene,
        _rake_scene,
        _eight_class_scene,
        _box_draw_scene,
        _wall_band_scene,
        _thin_snap_scene,
        _snap_closed_scene,
        _stall_end_scene,
        _carry_scene,
        _ms7f_scene,
        _ms7g_selector_scene,
    ],
)
def test_no_decision_reads_the_truth(scene, ms7d, monkeypatch):
    """MS7c design section 10: under perceived no decision path reads the truth.

    Every truth read of the loop goes through one of three accessors: ``perception_truth``
    (the input of a selection), ``wrist_class`` (the part between the jaws at a lift) and
    ``truth_reads`` (the metrics). The test runs the same perceived rollout twice. The second
    run replaces the metric accessor with noise (``_garbage_reads``), makes every other truth
    read the decision paths made before MS7c raise (``TRUTH_DECISION_READS``, the loop rig's
    ``snapshot``, and ``env.destination_bins`` and ``env.desk_box_poses`` after the setup,
    ``_TruthlessEnv``), and leaves the two decision accessors alone. A decision that read any
    of these would raise or, fed noise, serve something else; so the second run must serve
    exactly what the first did on every step: the target row, the box row, the marker point
    and the action, and end the same cycles the same way. The teacher scene lifts and places
    parts through the serving rules' stop flag; the rake scene runs a rake takeover, and the
    carry scene the rig's perceived carry (MS7g ``carry_guard``), whose rig must read neither
    ``snapshot``, the contacts, a part's pose nor its pinch either.
    """
    env, rig, policy, budget, switches = scene(monkeypatch)
    switches |= ms7d
    served = _served(env)
    first = _rollout(env, rig, policy, budget=budget, perceived=True, **switches)
    env.close()

    env, rig, policy, budget, switches = scene(monkeypatch)
    switches |= ms7d
    blind = _served(env)
    metric_calls, gen = [], torch.Generator().manual_seed(7)

    def garbage(env, per_part):
        metric_calls.append(1)
        return _garbage_reads(env, per_part, gen)

    monkeypatch.setattr(loop, "truth_reads", garbage)
    for name in TRUTH_DECISION_READS:
        monkeypatch.setattr(loop, name, _raises(name))
    rig.snapshot = _raises("rig.snapshot")
    rig._brick_pos = _raises("rig._brick_pos")
    rig.pinch_evidence = _raises("rig.pinch_evidence")
    monkeypatch.setattr(oracle_runtime, "held_brick_from_forces", _raises("the rig's contacts"))
    second = _rollout(_TruthlessEnv(env), rig, policy, budget=budget, perceived=True, **switches)
    env.close()

    assert metric_calls and len(blind) == len(served) > 0
    for s, (a, b) in enumerate(zip(served, blind)):
        row, box, marker, marked, action = a
        assert np.array_equal(row, b[0]) and np.array_equal(box, b[1]), s
        assert torch.equal(marker.isnan(), b[2].isnan()), s
        assert torch.equal(marker.nan_to_num(), b[2].nan_to_num()), s
        assert marked == b[3] == -1 and torch.equal(action, b[4]), s
    ep, blind_ep = first["episodes"][0], second["episodes"][0]
    for key in ("cycle_reasons", "cycle_targets", "cycle_steps", "rakes", "done_reason"):
        assert ep[key] == blind_ep[key], key
    assert first["perception"] == second["perception"]
    assert first["perception"]["selections"] > 0
    if scene is _rake_scene:
        assert ep["rakes"] >= 1 and "budget" in ep["cycle_reasons"], ep
    elif scene is _wall_band_scene:
        assert ep["cycle_reasons"] == ["placed", "placed"] and ep["rakes"] == 0, ep
    else:
        assert "placed" in ep["cycle_reasons"] and first["perception"]["lifts_classified"] > 0
    if scene is _stall_end_scene and not ms7d:  # the MS7d levers place every cycle there
        assert "stall_end" in ep["cycle_reasons"], ep
    if scene is _carry_scene:
        assert "placed" in {row["handoff"] for row in ep["cycle_census"]}, ep["cycle_census"]
        guard = [[row[k] for k in loop.CARRY_GUARD_CENSUS] for row in ep["cycle_census"]]
        assert guard == [
            [row[k] for k in loop.CARRY_GUARD_CENSUS] for row in blind_ep["cycle_census"]
        ]
