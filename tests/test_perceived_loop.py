"""The heap loop under ``perceived`` (MS7c design section 6): the pure rules, the served row,
box and frozen marker, the failure memory by position, the class at lift, and the no-truth
test (MS7c design section 10), on the three-part MuJoCo heap with four desk boxes, cameras off.
"""

import math
import random

import numpy as np
import pytest
import torch

from maker_arm_lab.core import desk_boxes, grasp, layout, loop, oracle, parts, perception
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig
from test_heap_loop import SPREAD
from test_mj_desk_boxes import FIXED, _TeacherPolicy
from test_mj_heap_oracle import PARTS, _place

ZERO = perception.parse_scales("all=0")


def _env_and_rig(part_list=PARTS, budget=4, tray=False):
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1,
        draw,
        parts=part_list,
        cameras=False,
        desk_boxes=4,
        tray=tray,
        gripper_mode="continuous",
        context="camera_box_target",
        episode_length_s=loop.episode_length_s(budget),
    )
    env.reset(seed=11)
    env.place_desk_boxes(0, FIXED)
    rng = random.Random(1)

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


def test_over_box_reads_each_boxs_yawed_footprint():
    _, (h, _, _) = desk_boxes.stow_region(2)  # the footprint is square
    boxes = [(0.10, 0.20, 0.0), (0.10, -0.20, math.pi / 4)]
    assert loop.over_box((0.10, 0.20), boxes) and loop.over_box((0.10, -0.20), boxes)
    assert loop.over_box((0.10 + 0.9 * h, 0.20 + 0.9 * h), boxes)  # box 0's corner
    assert not loop.over_box((0.10 + 1.1 * h, 0.20), boxes)
    # Box 1 turned 45 degrees: a corner points along world y, and the square's own corner
    # direction now crosses an edge.
    assert loop.over_box((0.10, -0.20 + 1.25 * h), boxes)
    assert not loop.over_box((0.10 + 0.9 * h, -0.20 + 0.9 * h), boxes)
    assert not loop.over_box((0.30, 0.0), boxes)


def test_the_class_at_lift_serves_the_box_of_the_class_drawn_for_the_part_in_the_jaws():
    boxes = {"slope": 2, "tall": 3, "narrow": 1, "wide": 0}
    draws = []

    def classify(cls):
        draws.append(cls)
        return "slope" if cls == "tall" else cls  # a confusion on tall parts

    # Nothing between the jaws (a false lift): no draw, the row stays.
    assert loop.lift_class_box(None, classify, boxes, 0) == (None, None) and draws == []
    # The served box already fits the drawn class: no switch.
    assert loop.lift_class_box("wide", classify, boxes, 0) == ("wide", None)
    # Another class, or a confused one: the box of the class drawn, not of the true one.
    assert loop.lift_class_box("narrow", classify, boxes, 0) == ("narrow", 1)
    assert loop.lift_class_box("tall", classify, boxes, 0) == ("slope", 2)
    assert draws == ["wide", "narrow", "tall"]


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


# -- the rollout --------------------------------------------------------------------------------


def test_perceived_needs_a_policy_desk_boxes_and_a_row_and_refuses_the_truth_switches():
    env, rig = _env_and_rig()
    cases = [
        (None, {}, "perceived"),
        (_Still(), {"row_refresh": True}, "perceived"),
        (_Still(), {"scripted_place": True}, "perceived"),
        (_Still(), {"takeover": loop.TakeoverConfig(random_interval=None)}, "perceived"),
    ]
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
        box = boxes[class_to_box[parts.PART_CLASSES.index(seen.cls)]]
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
        "noise_scales": "pos=1,heading=1,size=1,box=1,miss=1,class=1",
        "contact_hold": False,
        "selections": 9,
        "wrong_overhead_class": payload["perception"]["wrong_overhead_class"],
        "lifts_classified": 0,
        "wrong_wrist_class": 0,
        "lifts_empty": 0,
        # The still tool stands 15 mm off the tray footprint: the empty selection looks again
        # for the retreat wait's cap, 15 steps at this budget, before it ends the episode.
        "look_again_waits": 1,
        "look_again_steps": int(loop.SELECT_WAIT_FRACTION * 40),
        "look_again_found": 0,
    }
    recorded = payload["inference"]["loop"]
    assert (recorded["perceived"], recorded["contact_hold"]) == (True, False)
    assert recorded["noise_scales"] == payload["perception"]["noise_scales"]


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
            monkeypatch.setattr(loop, "look_again", lambda tool_xy, lean_deg, streak: False)
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
    jaws, each release over a box ends its cycle placed, and at the episode's end every placed
    part rests in its own box, so each placed cycle counts as a stow. At a class scale that
    always confuses, every overhead and wrist class is wrong and the teacher still stows."""
    for scale, wrong in ((0.0, 0), (1000.0, 3)):
        env, rig = _env_and_rig()
        _place(env, SPREAD)
        payload = _rollout(
            env,
            rig,
            _Teacher(env, rig.oracle_factory),
            perceived=True,
            noise_scales={"class": scale},
            select_at_release=True,
        )
        env.close()
        ep, seen = payload["episodes"][0], payload["perception"]
        assert ep["cycle_reasons"] == ["placed"] * 3 and ep["cycle_outcomes"] == [True] * 3, ep
        assert ep["cycle_classes"] == ["stowed"] * 3 and sorted(ep["stowed_parts"]) == [0, 1, 2]
        assert ep["bricks_stowed"] == 3 and payload["cycles_per_stowed"] == pytest.approx(1.0)
        assert (seen["lifts_classified"], seen["lifts_empty"]) == (3, 0), seen
        assert seen["wrong_wrist_class"] == wrong and seen["wrong_overhead_class"] == wrong


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
        if self._strict and name in ("destination_bins", "desk_box_poses"):
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


def _teacher_scene(monkeypatch):
    env, rig = _env_and_rig()
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


@pytest.mark.parametrize("scene", [_teacher_scene, _rake_scene])
def test_no_decision_reads_the_truth(scene, monkeypatch):
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
    parts through the serving rules' stop flag; the rake scene runs a rake takeover, whose rig
    must read neither ``snapshot`` nor the contacts either.
    """
    env, rig, policy, budget, switches = scene(monkeypatch)
    served = _served(env)
    first = _rollout(env, rig, policy, budget=budget, perceived=True, **switches)
    env.close()

    env, rig, policy, budget, switches = scene(monkeypatch)
    blind = _served(env)
    metric_calls, gen = [], torch.Generator().manual_seed(7)

    def garbage(env, per_part):
        metric_calls.append(1)
        return _garbage_reads(env, per_part, gen)

    monkeypatch.setattr(loop, "truth_reads", garbage)
    for name in TRUTH_DECISION_READS:
        monkeypatch.setattr(loop, name, _raises(name))
    rig.snapshot = _raises("rig.snapshot")
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
    if scene is _teacher_scene:
        assert "placed" in ep["cycle_reasons"] and first["perception"]["lifts_classified"] > 0
    else:
        assert ep["rakes"] >= 1 and "budget" in ep["cycle_reasons"], ep
