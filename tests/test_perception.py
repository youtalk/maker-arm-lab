"""The MS7c perceived state (core.perception): the knobs, the structural rules, the seeding,
the position memory and the adapters that feed the heap selector."""

import math
import random

import pytest
import torch

from maker_arm_lab.core import heap_oracle, layout, oracle, parts, perception
from maker_arm_lab.core.perception import KNOBS, Detection, Perceiver, PositionMemory, Truth
from maker_arm_lab.core.quat import quat_from_rpy_deg

BRICK = parts.make_part("brick", 2, 4)  # 31.8 x 15.8 x 9.6 mm
SLOPE = parts.make_part("slope", 2, 3)
IDENTITY = (1.0, 0.0, 0.0, 0.0)
Z = BRICK.size[2] / 2
BOXES = ((0.08, 0.20, 0.3), (0.10, -0.20, -0.5))
SPREAD = [[0.15 + 0.04 * (i % 4), -0.08 + 0.05 * (i // 4), Z] for i in range(8)]
"""Eight parts 40 mm or more apart, none covering another, all far from a tool at (0, 0)."""
ZERO = perception.parse_scales("all=0")
NOMINAL = (
    "pos=1,heading=1,size=1,box=1,miss=1,class=1,"
    "tool=1,wrist=1,wrist_bias=1,wrist_miss=1,rho=1,torque=1,torque_k=1"
)


def _truth(centres, part_list=None, quats=None, tool_xy=(0.0, 0.0)):
    part_list = part_list or [BRICK] * len(centres)
    return Truth(
        pos=torch.tensor(centres),
        quat=torch.tensor(quats or [IDENTITY] * len(centres)),
        sizes=torch.tensor([p.size for p in part_list]),
        classes=tuple(parts.part_class(p) for p in part_list),
        tool_xy=tool_xy,
    )


def _perceiver(seed=1, env_index=0, scales=None, boxes=BOXES):
    p = Perceiver(seed, env_index, scales)
    p.reset_episode(boxes)
    return p


def _assert_truth(state, truth):
    for d in state.detections:
        b = d.sim_index
        assert d.centre == tuple(truth.pos[b].tolist())
        assert d.quat == tuple(truth.quat[b].tolist())
        assert d.size == tuple(truth.sizes[b].tolist())
        assert d.cls == truth.classes[b]


def test_at_scale_zero_a_stacked_part_hides_the_one_under_it():
    # Part 1 lies on part 0, its footprint over part 0's centre; part 3 is a slope turned on
    # its side at 30 degrees, alone.
    centres = [[0.20, 0.0, Z], [0.205, 0.003, 3 * Z], [0.20, -0.10, Z], [0.28, 0.08, Z]]
    quats = [IDENTITY, IDENTITY, IDENTITY, quat_from_rpy_deg((90.0, 0.0, 30.0))]
    truth = _truth(centres, [BRICK, BRICK, BRICK, SLOPE], quats)
    state = _perceiver(scales=ZERO).perceive(truth)
    assert [d.sim_index for d in state.detections] == [1, 2, 3]
    _assert_truth(state, truth)
    assert state.box_poses == BOXES


def test_at_scale_zero_the_tool_hides_a_part_within_40_mm():
    tool = (0.30, 0.135)
    centres = [[0.30, 0.10, Z], [0.30, 0.185, Z], [0.20, -0.10, Z]]  # 35 mm, 50 mm, far
    truth = _truth(centres, tool_xy=tool)
    state = _perceiver(scales=ZERO).perceive(truth)
    assert [d.sim_index for d in state.detections] == [1, 2]
    _assert_truth(state, truth)


def test_a_seed_repeats_and_another_env_index_differs():
    truth = _truth(SPREAD)

    def run(env_index):
        p = _perceiver(7, env_index)
        return [p.perceive(truth), p.perceive(truth)], p.classify("tall"), p.box_poses

    assert run(0) == run(0)
    assert run(0) != run(1)


def test_the_draws_are_paired_across_scales():
    truth = _truth(SPREAD)

    def reads(scales):
        p = _perceiver(3, 0, scales)
        return [p.perceive(truth) for _ in range(4)]

    base, double, missing = reads({}), reads({"pos": 2.0}), reads({"miss": 10.0})
    moved = 0
    for one, two in zip(base, double):
        assert [d.sim_index for d in one.detections] == [d.sim_index for d in two.detections]
        for a, b in zip(one.detections, two.detections):
            assert (a.quat, a.size, a.cls) == (b.quat, b.size, b.cls)
            true = truth.pos[a.sim_index].tolist()
            for k in range(3):
                assert b.centre[k] - true[k] == pytest.approx(
                    2 * (a.centre[k] - true[k]), abs=1e-12
                )
            moved += a.centre != tuple(true)
    assert moved > 0
    # A part the miss knob removes still spends its draws: the parts seen in both reads match.
    for one, other in zip(base, missing):
        seen = {d.sim_index: d for d in other.detections}
        assert len(seen) < len(one.detections)
        assert all(seen[d.sim_index] == d for d in one.detections if d.sim_index in seen)


MS7C_DRAWS = {
    0: [
        [-0.000108, 0.001035, -0.000483],
        [0.078318, 0.197641, 0.347844],
        [0.102587, -0.197989, -0.499882],
        [0, 0.149108, -0.079932, 0.002497, 0.031668, 0.015448, 0.008677, -0.001772, "wide"],
        [1, 0.192954, -0.081804, 0.003163, 0.023677, 0.017372, 0.012362, 0.020681, "slope"],
        [0, 0.149018, -0.079731, 0.002976, 0.029912, 0.017807, 0.008028, -0.031607, "wide"],
        [1, 0.189857, -0.076529, 0.0037, 0.024065, 0.015494, 0.011476, 0.021436, "slope"],
        ["tall", 0.604433],
    ],
    3: [
        [0.001226, 0.000747, 0.000781],
        [0.08043, 0.204194, 0.256346],
        [0.101959, -0.194316, -0.435433],
        [0, 0.151479, -0.080274, 0.00676, 0.029735, 0.015819, 0.004057, -0.022749, "wide"],
        [1, 0.18972, -0.079519, 0.004772, 0.023781, 0.017249, 0.008482, -0.05418, "slope"],
        [0, 0.151229, -0.078847, 0.010302, 0.031777, 0.014581, 0.009471, 0.00841, "wide"],
        [1, 0.191772, -0.079768, 0.003511, 0.021608, 0.016305, 0.008996, 0.036269, "slope"],
        ["tall", 0.848383],
    ],
}
"""A ``Perceiver`` sequence recorded at MS7c's main (cc5f009), run seed 20261700, envs 0 and
3, rounded to 1e-6: the episode bias, the two box poses, two ``perceive`` calls (each
detection's index, centre, size, quaternion z and class), a ``classify`` and the generator's
next draw."""


def test_the_draws_with_every_ms7d_switch_off_equal_the_ms7c_sequence():
    """The MS7d design's pairing rule: the new knobs draw nothing in the ``Perceiver``, so its
    stream and the checked reads pair with the MS7c candidates."""
    truth = _truth([[0.15, -0.08, Z], [0.19, -0.08, Z]], [BRICK, SLOPE])

    def r(values):
        return [round(v, 6) for v in values]

    for env_index, expected in MS7C_DRAWS.items():
        p = _perceiver(20261700, env_index)
        got = [r(p.bias), *(r(b) for b in p.box_poses)]
        for _ in range(2):
            for d in p.perceive(truth).detections:
                got.append([d.sim_index, *r(d.centre), *r(d.size), round(d.quat[3], 6), d.cls])
        got.append([p.classify("tall"), round(p.rng.random(), 6)])
        assert got == expected, env_index


TOOL_POINT = (0.25, 0.02, 0.12)  # an FK pick point for the marker looks


def test_at_tool_zero_one_look_reads_the_bias_and_the_detections_keep_only_their_noise():
    """MS7d L3: with ``tool`` at 0 the first look's estimate is the episode bias, and the
    corrected detections are the truth plus the per-detection noise. The marker's generator is
    its own: the ``Perceiver``'s detections are the same with looks between its calls."""
    truth = _truth(SPREAD)
    p = _perceiver(7, scales={"tool": 0.0})
    marker = perception.MarkerBias(p, 7, 0)
    assert marker.estimate == (0.0, 0.0, 0.0) and math.hypot(*p.bias[:2]) > 1e-4
    marker.look(TOOL_POINT)
    assert marker.estimate == pytest.approx(p.bias, abs=1e-12)
    assert marker.error_mm() == pytest.approx(0.0, abs=1e-9)
    raw = p.perceive(truth)
    marker.look(TOOL_POINT)
    assert raw == _perceiver(7, scales={"tool": 0.0}).perceive(truth)
    fixed = marker.correct(raw)
    assert fixed.box_poses == raw.box_poses and len(fixed.detections) == len(raw.detections) > 0
    for f, r in zip(fixed.detections, raw.detections):
        assert (f.quat, f.size, f.cls, f.sim_index) == (r.quat, r.size, r.cls, r.sim_index)
        true = truth.pos[f.sim_index].tolist()
        noise = [c - t - b for c, t, b in zip(r.centre, true, p.bias)]
        assert [c - t for c, t in zip(f.centre, true)] == pytest.approx(noise, abs=1e-12)


def test_the_marker_estimate_error_falls_with_the_looks():
    """MS7d L3 at the nominal ``tool``: the estimate is the mean of the looks, so its error on
    each axis falls as TOOL_SIGMA_M / sqrt(looks), within a loose 25 % over 300 seeds."""
    looks = (1, 4, 16)
    errors = {n: [] for n in looks}
    for seed in range(300):
        p = _perceiver(seed)
        marker = perception.MarkerBias(p, seed, 0)
        for n in range(1, looks[-1] + 1):
            marker.look(TOOL_POINT)
            if n in errors:
                errors[n] += [e - b for e, b in zip(marker.estimate, p.bias)]
    for n in looks:
        rms = math.sqrt(sum(e * e for e in errors[n]) / len(errors[n]))
        assert rms == pytest.approx(perception.TOOL_SIGMA_M / math.sqrt(n), rel=0.25), n


def test_the_tool_knob_scales_paired_marker_draws():
    """Every look draws three values at every scale, so a scale only multiplies them."""
    markers = [
        perception.MarkerBias(_perceiver(3, scales={"tool": s}), 3, 0) for s in (0.0, 1.0, 2.0)
    ]
    for m in markers:
        for _ in range(5):
            m.look(TOOL_POINT)
    bias = markers[0].perceiver.bias
    zero, one, two = ([e - b for e, b in zip(m.estimate, bias)] for m in markers)
    assert zero == pytest.approx([0.0] * 3, abs=1e-12) and max(map(abs, one)) > 0
    assert two == pytest.approx([2 * e for e in one], abs=1e-12)
    assert len({m.rng.random() for m in markers}) == 1


def test_class_views_keeps_every_other_draw_of_the_ms7c_sequence():
    """MS7d L5: the per-part class model draws on the ``Perceiver``'s two class draws, so with
    it on every draw but the class stays paired with MS7c, the generator's next one included."""
    truth = _truth([[0.15, -0.08, Z], [0.19, -0.08, Z]], [BRICK, SLOPE])
    scales = {"class": 1000.0}  # every class is wrong

    def r(values):
        return [round(v, 6) for v in values]

    for env_index, expected in MS7C_DRAWS.items():
        p = _perceiver(20261700, env_index, scales)
        p.views = perception.ClassViews(20261700, env_index, p.scales, 2)
        got = [r(p.bias), *(r(b) for b in p.box_poses)]
        for _ in range(2):
            for d in p.perceive(truth).detections:
                assert d.cls != truth.classes[d.sim_index]
                got.append([d.sim_index, *r(d.centre), *r(d.size), round(d.quat[3], 6)])
        p.classify("tall", 0)
        got.append(round(p.rng.random(), 6))
        paired = [*expected[:3], *(row[:-1] for row in expected[3:7]), expected[7][1]]
        assert got == paired, env_index


def test_a_persistent_part_reads_one_wrong_class_in_every_view():
    """MS7d L5 at a persistent share of 1 (rho 2): the overhead class at every perceive call, a
    wrist view and a third view of a part name one class, wrong on the persistent parts only."""
    truth = _truth(SPREAD)
    true_cls = parts.part_class(BRICK)
    scales = {"class": 10.0, "rho": 2.0, "miss": 0.0}  # p 0.3, share 1, so q is 0
    persistent = 0
    for seed in range(5):
        p = _perceiver(seed, 0, scales)
        p.views = perception.ClassViews(seed, 0, p.scales, len(SPREAD))
        views = [[d.cls for d in p.perceive(truth).detections] for _ in range(3)]
        views.append([p.classify(true_cls, b) for b in range(len(SPREAD))])
        views.append([p.views.view(b, true_cls) for b in range(len(SPREAD))])
        for b in range(len(SPREAD)):
            seen = {v[b] for v in views}
            assert len(seen) == 1, (seed, b, seen)
            persistent += seen != {true_cls}
    assert 0 < persistent < 5 * len(SPREAD)


@pytest.mark.parametrize("rho", [0.0, 1.0, 2.0])
def test_one_class_view_errs_near_p_at_every_persistent_share(rho):
    """MS7d L5: at shares 0, 0.5 and 1 a single view, through the ``Perceiver``'s draws or the
    model's own, errs at ``CLASS_ERROR_PROBABILITY``: 20000 parts, a loose 0.008 bound (the
    standard error is 0.0012)."""
    n, wrong, third = 4000, 0, 0
    for seed in range(5):
        p = _perceiver(seed, 0, {"rho": rho})
        p.views = perception.ClassViews(seed, 0, p.scales, n)
        wrong += sum(p.classify("narrow", b) != "narrow" for b in range(n))
        third += sum(p.views.view(b, "narrow") != "narrow" for b in range(n))
    for errors in (wrong, third):
        assert abs(errors / (5 * n) - perception.CLASS_ERROR_PROBABILITY) < 0.008, (wrong, third)


def test_the_confusion_fires_near_its_rate_and_never_keeps_the_true_class():
    p = _perceiver(5, 0, {"class": 10.0})  # 0.3
    draws = [p.classify("narrow") for _ in range(4000)]
    wrong = [c for c in draws if c != "narrow"]
    assert abs(len(wrong) / len(draws) - 0.3) < 0.03
    assert set(wrong) == {"slope", "tall", "wide"}


def test_the_box_bias_holds_for_the_episode_and_is_redrawn_at_reset():
    p = _perceiver(2, 0)
    first = p.box_poses
    assert len(first) == len(BOXES) and first != BOXES
    for _ in range(3):
        assert p.perceive(_truth(SPREAD)).box_poses == first
    assert p.box_poses == first
    p.reset_episode(BOXES)
    assert p.box_poses != first


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", {}),
        ("heading=2", {"heading": 2.0}),
        ("pos=0,miss=4", {"pos": 0.0, "miss": 4.0}),
        ("all=0", dict.fromkeys(KNOBS, 0.0)),
        ("all=0, heading=1.5", {**dict.fromkeys(KNOBS, 0.0), "heading": 1.5}),
    ],
)
def test_parse_scales(text, expected):
    assert perception.parse_scales(text) == {
        **dict.fromkeys(KNOBS + perception.EXTRA_KNOBS, 1.0),
        **expected,
    }


def test_format_scales_is_canonical_and_round_trips():
    canonical = perception.format_scales(perception.parse_scales(""))
    assert canonical == NOMINAL
    for text in ("miss=4,pos=0", "all=0,heading=0.25", "size=0.1,class=1e-3"):
        scales = perception.parse_scales(text)
        assert perception.parse_scales(perception.format_scales(scales)) == scales


def test_pos_z_scales_only_the_z_axis_and_stays_out_of_all_and_the_default_text():
    assert perception.parse_scales("all=0")["pos_z"] == 1.0
    assert perception.format_scales(perception.parse_scales("pos_z=1")).count("pos_z") == 0
    text = perception.format_scales(perception.parse_scales("pos_z=0"))
    assert text.endswith(",pos_z=0") and perception.parse_scales(text)["pos_z"] == 0.0
    truth = _truth(SPREAD)
    base, flat = _perceiver(7), _perceiver(7, scales={"pos_z": 0.0})
    assert base.bias[:2] == flat.bias[:2] and flat.bias[2] == 0.0 and base.bias[2] != 0.0
    for a, b in zip(base.perceive(truth).detections, flat.perceive(truth).detections):
        assert a.centre[:2] == b.centre[:2] and a.size == b.size and a.cls == b.cls
        assert b.centre[2] == pytest.approx(float(truth.pos[b.sim_index][2]))


@pytest.mark.parametrize("text", ["speed=1", "pos=-1", "pos=abc", "pos", "pos=nan", "all=inf"])
def test_parse_scales_refuses_a_bad_item(text):
    with pytest.raises(ValueError, match="perception knob"):
        perception.parse_scales(text)


def _detection(x, y, index):
    return Detection((x, y, Z), IDENTITY, BRICK.size, "wide", index)


def test_a_record_follows_its_part_within_the_association_radius():
    memory = PositionMemory()
    memory.book((0.20, 0.0))
    memory.book((0.205, 0.0))  # 5 mm away: the same record
    assert memory.records == [[0.20, 0.0, 2]]
    assert memory.assign([_detection(0.21, 0.0, 4)]) == {4: 2}  # 10 mm: taken, and moved
    assert memory.records == [[0.21, 0.0, 2]]
    assert memory.assign([_detection(0.23, 0.0, 4)]) == {}  # 20 mm: not taken
    assert memory.records == [[0.21, 0.0, 2]]


def test_two_records_on_one_detection_the_nearer_wins():
    memory = PositionMemory()
    memory.book((0.20, 0.0))
    memory.book((0.20, 0.02), 3)
    # 12 mm from the first record, 8 mm from the second.
    assert memory.assign([_detection(0.20, 0.012, 1)]) == {1: 3}
    assert memory.records == [[0.20, 0.0, 1], [0.20, 0.012, 3]]


def test_undetected_slots_are_parked_out_of_the_selectors_reach():
    turned = quat_from_rpy_deg((0.0, 0.0, 30.0))
    centres = [[0.20, -0.04, Z], [0.20, 0.0, Z], [0.20, 0.04, Z]]
    truth = _truth(centres, [BRICK, SLOPE, BRICK], [turned, IDENTITY, IDENTITY], (0.20, 0.04))
    state = _perceiver(scales=ZERO).perceive(truth)
    assert [d.sim_index for d in state.detections] == [0, 1]
    scene = state.scene(3, (0.28, 0.0, 0.20))
    selector_parts = state.parts(3)
    assert float(scene.brick_pos[2, 2]) > heap_oracle.CANDIDATE_MAX_Z
    assert float(scene.brick_yaw[0]) == pytest.approx(0.5236, abs=1e-4)  # body x at 30 degrees
    assert selector_parts[0].size == pytest.approx(BRICK.size)
    assert selector_parts[0].closing_axes == parts.BOX_AXES
    assert selector_parts[1].closing_axes == parts.SLOPE_AXES
    selector = heap_oracle.HeapOracle(
        layout.DEFAULT_BIN_LAYOUT, selector_parts, 1, oracle.nominal_draw()
    )
    mask = selector._candidates(scene)
    assert mask.tolist() == [True, True, False]
    assert {g.part for g in selector._ranked(scene, mask)} == {0, 1}


def test_a_wrist_look_is_the_truth_at_zero_scales_and_nothing_when_missed_or_covered():
    """MS7d L4. Part 1 lies on part 0 over its centre: the covering rule, the overhead camera's
    too, covers part 0 only."""
    truth = _truth([[0.20, 0.0, Z], [0.205, 0.003, 3 * Z], [0.20, -0.10, Z]])
    cover = [perception.covered(truth.pos, truth.quat, truth.sizes, b) for b in range(3)]
    assert cover == [True, False, False]
    cam = perception.WristCamera(3, 0, {"wrist": 0.0, "wrist_bias": 0.0, "wrist_miss": 0.0})
    assert cam.bias == (0.0, 0.0)
    assert cam.look((0.2, -0.1), 0.4, False) == (0.2, -0.1, 0.4)
    assert cam.look((0.2, -0.1), 0.4, True) is None
    missing = perception.WristCamera(3, 0, {"wrist_miss": 1000.0})
    assert all(missing.look((0.2, -0.1), 0.4, False) is None for _ in range(20))


def test_the_wrist_draws_are_paired_across_scales_and_outcomes():
    """Two bias draws per episode and four per look, whatever the scales and whether the look
    finds the part, from the camera's own generator: a scale multiplies what was drawn."""

    def run(scales):
        cam = perception.WristCamera(5, 1, {"wrist_miss": 0.0, **scales})
        looks = [cam.look((0.2, 0.0), 0.5, hidden) for hidden in (False, True, False)]
        return cam.bias, looks, cam.rng.random()

    base, double, missed = run({}), run({"wrist": 2.0, "wrist_bias": 2.0}), run({"wrist_miss": 1e3})
    assert base[2] == double[2] == missed[2]
    assert base[1][1] is None and missed[1] == [None] * 3
    assert double[0] == pytest.approx(tuple(2.0 * v for v in base[0]), abs=1e-15)
    truth = (0.2, 0.0, 0.5)
    for one, two in zip(base[1][::2], double[1][::2]):
        assert [b - t for b, t in zip(two, truth)] == pytest.approx(
            [2.0 * (a - t) for a, t in zip(one, truth)], abs=1e-12
        )
    first = random.Random("wrist:5:1").gauss(0.0, 1.0)
    assert base[0][0] == first * perception.WRIST_BIAS_SIGMA_M
