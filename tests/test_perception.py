"""The MS7c perceived state (core.perception): the knobs, the structural rules, the seeding,
the position memory and the adapters that feed the heap selector."""

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
