"""The MS7f base branch (MS7f design sections 4, 6.1, 6.2, 8.1 and 12.2): the six categories,
the category map, the grader's category boxes, the payload's throughput and held-out rows, the
S0 cycle intervals and the switch plumbing. With every MS7f switch off, every draw, map and
payload field equals MS7e's: the pinned values were computed from 29a36ef before the change."""

import argparse
import importlib.util
import random
from collections import Counter
from pathlib import Path

import pytest
import torch

from maker_arm_lab.core import batch_loop, checker, desk_boxes, loop, oracle, parts, perception
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.core.stowed import per_brick_stowed
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("round", "ldraw", "flat")
PIN_HEAP = {"heap_k": 12, "parts_seed": 20260915, "parts_split": "all", "layout_seed": 7}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wave_env(num_envs=2, heap=None, **kw):
    """The MS7e wave scene (B2, eight boxes drawn, the tray leaned 30 degrees, the deep floors,
    every MS7e family), without cameras."""
    if heap is None:
        heap = batch_loop.heap_parts(argparse.Namespace(**PIN_HEAP, families=FAMILIES))
    draw = sample_run_draw(random.Random(7))
    draw["slot_colors"] = (0, 1, 2, 3)
    return MjLegoSortEnv(
        num_envs,
        draw,
        parts=heap,
        cameras=False,
        desk_boxes=8,
        tray=True,
        tray_lean_deg=30.0,
        gripper_mode="continuous",
        context="camera_box_target",
        box_draw=8,
        deep_box_floor=True,
        release_lean=True,
        round_physics=True,
        **kw,
    )


def _draw_pins(env) -> list:
    """Two resets of both envs: each episode's box count, class map and part boxes, and the
    env generator's next value, which every draw of the reset (boxes, map, colors, the drop)
    moves. Pure Python draws, so they pin on any host."""
    rows = []
    for seed in (7, 8):
        env.reset(seed=seed)
        rows.append([env._rng.random()])
        for i in range(env.num_envs):
            dest = env.destination_bins[i].tolist()
            rows[-1] += [int(env.box_count[i]), env.class_to_box[i].tolist(), dest]
    return rows


# The pins of 29a36ef (MS7e, PR #150), computed with the functions above before the change.
MS7E_DRAWS = [
    [
        0.5129812108526537,
        6,
        [1, 0, 1, 2, 0, 4, 3, 5],
        [1, 2, 4, 0, 1, 1, 3, 1, 0, 5, 1, 3],
        4,
        [3, 0, 3, 1, 1, 2, 0, 2],
        [3, 1, 2, 0, 3, 3, 0, 3, 1, 2, 3, 0],
    ],
    [
        0.6556115457923952,
        5,
        [0, 2, 2, 0, 1, 1, 4, 3],
        [0, 0, 1, 2, 2, 2, 4, 2, 1, 3, 2, 4],
        7,
        [0, 6, 4, 0, 3, 1, 5, 2],
        [0, 0, 1, 6, 4, 4, 5, 4, 3, 2, 4, 5],
    ],
]


def test_every_catalog_kind_has_one_category():
    everything = parts.catalog(tuple(parts.FAMILY_KEYS))
    categories = Counter(parts.part_category(p) for p in everything)
    assert categories == {"brick": 23, "special": 14, "flat": 10, "round": 4}
    assert set(parts.FAMILY_CATEGORY) >= set(parts.FAMILIES)
    assert set(parts.FAMILY_CATEGORY.values()) == set(parts.PART_CATEGORIES)
    tall = parts.make_part("brick", 1, 2, 5)
    assert parts.part_category(tall) == "brick" and parts.part_class(tall) == "tall"


def test_the_active_categories_are_the_catalogs_in_table_order():
    assert parts.active_categories() == ("brick", "special")
    assert parts.active_categories(("flat", "round")) == ("brick", "special", "flat", "round")
    assert parts.active_categories(("ldraw",)) == ("brick", "special")
    for keys in (("round",), ("round", "ldraw", "flat")):
        present = {parts.part_category(p) for p in parts.catalog(keys)}
        assert set(parts.active_categories(keys)) == present


def test_with_categories_off_the_draws_and_the_map_are_ms7es():
    env = _wave_env(class_names=parts.PART_CLASSES)
    assert _draw_pins(env) == MS7E_DRAWS
    assert env.part_box_sets(0) is None and getattr(env, "destination_mask", None) is None
    env.close()


# -- the category map (design section 6.2) --------------------------------------------------

CATS = ("brick", "special", "flat", "round")
SIZES8 = [(0.08, 0.08, 0.04)] * 8


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 9])
def test_the_category_map_reproduces_and_maps_every_category(n):
    volumes = [4e-5, 1e-5, 3e-5, 2e-5]
    sizes = [(0.1, 0.1, 0.05)] * n
    got = desk_boxes.category_map(random.Random(3), sizes, CATS, volumes)
    assert got == desk_boxes.category_map(random.Random(3), sizes, CATS, volumes)
    assert len(got) == 4 and all(own for own in got)
    used = [b for own in got for b in own]
    assert set(used) == set(range(n)) if n >= 4 else set(used) <= set(range(n))
    if n >= 4:
        assert sorted(used) == list(range(n))  # every box once
    else:
        assert all(len(own) == 1 for own in got)  # R11's fold: several categories to a box


def test_the_second_boxes_go_in_volume_order_largest_first():
    volumes = [2e-5, 1e-5, 4e-5, 3e-5]  # flat, round, brick, special by volume
    for n, counts in ((5, [1, 1, 2, 1]), (6, [1, 1, 2, 2]), (7, [2, 1, 2, 2]), (8, [2, 2, 2, 2])):
        got = desk_boxes.category_map(random.Random(n), [(0.1, 0.1, 0.05)] * n, CATS, volumes)
        assert [len(own) for own in got] == counts, (n, got)
    # a tie keeps the category order; round again past one second box each
    got = desk_boxes.category_map(random.Random(1), [(0.1, 0.1, 0.05)] * 7, CATS[:2], [1, 1])
    assert [len(own) for own in got] == [4, 3]


def test_the_category_map_fits_the_boxes_to_the_expected_volume():
    """R11's floor: a big category's boxes are the big ones when a draw fits."""
    sizes = [(0.07, 0.07, 0.03)] * 4 + [(0.1, 0.1, 0.05)]
    volumes = [2.0e-4, 1e-5, 1e-5, 1e-5]  # the brick load fits only a 0.1 m box and a second
    got = desk_boxes.category_map(random.Random(5), sizes, CATS, volumes)
    assert 4 in got[0] and len(got[0]) == 2
    fold = desk_boxes.category_map(random.Random(5), sizes[:3], CATS, volumes)
    assert [len(own) for own in fold] == [1, 1, 1, 1]


def test_the_lighter_box_takes_the_least_volume_and_a_tie_the_lower_index():
    assert desk_boxes.lighter_box((5, 2), [0.0] * 8) == 2
    assert desk_boxes.lighter_box((2, 5), [0, 0, 3e-6, 0, 0, 1e-6, 0, 0]) == 5
    assert desk_boxes.lighter_box((3,), [9.0] * 4) == 3


# -- the grader and the teacher -------------------------------------------------------------


def test_a_part_at_rest_in_any_box_of_its_category_grades_correct():
    pos = torch.zeros(1, 2, 3)
    boxes = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    half = torch.tensor([0.05, 0.05, 0.05])
    pos[0, 1, 0] = 0.5
    allowed = torch.tensor([[[True, False, True], [True, False, True]]])
    ok = checker.parts_at_rest_in_box_sets(pos, torch.zeros(1, 2, 3), boxes, half, allowed)
    assert ok.tolist() == [[True, False]]
    moving = torch.full((1, 2, 3), 0.1)
    assert not checker.parts_at_rest_in_box_sets(pos, moving, boxes, half, allowed).any()


def _categories_env(heap):
    env = _wave_env(1, heap, class_names=parts.active_categories(), categories=True)
    env.reset(seed=3)
    return env


def _rest_in(env, b: int, box: int) -> None:
    """Part ``b`` of env 0 at rest on box ``box``'s floor."""
    d, h = env.datas[0], env.handles
    x, y, _ = env.desk_box_poses[0, box]
    adr = h.brick_qpos[b]
    d.qpos[adr : adr + 7] = (x, y, desk_boxes.DESK_BOX_WALL + env.parts[b].size[2] / 2, 1, 0, 0, 0)
    d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
    import mujoco

    mujoco.mj_forward(env.models[0], d)
    env._refresh([0], images=False)


def test_the_env_grades_either_box_of_a_category_and_no_other():
    heap = [parts.make_part("brick", 2, 4)] * 3 + [parts.make_part("slope", 2, 3)]
    env = _categories_env(heap)
    brick, special = env.class_boxes[0]
    assert env.class_names == ("brick", "special") and env.part_classes[3] == "special"
    assert len(brick) + len(special) == int(env.box_count[0]) and len(brick) >= 2
    assert env.class_to_box[0].tolist() == [brick[0], special[0]]
    assert env.destination_bins[0].tolist() == [brick[0]] * 3 + [special[0]]
    mask = env.destination_mask[0]
    assert [set(torch.nonzero(mask[b]).flatten().tolist()) for b in (0, 3)] == [
        set(brick),
        set(special),
    ]
    assert env.part_box_sets(0) == (brick,) * 3 + (special,)
    _rest_in(env, 0, brick[1])
    _rest_in(env, 3, brick[0])
    stowed = per_brick_stowed(env)[0].tolist()
    env.close()
    assert stowed[0] and not stowed[3]


def test_the_teacher_sends_a_part_to_its_categorys_lighter_box_by_its_own_record():
    heap = [parts.make_part("brick", 2, 4)] * 2
    sets = ((4, 1), (4, 1))
    load = [0.0] * 8
    boxes = [(0.1 * m, 0.2, 0.0) for m in range(8)]
    orc = HeapOracle(
        oracle.layout.DEFAULT_BIN_LAYOUT,
        heap,
        torch.tensor([4, 4]),
        oracle.nominal_draw(),
        boxes=boxes,
        box_sets=sets,
        box_load=load,
    )
    assert orc.release_box(0) == 1  # a tie goes to the lower index
    load[1] += 1e-6
    assert orc.release_box(0) == 4 and orc._release_target(1)[0] == pytest.approx(0.4)
    plain = HeapOracle(
        oracle.layout.DEFAULT_BIN_LAYOUT, heap, torch.tensor([4, 4]), oracle.nominal_draw(),
        boxes=boxes,
    )  # fmt: skip
    assert plain.release_box(0) == 4 and plain.box_sets is None


def test_the_teacher_spreads_a_category_over_its_boxes_and_stows_them_correct():
    """The oracle driver on a categories heap: every stow lands in a box of its category, and
    the bricks fill more than one brick box, the lighter first."""
    heap = [parts.make_part("brick", 2, n) for n in (2, 3, 4, 2, 3, 4)]
    env = _categories_env(heap)
    loads = [[0.0] * env.desk_boxes]
    rng = random.Random(1)

    def factory(i, bl, draw):
        return HeapOracle(
            bl, env.parts, env.destination_bins[i], draw, rng=rng, cycle_budget=8,
            boxes=env.desk_box_poses[i], box_sizes=env.desk_box_sizes[i], tray=True,
            tray_lean_deg=30.0, box_sets=env.part_box_sets(i), box_load=loads[i],
        )  # fmt: skip

    rig = OracleRig(
        env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory,
        release_lean=True, carry_via=True,
    )  # fmt: skip
    payload = loop.rollout(
        env, rig, None, run_seed=3, layout_seed=7, permutation=(0, 1, 2, 3), cycle_budget=8,
        max_wall_minutes=10.0,
    )  # fmt: skip
    env.close()
    ep = payload["episodes"][0]
    brick = ep["class_boxes"]["brick"]  # the env's own map moved on with its auto-reset
    assert ep["bricks_stowed"] >= 4, ep["cycle_reasons"]
    used = [m for m, row in enumerate(ep["stowed_by_box"]) if row["stowed"]]
    assert set(used) <= set(brick) and len(used) >= 2, (used, brick)
    assert ep["class_to_box"]["brick"] == brick[0] and ep["wrong_box_parts"] == 0
    assert ep["stowed_by_category"] == ep["stowed_by_class"]
    assert payload["stowed_by_category"]["brick"]["stowed"] == ep["bricks_stowed"]
    assert sum(loads[0][m] > 0 for m in brick) >= 2


# -- the class path under categories --------------------------------------------------------


def test_the_categories_switch_turns_the_classes_into_the_active_categories():
    args = argparse.Namespace(families=FAMILIES, box_draw=8, categories=True)
    assert batch_loop.heap_classes(args) == ("brick", "special", "flat", "round")
    args.categories = False
    assert batch_loop.heap_classes(args) == parts.PART_CLASSES


def test_the_stand_in_errs_among_the_active_categories_on_the_same_draws():
    """Knob rows stay paired (design section 16): a classification draws twice whatever the
    class list, so the classes and the categories fire on the same calls."""
    cats = parts.active_categories(FAMILIES)
    a = perception.Perceiver(7, 2, {"class": 20.0}, parts.PART_CLASSES)
    b = perception.Perceiver(7, 2, {"class": 20.0}, cats)
    fired = []
    for true_cls, true_cat in [("wide", "brick"), ("slope", "special"), ("tile", "flat")] * 20:
        fired.append((a.classify(true_cls) != true_cls, b.classify(true_cat) != true_cat))
    assert all(x == y for x, y in fired) and any(x for x, _ in fired)
    assert a.rng.getstate() == b.rng.getstate()
    wrong = {b.classify("round") for _ in range(400)} - {"round"}
    assert wrong == set(cats) - {"round"}
    v = perception.ClassViews(7, 2, {"class": 1000.0, "rho": 0.0}, 1, cats)
    assert {v.view(0, "flat") for _ in range(200)} == set(cats) - {"flat"}


def test_the_env_refuses_categories_without_the_box_draw():
    with pytest.raises(ValueError, match="box_draw"):
        MjLegoSortEnv(
            1, sample_run_draw(random.Random(5)), parts=[parts.make_part("brick", 2, 4)],
            cameras=False, desk_boxes=4, categories=True,
        )  # fmt: skip


def test_the_recorder_writes_the_category_boxes_and_wide_part_ids(tmp_path):
    import h5py

    from maker_arm_lab.mj.recorder import ShardRecorder

    rec = ShardRecorder(tmp_path / "s.hdf5", 1, export_all=True, cameras=False, wide_ids=True)
    state = {
        "robot_joint_position": torch.zeros(8).numpy(),
        "robot_joint_velocity": torch.zeros(8).numpy(),
        "robot_root_pose": torch.zeros(7).numpy(),
        "robot_root_velocity": torch.zeros(6).numpy(),
        "brick_root_pose": torch.zeros(1, 7).numpy(),
        "brick_root_velocity": torch.zeros(1, 6).numpy(),
        "class_to_box": torch.tensor([0, 1]).numpy(),
        "class_box_mask": torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool).numpy(),
    }
    rec.begin(0, state)
    rec.pre_step(0, torch.zeros(8).numpy(), torch.zeros(7).numpy(), 1, 140, None, None)
    rec.post_step(0, torch.zeros(8).numpy(), False)
    rec.end(0, True)
    rec.close()
    with h5py.File(tmp_path / "s.hdf5") as f:
        demo = f["data/demo_0"]
        assert demo["obs/cycle_brick"][0] == 140 and demo["obs/cycle_brick"].dtype.itemsize == 2
        assert demo["initial_state/class_box_mask"][()].tolist() == [
            [True, False, True],
            [False, True, False],
        ]


# -- the switch in the clients --------------------------------------------------------------

HEAP_EVAL = ["--heap_k", "3", "--oracle_driver", "--desk_boxes", "4"]
HEAP_BATCH = ["--out_dir", "/tmp/x", "--heap_k", "3", "--desk_boxes", "4"]


@pytest.mark.parametrize("name, argv, token", [
    ("mj_eval_heap", HEAP_EVAL, "EVAL_FAIL "),
    ("mj_generate_batch", HEAP_BATCH, "BATCH_FAIL usage: "),
])  # fmt: skip
def test_the_clients_refuse_categories_without_the_box_draw(name, argv, token, capsys):
    script = _load(name)
    with pytest.raises(SystemExit) as exit_info:
        script.main(argv + ["--categories"])
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith(token) and "--box_draw" in out[0], out
    args = script.parse_args(argv + ["--categories", "--box_draw", "4"])
    assert batch_loop.ms7f_switches(args) == {**batch_loop.MS7F_OFF, "categories": True}
    assert batch_loop.ms7f_switches(script.parse_args(argv)) == batch_loop.MS7F_OFF
