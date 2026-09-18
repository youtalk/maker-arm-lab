"""The bottom-open pick probe: its token contract and its descend."""

import ast
import importlib.util
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import mujoco
import pytest

from maker_arm_lab.core import grasp, layout, oracle, parts
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.mj.env import MjLegoSortEnv

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mj_probe_bottom_pick.py"

# The hand-placed layout of tests/test_mj_heap_oracle.py: three spread parts in the pickable
# band, clear of each other, so the descend under test is an ordinary one.
PARTS = [
    parts.make_part("brick", 2, 4),
    parts.make_part("brick", 2, 2),
    parts.make_part("brick", 1, 4),
]
POSES = [(0.195, -0.045, 0.006, 0.3), (0.195, 0.0, 0.006, -0.8), (0.195, 0.045, 0.006, 1.2)]


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _place(env, poses) -> None:
    m, d, h = env.models[0], env.datas[0], env.handles
    for b, (x, y, z, yaw) in enumerate(poses):
        adr = h.brick_qpos[b]
        d.qpos[adr : adr + 3] = (x, y, z)
        d.qpos[adr + 3 : adr + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=120)
    env._refresh([0], images=False)


def test_the_script_prints_one_token_and_exits_through_os_exit():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    assert '"PROBE_OK "' in source and '"PROBE_FAIL "' in source
    exits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_exit"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
    ]
    assert exits, "the script must leave through os._exit"
    assert "except SystemExit" in source  # an argparse error is PROBE_FAIL, not a hang


def test_an_argparse_error_is_a_fail_token():
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--out_dir", "/tmp/x", "--mode", "nonsense"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 2
    assert done.stdout.strip().splitlines()[-1].startswith("PROBE_FAIL ")


def test_the_descend_holds_the_narrow_opening_and_opens_at_the_pick_depth():
    module = _load(SCRIPT)
    draw = layout.sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, parts=PARTS, cameras=False)
    env.reset(seed=11)
    rng = random.Random(1)

    def factory(i, bl, episode_draw):
        return HeapOracle(bl, env.parts, int(env.destination_bins[i]), episode_draw, rng=rng)

    rig = module.BottomPickRig(
        env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory
    )
    _place(env, POSES)
    # One cycle is enough: the first DESCEND is the leg under test.
    steps = 0
    while steps < 3000 and not any(e["phase"] == "CLOSE" for e in rig.phase_log):
        out = rig.step()
        steps += 1
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    env.close()

    descending = [e for e in rig.jaw_log if e["descending"]]
    at_depth = [e for e in rig.jaw_log if not e["descending"]]
    assert descending and at_depth, rig.jaw_log[:5]

    target = descending[-1]["target"]
    # The opening is the part width, two tip thicknesses and 4 mm, halved: well inside the
    # shipped fully-open landing spot, and never narrower than the part itself.
    assert target < grasp.OPEN_HALF_GAP / 2.0
    assert min(e["gap"] for e in descending) < grasp.OPEN_HALF_GAP - 0.02
    # The jaws reach the held opening before the tips reach the pick depth, and the bang-bang
    # band keeps them from closing onto the part on the way down.
    held = [e["gap"] for e in descending[len(descending) // 2 :]]
    assert min(held) >= target - 0.001, min(held)
    assert max(held) <= target + 3.0 * module.JAW_BAND_M, max(held)
    # At the pick depth the shipped fully-open command stands, and the descend holds there
    # until the jaws have followed it out to the full opening.
    assert all(e["command"] >= 0.0 for e in at_depth)
    assert max(e["gap"] for e in at_depth) >= grasp.OPEN_HALF_GAP - 2.0 * module.JAW_BAND_M
    assert len(at_depth) <= module.JAW_OPEN_STEPS + 1, len(at_depth)


def test_the_reach_grid_is_the_designs_radii_and_azimuths():
    module = _load(SCRIPT)
    grid = module.reach_grid()
    radii = sorted({r for r, _ in grid})
    azimuths = sorted({a for _, a in grid})
    assert radii == [round(0.10 + 0.02 * i, 2) for i in range(13)]
    assert radii[0] == 0.10 and radii[-1] == 0.34
    assert azimuths == [-40, -20, 0, 20, 40]
    assert len(grid) == len(radii) * len(azimuths)
    # The control is a point of the grid, not a separate run.
    assert module.REACH_CONTROL_RADIUS_M in radii
    assert all((module.REACH_CONTROL_RADIUS_M, a) in grid for a in azimuths)


def test_the_parser_honors_the_grid_flags_and_defaults_to_the_designs_grid():
    module = _load(SCRIPT)
    base = ["--mode", "reach_band", "--out_dir", "/tmp/x"]
    default = module.parse_args(base)
    assert default.radii_m == [round(0.10 + 0.02 * i, 2) for i in range(13)]
    assert default.azimuths_deg == [-40, -20, 0, 20, 40]
    assert module.reach_grid(default.radii_m, default.azimuths_deg) == module.reach_grid()

    fine = module.parse_args(base + ["--radii_mm", "40,50,60,200", "--azimuths_deg", "0,40"])
    assert fine.radii_m == [0.04, 0.05, 0.06, 0.20]
    assert fine.azimuths_deg == [0, 40]
    grid = module.reach_grid(fine.radii_m, fine.azimuths_deg)
    assert len(grid) == 8
    assert grid[0] == (0.04, 0) and grid[-1] == (0.20, 40)


def test_the_heap_seeds_are_defaults_and_a_passed_seed_survives():
    module = _load(SCRIPT)
    base = ["--mode", "heap", "--out_dir", "/tmp/x"]

    default = module.parse_args(base)
    assert (default.run_seed, default.layout_seed) == (20261000, 20261010)
    assert (default.run_seed, default.layout_seed) == (
        module.HEAP_RUN_SEED,
        module.HEAP_LAYOUT_SEED,
    )
    assert default.heap_k == module.HEAP_K and default.parts_split == "heldout"
    # Without this the reset RNG runs differently and the heap is a different heap.
    assert default.episode_draws is True

    # A flag the caller passes is honored, or three seed pairs are three copies of heap 0.
    passed = module.parse_args(base + ["--run_seed", "20261001", "--layout_seed", "20261011"])
    assert (passed.run_seed, passed.layout_seed) == (20261001, 20261011)
    assert passed.heap_k == module.HEAP_K  # untouched flags keep the read's default

    equals_form = module.parse_args(base + ["--run_seed=20261002", "--heap_k=12"])
    assert equals_form.run_seed == 20261002 and equals_form.heap_k == 12
    assert equals_form.layout_seed == module.HEAP_LAYOUT_SEED


def test_the_heap_mode_resolves_to_the_reads_flags():
    module = _load(SCRIPT)
    args = module.parse_args(["--mode", "heap", "--out_dir", "/tmp/x"])
    # The whole read, not just the heap's identity: a default left at argparse's own value is a
    # configuration difference nobody sees, and the jitter band alone moves heap 0 from 15 to 18.
    assert args.nearest_first_pick_order is True
    assert args.frame_jitter is True
    assert (args.hover_jitter_min, args.hover_jitter_max) == module.HEAP_HOVER_JITTER
    assert (args.hover_jitter_min, args.hover_jitter_max) == (0.035, 0.080)
    assert args.episode_draws is True
    assert (args.run_seed, args.layout_seed) == (20261000, 20261010)
    assert args.heap_k == module.HEAP_K and args.parts_split == "heldout"


def test_a_caller_passed_read_flag_still_wins():
    module = _load(SCRIPT)
    base = ["--mode", "heap", "--out_dir", "/tmp/x"]
    passed = module.parse_args(base + ["--hover_jitter_min", "0.0", "--hover_jitter_max", "0.0"])
    assert (passed.hover_jitter_min, passed.hover_jitter_max) == (0.0, 0.0)
    assert passed.nearest_first_pick_order is True  # untouched flags keep the read's value
    equals_form = module.parse_args(base + ["--hover_jitter_max=0.05"])
    assert equals_form.hover_jitter_max == 0.05
    assert equals_form.hover_jitter_min == module.HEAP_HOVER_JITTER[0]


def test_no_read_flags_skips_the_driving_flags_and_keeps_the_pins():
    module = _load(SCRIPT)
    # The merged report gives this as the WHOLE command for the archived no-band runs, so it
    # has to start: the five pins say which heap this is and survive the flag.
    args = module.parse_args(["--mode", "heap", "--out_dir", "/tmp/x", "--no_read_flags"])
    assert args.nearest_first_pick_order is False
    assert args.frame_jitter is False
    assert (args.hover_jitter_min, args.hover_jitter_max) == (0.0, 0.0)
    assert args.heap_k == module.HEAP_K and args.parts_split == "heldout"
    assert (args.run_seed, args.layout_seed) == (module.HEAP_RUN_SEED, module.HEAP_LAYOUT_SEED)
    assert args.episode_draws is module.HEAP_EPISODE_DRAWS is True
    assert module.resolved_config(args)["read_flags"] is False
    # The two groups are disjoint and the pins are not in the skippable one.
    pinned = {flag for flag, _ in module.HEAP_PINS}
    assert pinned.isdisjoint(flag for flag, _ in module.HEAP_READ_FLAGS)
    assert "--episode_draws" in pinned and "--hover_jitter_min" not in pinned


def test_a_caller_passed_flag_still_wins_under_no_read_flags():
    module = _load(SCRIPT)
    args = module.parse_args(
        ["--mode", "heap", "--out_dir", "/tmp/x", "--no_read_flags", "--run_seed=20261002"]
    )
    assert args.run_seed == 20261002
    assert args.layout_seed == module.HEAP_LAYOUT_SEED


def test_the_heap_mode_refuses_a_non_positive_k():
    module = _load(SCRIPT)
    with pytest.raises(SystemExit):
        module.parse_args(["--mode", "heap", "--out_dir", "/tmp/x", "--heap_k", "0"])


def test_the_three_verified_commands_resolve_as_measured():
    module = _load(SCRIPT)
    out = ["--out_dir", "/tmp/x"]

    # matched_0, 15 stowed: the read's flags. The clearance model that run reached through
    # --matched_clearance is the SHIPPED model since MS6 rung 1, so the flag is gone and the
    # archived heap rows of docs/ms6-lever-verification.md no longer reproduce from this
    # script. What still reproduces is WHICH HEAP each row drove and how.
    bare = module.parse_args(["--mode", "heap", *out])
    assert (bare.hover_jitter_min, bare.hover_jitter_max) == module.HEAP_HOVER_JITTER
    assert bare.heap_k == module.HEAP_K

    # minimal_matched_0 / iso_nojitter_0, 18 stowed: same heap, no band.
    no_band = module.parse_args(["--mode", "heap", *out, "--no_read_flags"])
    assert (no_band.hover_jitter_min, no_band.hover_jitter_max) == (0.0, 0.0)
    assert no_band.heap_k == module.HEAP_K

    # Both are the same heap; only the driving differs.
    identity = ("heap_k", "parts_split", "run_seed", "layout_seed", "episode_draws")
    for name in identity:
        assert getattr(bare, name) == getattr(no_band, name), name


def test_the_resolved_configuration_is_reported():
    module = _load(SCRIPT)
    args = module.parse_args(["--mode", "heap", "--out_dir", "/tmp/x"])
    config = module.resolved_config(args)
    assert config["hover_jitter"] == [0.035, 0.080]
    assert config["nearest_first_pick_order"] is True and config["frame_jitter"] is True
    assert config["episode_draws"] is True and config["read_flags"] is True
    assert (config["run_seed"], config["layout_seed"]) == (20261000, 20261010)
    assert config["heap_k"] == 50 and config["cycle_budget"] == args.cycle_budget


def test_the_config_is_stamped_into_the_logs_end_event(tmp_path):
    module = _load(SCRIPT)
    log = tmp_path / "events.jsonl"
    log.write_text(
        json.dumps({"kind": "scan", "step": 1})
        + "\n"
        + json.dumps({"kind": "end", "step": 9, "stowed": 3, "parts": 50, "extra_stats": {}})
        + "\n"
    )
    module.stamp_config(str(log), {"run_seed": 7, "hover_jitter": [0.035, 0.08]})
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(events) == 2 and events[0]["kind"] == "scan"
    end = events[-1]
    assert end["kind"] == "end" and end["stowed"] == 3  # the log's own fields survive
    assert end["config"] == {"run_seed": 7, "hover_jitter": [0.035, 0.08]}


def test_a_point_outside_the_solvers_reach_reads_no_solution_not_lost():
    module = _load(SCRIPT)
    unplanned = [
        {"phase": "HOVER", "outcome": "converged"},
        {"phase": "DESCEND", "outcome": "unplanned"},
    ]
    driven = [
        {"phase": "HOVER", "outcome": "converged"},
        {"phase": "DESCEND", "outcome": "stalled"},
        {"phase": "LIFT", "outcome": "converged"},
    ]
    # Unreachable is not a failed pick: the two causes stay apart.
    assert module.reach_outcome(unplanned, False) == "no_solution"
    assert module.reach_outcome(driven, False) == "lost"
    assert module.reach_outcome(driven, True) == "stowed"
    # A leg the solver missed before a retry that stowed the brick is not no_solution.
    assert module.reach_outcome(unplanned, True) == "stowed"
    # An unplanned CARRY leg is not evidence about reaching the brick.
    assert module.reach_outcome([{"phase": "EXTEND", "outcome": "unplanned"}], False) == "lost"
