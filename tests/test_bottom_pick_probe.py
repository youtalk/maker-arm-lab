"""The bottom-open pick probe: its token contract and its descend."""

import ast
import importlib.util
import math
import random
import subprocess
import sys
from pathlib import Path

import mujoco

from maker_arm_lab.core import grasp, layout, oracle, parts
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.mj.env import MjLegoSortEnv

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mj_probe_bottom_pick.py"

# The hand-placed layout of tests/test_mj_heap_oracle.py: three spread parts in the pickable
# band, clear of each other and of the tray walls, so the descend under test is an ordinary one.
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
