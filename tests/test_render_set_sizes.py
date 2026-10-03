"""scripts/render_set_sizes.py: the token contract, and on a hand-made render file, each overhead
row's part size with the perception noise and each wrist row the size of its selection target."""

import importlib.util
import json
import pathlib
import random
import subprocess
import sys

import numpy as np

from maker_arm_lab.core import parts

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "render_set_sizes.py"


def _load():
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_usage_errors_print_the_fail_token():
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False)
    assert out.returncode == 2
    assert out.stdout.strip().startswith("SIZES_FAIL ")


def test_wrist_rows_take_their_selection_target_s_size(tmp_path):
    # env 0 selects brick_1x6 (index 4, then a plate beside it) at step 5 and lifts the plate at
    # 9; env 1 lifts at 3 before any selection of its own, then selects a tile at 7.
    rows = [
        ("overhead", "brick_1x6", 4, 0, 5),
        ("overhead", "plate_1x4", 7, 0, 5),
        ("lift", "plate_1x4", 7, 0, 9),
        ("lift", "", -1, 1, 3),
        ("overhead", "tile_2x2", 2, 1, 7),
        ("carry", "tile_2x2", 2, 1, 22),
    ]
    view, name, held, env, step = (np.array(c) for c in zip(*rows))
    path = tmp_path / "render_42.npz"
    args = np.array(json.dumps({"run_seed": 42}))
    np.savez(path, view=view, part=name, held=held, env=env, step=step, args=args)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True, check=False
    )
    lines = out.stdout.strip().splitlines()
    assert out.returncode == 0 and len(lines) == 1 and lines[0].startswith("SIZES_OK ")
    assert json.loads(lines[0].split(" ", 1)[1]) == {"files": 1, "rows": 6, "other": 1}
    got = np.load(tmp_path / "render_42.size.npy")
    for n in (0, 1, 4):
        true = sorted(parts.by_name([name[n]])[0].size)
        assert np.all(np.abs(got[n] - true) < 0.01) and list(got[n]) == sorted(got[n])
    assert (got[2] == got[0]).all() and (got[5] == got[4]).all() and (got[3] == -1.0).all()
    render = _load()
    assert render.perceived((0.01, 0.02, 0.03), random.Random(0)) == render.perceived(
        (0.01, 0.02, 0.03), random.Random(0)
    )
