"""MS7j design sections 6.1 and 6.4: the readiness suite's rows and its tolerance table."""

import json
import math
import shlex
import statistics
import subprocess
import sys

import pytest

from helpers import ROOT, load_script
from maker_arm_lab.mj.stress import stress_from_args

SCRIPT = ROOT / "scripts" / "readiness_suite.py"
suite = load_script(SCRIPT)
ge = load_script("run_gate_eval")
SCENE = {
    "backend",
    "num_envs",
    "heap_k",
    "parts_seed",
    "parts_split",
    "episode_draws",
    "context",
    "gripper",
    "tray",
    "desk_boxes",
    "tray_lean",
    "deep_tray",
    "deep_box_floor",
    "sparse_gap",
    "perceived",
    "max_wall_minutes",
    "eval_timeout_s",
    "max_cycles_per_stowed",
    "threshold",
    "min_stows_per_hour",
    "heldout_threshold",
}


def _flags(argv) -> dict:
    out, key = {}, None
    for token in argv:
        if token.startswith("--"):
            key = token[2:]
            assert key not in out, f"--{key} twice"
            out[key] = True
        else:
            out[key] = token
    return out


def test_the_suite_serves_the_real_clients_rules():
    real = load_script("real_eval_heap")
    served = {
        **real.SERVED_SWITCHES,
        **real.SERVED_RIG,
        **real.SERVED_ORACLE,
        **real.SERVED_INFERENCE,
    }
    flags = {**_flags(suite.BASE_ARGS), **_flags(shlex.split(suite.BASE_CLIENT))}
    assert set(flags) <= set(served) | SCENE, set(flags) - set(served) - SCENE
    for key, value in served.items():
        assert key in flags, key
        assert flags[key] is True if value is True else float(flags[key]) == value, key
    argv = [*suite.BASE_ARGS, "--checkpoint", "/x", "--seed", "1", "--ledger", "/x"]
    argv += ["--processes", "1", "--out_dir", "/tmp/x", "--client_flags", suite.BASE_CLIENT]
    cfg = ge._build_cfg(ge._parse_args(argv))
    assert (cfg.heap_k, cfg.num_envs, cfg.desk_boxes, cfg.threshold) == (10, 5, 4, 0.0)
    assert cfg.families == () and cfg.box_draw == 0 and not cfg.frame_jitter
    assert cfg.min_stows_per_hour is None and cfg.heldout_threshold is None
    assert ge.stress_block(cfg.client_flags) == {"sparse_gap": 0.035}


SPOT = {
    "overhead_pose/tilt/-3": {"overhead_offset": [-3, 0, 0, 0, 0, 0]},
    "overhead_pose/height/0.05": {"overhead_offset": [0, 0, 0, 0, 0, 0.05]},
    "overhead_pose/forward/-0.01": {"overhead_offset": [0, 0, 0, -0.01, 0, 0]},
    "overhead_pose/lateral/0.01": {"overhead_offset": [0, 0, 0, 0, 0.01, 0]},
    "wrist_pose/pitch/5": {"wrist_offset": [5, 0, 0, 0, 0, 0]},
    "wrist_pose/yaw/2": {"wrist_offset": [0, 2, 0, 0, 0, 0]},
    "wrist_pose/forward/0.01": {"wrist_offset": [0, 0, 0, 0, 0, 0.01]},
    "wrist_pose/lateral/0.002": {"wrist_offset": [0, 0, 0, 0.002, 0, 0]},
    "mount/height/-0.04": {"overhead_offset": [0, 0, 0, 0, 0, -0.04], "warp_back": True},
    "mount/tilt/-6": {"overhead_offset": [-6, 0, 0, 0, 0, 0], "warp_back": True},
    "mount/wrist/0.01": {"wrist_offset": [0, 0, 0, 0.01, 0, 0], "warp_back": True},
    "fov/wrist/-10": {"wrist_fov_delta": -10, "warp_back": True},
    "image/exposure/-1": {"exposure_stops": -1},
    "light/hard/on": {"hard_light": True},
}


def test_every_row_parses_with_the_clients_stress_parser():
    heap = load_script("mj_eval_heap")
    assert len({r.name for r in suite.ROWS}) == len(suite.ROWS) == 59
    for row in suite.ROWS:
        client = shlex.split(f"{suite.BASE_CLIENT} {row.client}")
        block = stress_from_args(heap.parse_args(["--heap_k", "10", "--tray", *client])).block()
        assert block == ge.stress_block(client), row.name  # the supervisor's cache key
        assert block.pop("sparse_gap") == 0.035
        assert bool(block) != bool(row.server), row.name  # one knob: the client's or the server's
        if row.server:
            assert row.server == f"--infer_delay {row.level}"
        elif row.level != "on":
            on = [v for x in block.values() for v in (x if isinstance(x, list) else [x])]
            assert [v for v in on if v is not True and v != 0] == [float(row.level)], row.name
        if row.name in SPOT:
            assert block == SPOT[row.name], row.name


EXPECTED = {
    "obs_delay/steps/3",
    "infer_delay/steps/1",
    *(f"overhead_pose/tilt/{v}" for v in ("1", "-1")),
    *(
        f"overhead_pose/{a}/{v}"
        for a in ("height", "forward", "lateral")
        for v in ("0.01", "-0.01")
    ),
    "wrist_pose/pitch/2",
    "wrist_pose/yaw/2",
    "wrist_pose/forward/0.002",
    "wrist_pose/lateral/0.002",
    "mount/height/0.04",
    "mount/height/-0.04",
    *(f"mount/{a}/{v}" for a in ("forward", "lateral") for v in ("0.02", "-0.02")),
    "mount/tilt/3",
    "mount/tilt/-3",
    *(f"fov/{a}/{v}" for a in ("overhead", "wrist") for v in ("2", "-2")),
    "image/jpeg/85",
    "image/blur/1.0",
    "image/exposure/1",
    "image/exposure/-1",
    "light/hard/on",
}


def _cli(*args):
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )
    return out.returncode, out.stdout.strip().splitlines()


def test_the_expected_rows_are_the_a1_levels():
    assert {r.name for r in suite.ROWS if r.expected} == EXPECTED
    only = {r.name: (r.act, r.pi05) for r in suite.ROWS if not (r.act and r.pi05)}
    assert only == {
        "infer_delay/steps/1": (True, False),
        "infer_delay/steps/6": (False, True),
        "infer_delay/steps/12": (False, True),
    }
    code, lines = _cli("rows", "--candidate", "pi05", "--expected")
    assert code == 0 and lines[-1].startswith("SUITE_OK ")
    rows = [json.loads(line) for line in lines[:-1]]
    assert rows[0]["name"] == "base" and {r["name"] for r in rows[1:]} == EXPECTED - {
        "infer_delay/steps/1"
    }
    tilt = next(r for r in rows if r["name"] == "overhead_pose/tilt/-1")
    assert tilt["argv"] == [
        *suite.BASE_ARGS,
        "--client_flags",
        f"{suite.BASE_CLIENT} --overhead_offset=-1,0,0,0,0,0",
    ]
    code, lines = _cli("rows", "--candidate", "pi05")
    delay = next(json.loads(x) for x in lines if '"infer_delay/steps/12"' in x)
    assert delay["argv"][-4:-2] == ["--infer_delay", "12"]
    assert delay["argv"][-1] == suite.BASE_CLIENT


def _episode(correct, parts=50, steps=300):
    return {
        "bricks_stowed": correct,
        "num_bricks": parts,
        "wrong_box_parts": 0,
        "cycle_classes": [],
        "cycle_census": [],
        "stowed_parts": list(range(correct)),
        "steps": steps,
    }


def _payload(i, correct, stress=None, run_seed=None):
    return {
        "run_seed": 100 + i if run_seed is None else run_seed,
        "layout_seed": 600 + i,
        "permutation": [0, 1, 2, 3],
        "parts": ["brick_2x4"] * 50,
        "inference": {"n_action_steps": 50},
        "stress": {"sparse_gap": 0.035, **(stress or {})},
        "episodes": [_episode(correct)],
    }


def _draws(counts, **kw):
    return {f"eval_{i:03d}.json": _payload(i, c, **kw) for i, c in enumerate(counts)}


def test_paired_drop_and_se_on_known_counts():
    got = suite.paired(_draws((40, 45, 50)), _draws((38, 45, 44)))
    assert got["n"] == 3
    assert math.isclose(got["drop"], 100 * 8 / 150)
    assert math.isclose(got["se"], 100 * math.sqrt(3) * statistics.stdev([2, 0, 6]) / 150)
    assert (got["passes"], got["better"], got["same"], got["worse"]) == (False, 0, 1, 2)
    got = suite.paired(_draws((40, 45, 50)), _draws((41, 45, 49)))
    assert got["drop"] == 0 and got["passes"] and (got["better"], got["worse"]) == (1, 1)


TILT = {"overhead_offset": [1, 0, 0, 0, 0, 0]}


def _write(path, draws):
    path.mkdir()
    for name, payload in draws.items():
        (path / name).write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize(
    "draws",
    [
        _draws((40, 45), stress=TILT),  # eval_002.json is missing
        {**_draws((40, 45, 50), stress=TILT), "eval_001.json": _payload(1, 45, TILT, run_seed=7)},
        _draws((40, 45, 50, 50), stress=TILT),  # a draw the baseline lacks
        _draws((40, 45, 50), stress={"overhead_offset": [-1, 0, 0, 0, 0, 0]}),  # another row's
    ],
    ids=["missing", "seed", "extra", "knobs"],
)
def test_an_unpaired_row_is_refused(tmp_path, draws):
    base = _write(tmp_path / "base", _draws((40, 45, 50)))
    row = _write(tmp_path / "row", draws)
    out = tmp_path / "t.md"
    code, lines = _cli(
        "table", "--base", base, "--row", f"overhead_pose/tilt/1={row}", "--out", out
    )
    assert code == 1 and len(lines) == 1 and lines[0].startswith("SUITE_FAIL "), lines
    assert not out.exists()
    good = _write(tmp_path / "good", _draws((40, 44, 50), stress=TILT))
    code, lines = _cli(
        "table", "--base", base, "--row", f"overhead_pose/tilt/1={good}", "--out", out
    )
    assert code == 0 and lines[-1].startswith("SUITE_OK ") and out.exists(), lines
    token = json.loads(lines[-1].split(" ", 1)[1])
    assert token["largest"] == {"overhead_pose/tilt": None}  # the -1 row was not read
    assert token["failed_expected"] == []


def _passes(*names_and_flags):
    return {name: {"passes": ok} for name, ok in names_and_flags}


def test_largest_passing_needs_both_signs_and_every_smaller_level():
    results = _passes(
        ("overhead_pose/tilt/1", True),
        ("overhead_pose/tilt/-1", True),
        ("overhead_pose/tilt/3", True),
        ("overhead_pose/tilt/-3", False),
        ("overhead_pose/tilt/6", True),
        ("overhead_pose/tilt/-6", True),
        ("overhead_pose/height/0.01", True),
        ("overhead_pose/height/-0.01", False),
        ("overhead_pose/height/0.05", True),
        ("overhead_pose/height/-0.05", True),
        ("wrist_pose/pitch/2", True),
        ("wrist_pose/pitch/5", True),
        ("mount/height/0.04", True),  # its -0.04 mirror is missing
        ("light/hard/on", True),
        ("infer_delay/steps/6", True),
        ("infer_delay/steps/12", False),
    )
    assert suite.largest_passing(results) == {
        "overhead_pose/tilt": "1",
        "overhead_pose/height": None,
        "wrist_pose/pitch": "5",
        "mount/height": None,
        "light/hard": "on",
        "infer_delay/steps": "6",
    }


def test_lifted_counts_parts_not_cycles():
    first = {
        "bricks_stowed": 2,
        "num_bricks": 10,
        "wrong_box_parts": 1,
        "stowed_parts": [0, 1],
        "cycle_classes": [
            "stowed",
            "lifted_lost",
            "lifted_lost",
            "close_miss",
            "wrong_box",
            "carrying",
            "held_no_lift",
            "lifted_lost",
        ],
        "cycle_census": [{"last_held": b} for b in (0, 2, 2, 3, 4, 5, 6, -1)],
        "steps": 600,
    }
    second = {**_episode(1, parts=10, steps=300), "wrong_box_parts": 0}
    got = suite.measures({"episodes": [first, second]})
    # parts 0, 1, 2, 4 and 5 of the first episode, and part 0 of the second
    assert got == {"parts": 20, "lifted": 6, "correct": 3, "wrong": 1, "never": 14, "steps": 900}
