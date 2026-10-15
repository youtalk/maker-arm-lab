"""scripts/mj_part_table.py: the token contract, the seed rule, the MS7e part order and one real
trial (about 2 s on Machine A, so it runs in the CPU suite)."""

import json
import random
import subprocess
import sys

from helpers import ROOT, load_script
from maker_arm_lab.core import parts

SCRIPT = ROOT / "scripts" / "mj_part_table.py"


def _run(*argv):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, check=False
    )


def test_usage_errors_print_the_fail_token():
    for argv in (
        (),
        ("--parts", "brick_2x4", "--families", "round"),
        ("--families", "nope"),
        ("--parts", "brick_9x9"),
        ("--parts", "brick_2x4", "--trials", "0"),
    ):
        out = _run(*argv)
        assert out.returncode == 2, argv
        lines = out.stdout.strip().splitlines()
        assert len(lines) == 1 and lines[0].startswith("TABLE_FAIL "), argv
        assert json.loads(lines[0].split(" ", 1)[1])["reason"] == "usage"


def test_the_ms7e_part_order_and_its_families():
    table = load_script(SCRIPT)
    order = table.part_order()
    assert order[:29] == [p.name for p in parts.catalog()]
    assert len(order) == len(set(order)) == 51 + len(parts.JOINT_IDS) + len(parts.MINIFIG_IDS)
    pins = {
        "round_1x1": 29,
        "round_4x4": 32,
        "ldraw_3700": 33,
        "ldraw_3660": 40,
        "plate_1x2": 41,
        "plate_2x6": 46,
        "tile_1x2": 47,
        "tile_2x4": 50,
    }
    for name, index in pins.items():
        assert order.index(name) == index, name
    assert [parts.catalog_index(n) for n in order] == list(range(len(order)))
    assert table.family_parts("joint") == order[51:59] == [f"joint_{i}" for i in parts.JOINT_IDS]
    assert table.family_parts("minifig") == order[59:]
    assert table.family_parts("round") == order[29:33]
    assert table.family_parts("flat") == order[41:51]
    assert table.family_parts("tile") == order[47:51]
    assert len(table.family_parts("brick")) == 23 and len(table.family_parts("slope")) == 6


def test_the_seed_rule():
    table = load_script(SCRIPT)
    index = parts.catalog_index("brick_2x4")
    want = random.Random(20289000 + index)
    assert table.trial_seeds("brick_2x4", 12) == [want.randrange(2**31) for _ in range(12)]
    # A shorter table is a prefix: trial t's seed never depends on the trial count.
    assert table.trial_seeds("brick_2x4", 3) == table.trial_seeds("brick_2x4", 12)[:3]
    assert table.trial_seeds("round_1x1", 3) != table.trial_seeds("brick_2x4", 3)
    # --seed_base moves the base, and the default keeps MS7e's (MS7f ledger R1: 20303000).
    want = random.Random(20303000 + index)
    assert table.trial_seeds("brick_2x4", 2, 20303000) == [want.randrange(2**31) for _ in range(2)]
    assert table.parse_args(["--parts", "brick_2x4"]).seed_base == 20289000


def test_one_real_trial_on_brick_2x4_stows(tmp_path):
    out_json = tmp_path / "table.json"
    out = _run("--parts", "brick_2x4", "--trials", "1", "--out", str(out_json))
    lines = out.stdout.strip().splitlines()
    assert out.returncode == 0, out.stdout + out.stderr
    assert sum(line.startswith(("TABLE_OK ", "TABLE_FAIL ")) for line in lines) == 1
    assert lines[-1].startswith("TABLE_OK ")
    result = json.loads(out_json.read_text())
    (record,) = result["records"]
    assert record["seed"] == load_script(SCRIPT).trial_seeds("brick_2x4", 1)[0]
    # The base control: the teacher lifts and stows a 2x4 (84 of 84 on 2026-09-29).
    assert record["lifted"] and record["stowed"] and record["end_phase"] == "RETREAT>DONE"
    assert record["lifts"][0]["gate"]["force_n"][0] > 0.5
    assert result["families"]["brick"]["stowed"] == 1
    assert result["scene"]["eight_classes"] is False  # the control keeps MS7d's class map


def test_every_family_and_switch_builds_in_the_table_scene(tmp_path):
    out_json = tmp_path / "table.json"
    argv = ["--parts", "round_2x2,ldraw_3700,plate_2x4", "--trials", "1", "--out", str(out_json)]
    out = _run(*argv, "--deep_box_floor", "--box_draw", "8", "--release_lean")
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.strip().splitlines()[-1].startswith("TABLE_OK ")
    scene = json.loads(out_json.read_text())["scene"]
    assert scene["eight_classes"] and scene["deep_box_floor"] and scene["release_lean"]
    assert scene["box_draw"] == 8
    assert _run("--parts", "brick_2x4", "--box_draw", "8").returncode == 2  # B1 ceiling 6
