"""scripts/reach_map.py: the maximal-rectangle scan, the token contract and a solver sanity read."""

import importlib.util
import json
import pathlib

from maker_arm_lab.core import layout
from maker_arm_lab.core.parts import BRICK_HEIGHT

ROOT = pathlib.Path(__file__).resolve().parents[1]
REACH = ROOT / "scripts" / "reach_map.py"


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_maximal_rectangles_cover_only_pickable_points():
    reach = _load(REACH)
    mask = [
        [True, True, False],
        [True, True, True],
        [False, True, True],
    ]
    rects = reach.maximal_rectangles(mask)
    assert rects
    for row0, row1, col0, col1 in rects:
        for r in range(row0, row1 + 1):
            for c in range(col0, col1 + 1):
                assert mask[r][c] is True
    areas = [(row1 - row0 + 1) * (col1 - col0 + 1) for row0, row1, col0, col1 in rects]
    assert max(areas) == 4  # the two-by-two all-True blocks


def test_best_rectangles_reports_the_four_requested_picks():
    reach = _load(REACH)
    mask = [[True] * 4 for _ in range(2)]  # a 4-wide, 2-tall block: only "widest" should pick it
    xs = [0.0, 0.05, 0.1, 0.15]  # close to the arm base, so every corner is in band
    ys = [0.0, 0.05]
    picks = reach.best_rectangles(mask, xs, ys, step=0.05)
    for key in ("overall", "current_tray_ratio", "square", "widest"):
        assert picks[key] is not None
        assert picks[key]["area_m2"] > 0.0
        assert picks[key]["within_scatter_max_radius"] is True  # this mask sits near the origin
    assert picks["widest"]["size_m"][0] >= picks["square"]["size_m"][0]
    assert picks["admissible"]["overall"] == picks["overall"]  # nothing here is out of band


def test_rectangle_info_flags_within_scatter_max_radius():
    reach = _load(REACH)
    near = reach.rectangle_info((0, 1, 0, 1), [0.0, 0.05], [0.0, 0.05], step=0.05)
    assert near["within_scatter_max_radius"] is True

    far = reach.rectangle_info((0, 1, 0, 1), [0.25, 0.35], [0.15, 0.25], step=0.05)
    assert far["within_scatter_max_radius"] is False
    assert far["corner_radii_m"][-1] > layout.SCATTER_MAX_RADIUS


def test_best_rectangles_admissible_group_excludes_out_of_band_picks():
    reach = _load(REACH)
    mask = [[True, True], [True, True]]
    picks = reach.best_rectangles(mask, [0.25, 0.35], [0.15, 0.25], step=0.05)
    assert picks["overall"]["within_scatter_max_radius"] is False
    assert picks["admissible"]["overall"] is None
    assert picks["admissible"]["note"] == reach.ADMISSIBLE_NOTE


def test_reach_ok_token_and_json_on_a_tiny_grid(tmp_path, capsys):
    reach = _load(REACH)
    out = tmp_path / "reach.json"
    code = reach.main(
        [
            "--x_min",
            "0.15",
            "--x_max",
            "0.155",
            "--y_min",
            "0.0",
            "--y_max",
            "0.0",
            "--step",
            "0.005",
            "--headings",
            "1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1].startswith("REACH_OK ")
    payload = json.loads(lines[-1][len("REACH_OK ") :])
    assert payload["counts"]["total"] == 2
    assert payload["out"] == str(out)
    assert payload["solver_read_only"] is True
    data = json.loads(out.read_text())
    assert len(data["points"]) == 2
    assert data["solver_read_only"] is True


def test_argparse_error_becomes_a_fail_token(capsys):
    reach = _load(REACH)
    code = reach.main(["--nonsense"])
    assert code == 2
    out = capsys.readouterr().out
    assert out.startswith("REACH_FAIL ")
    payload = json.loads(out.strip()[len("REACH_FAIL ") :])
    assert "argparse exit" in payload["error"]


def test_missing_required_out_becomes_a_fail_token(capsys):
    reach = _load(REACH)
    code = reach.main(["--step", "0.1"])
    assert code == 2
    assert capsys.readouterr().out.startswith("REACH_FAIL ")


def test_current_tray_corners_are_pickable_by_the_real_solver():
    """The script's own sanity check: the current 0.13 by 0.20 tray sits inside
    SCATTER_MAX_RADIUS by construction (tests/test_layout.py); this checks the solver actually
    agrees that every one of its four corners is reachable at the descend attitude."""
    reach = _load(REACH)
    (x0, x1), (y0, y1) = layout.tray_bounds()
    ik = reach.GlobalIK()
    tilt = reach.MIN_DESCEND_TILT
    headings = reach.headings_over_half_turn(8)
    half_height = BRICK_HEIGHT / 2
    for x in (x0, x1):
        for y in (y0, y1):
            converged = reach.probe_point(ik, x, y, half_height, tilt, headings)
            assert all(converged), f"tray corner ({x}, {y}) is not pickable at every heading"
