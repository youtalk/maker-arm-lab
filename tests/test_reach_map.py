"""scripts/reach_map.py: the maximal-rectangle scan, the token contract and a solver sanity read."""

import importlib.util
import json
import math
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
    # nothing here is out of band, so the admissible group matches the unrestricted one
    assert picks["admissible_all_headings"]["overall"] == picks["overall"]


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
    assert picks["admissible_all_headings"]["overall"] is None
    assert picks["admissible_all_headings"]["note"] == reach.ADMISSIBLE_ALL_HEADINGS_NOTE


def test_admissible_all_headings_recovers_an_in_band_sub_rectangle_of_a_larger_out_of_band_run():
    """maximal_rectangles's own docstring says it does not enumerate every sub-rectangle a run
    contains, so filtering its output after the fact can miss an in-band rectangle that is a
    proper sub-rectangle of a larger, out-of-band one. This mask reproduces exactly that shape:
    every row is pickable out to x=0.15 (a plain solver read, radius-unaware, uniform across all
    three rows), so the only rectangle maximal_rectangles(mask) can ever build from the full
    block is the whole 3-row slab. That slab pokes past layout.SCATTER_MAX_RADIUS at its outer
    rows (y = +-0.278), so every candidate the old post-hoc filter sees fails, even at column
    width zero. The middle row alone (y = 0.0) stays in band across the full column width: a
    proper sub-rectangle of the out-of-band slab that the old filter could never reach.
    """
    reach = _load(REACH)
    xs = [0.0, 0.05, 0.10, 0.15]
    ys = [-0.278, 0.0, 0.278]
    mask = [[True, True, True, True] for _ in ys]
    picks = reach.best_rectangles(mask, xs, ys, step=0.05)

    # The pre-fix approach: search the raw mask, then filter its candidates by radius. It can
    # only pick among the candidates maximal_rectangles(mask) already produced, and every one of
    # those spans all three rows, so every one of them pokes past the bound at y = +-0.278.
    filtered_best_area = 0.0
    for rect in reach.maximal_rectangles(mask):
        info = reach.rectangle_info(rect, xs, ys, step=0.05)
        if info["within_scatter_max_radius"]:
            filtered_best_area = max(filtered_best_area, info["area_m2"])
    assert filtered_best_area == 0.0  # the old filter recovers nothing at all from this mask

    admissible = picks["admissible_all_headings"]["overall"]
    assert admissible is not None
    assert admissible["within_scatter_max_radius"] is True
    assert admissible["area_m2"] > filtered_best_area  # the fixed search finds strictly more
    assert admissible["y_range"] == [-0.025, 0.025]  # the middle row only, padded by step / 2
    assert admissible["x_range"] == [-0.025, 0.175]  # the full column width, padded by step / 2


def test_current_tray_admissible_picks_are_all_within_the_padded_bound():
    """The tray sits inside SCATTER_MAX_RADIUS by construction (tests/test_layout.py) and its
    four corners are pickable at every heading (test_current_tray_corners_are_pickable_by_the_
    real_solver). A single admissible rectangle needs every point along its edges to pass the
    same test, not only the four corners, so this runs the real solver over the tray footprint.

    Whether a group covers the tray as one contiguous rectangle depends on the grid step: a
    coarser or finer --step can move a candidate's edge onto or off of the tray's own boundary.
    Read the two asserts below as what this particular grid reports, not as a fixed geometric
    fact about the tray. The tray's own far corner is at radius 0.2786 m; padded by half this
    grid's step (0.01 m) its corner reads 0.2915 m, past the 0.28 m bound, so neither group
    reaches that far here.

    What is a fixed fact, independent of the grid: every rectangle either group returns must
    carry within_scatter_max_radius True, because both groups now search a mask already
    clipped on the same padded-corner geometry rectangle_info reports.
    """
    reach = _load(REACH)
    (x0, x1), (y0, y1) = layout.tray_bounds()
    args = reach.parse_args(
        [
            "--x_min",
            str(x0),
            "--x_max",
            str(x1),
            "--y_min",
            str(y0),
            "--y_max",
            str(y1),
            "--step",
            "0.02",
            "--headings",
            "8",
            "--out",
            "/dev/null",  # unused: run() never touches args.out, only main() writes it
        ]
    )
    result = reach.run(args)

    def covers_tray(rect):
        if rect is None:
            return False
        rx0, rx1 = rect["x_range"]
        ry0, ry1 = rect["y_range"]
        return rx0 <= x0 and rx1 >= x1 and ry0 <= y0 and ry1 >= y1

    all_headings = result["rectangles"]["admissible_all_headings"]
    any_heading = result["rectangles"]["admissible_any_heading"]
    assert not covers_tray(all_headings["overall"]), (
        "admissible_all_headings now covers the tray -- update this test"
    )
    assert not covers_tray(any_heading["overall"]), (
        "admissible_any_heading now covers the tray -- update this test"
    )
    for group in (all_headings, any_heading):
        for key in ("overall", "current_tray_ratio", "square", "widest"):
            rect = group[key]
            if rect is not None:
                assert rect["within_scatter_max_radius"] is True


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


def test_the_dish_rim_is_pickable_by_the_real_solver():
    """The script's own sanity check: the dish sits inside SCATTER_MAX_RADIUS by construction
    (tests/test_layout.py); this checks the solver actually agrees that every one of eight
    points spaced evenly around the rim is reachable at the descend attitude."""
    reach = _load(REACH)
    cx, cy = layout.TRAY_CENTER
    ik = reach.GlobalIK()
    tilt = reach.MIN_DESCEND_TILT
    headings = reach.headings_over_half_turn(8)
    half_height = BRICK_HEIGHT / 2
    for k in range(8):
        theta = 2.0 * math.pi * k / 8
        x = cx + layout.TRAY_RADIUS * math.cos(theta)
        y = cy + layout.TRAY_RADIUS * math.sin(theta)
        converged = reach.probe_point(ik, x, y, half_height, tilt, headings)
        assert all(converged), f"dish rim point ({x}, {y}) is not pickable at every heading"
