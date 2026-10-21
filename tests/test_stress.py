import argparse

import pytest

from maker_arm_lab.mj.stress import Stress, add_stress_arguments, stress_from_args


def parse(flags):
    parser = argparse.ArgumentParser()
    add_stress_arguments(parser)
    parser.add_argument("--frame_jitter", action="store_true")
    parser.add_argument("--tray", action="store_true")
    parser.add_argument("--tray_refill", action="store_true")
    return parser.parse_args(flags)


def test_all_off_is_the_empty_block():
    assert Stress().block() == {} and stress_from_args(parse([])) == Stress()


def test_the_block_names_only_the_knobs_that_are_on():
    s = stress_from_args(parse(["--overhead_offset", "0,3,0,0,0,0", "--jpeg_quality", "85"]))
    assert s.block() == {"overhead_offset": [0.0, 3.0, 0.0, 0.0, 0.0, 0.0], "jpeg_quality": 85}
    assert s.overhead_camera_on() and not Stress(wrist_offset=(0.0,) * 6).overhead_camera_on()


@pytest.mark.parametrize(
    "flags",
    [
        ["--overhead_fov_delta", "2"],
        ["--warp_back"],
        ["--overhead_offset", "1,2,3"],
        ["--jpeg_quality", "101"],
        ["--blur_sigma=-1"],
        ["--obs_delay=-1"],
        ["--overhead_fov_delta", "2", "--warp_back", "--frame_jitter"],
        ["--sparse_gap", "0.05"],
        ["--sparse_gap", "0.05", "--tray", "--tray_refill"],
    ],
)
def test_bad_combinations_are_refused(flags):
    with pytest.raises(ValueError):
        stress_from_args(parse(flags))


def test_the_heap_client_reports_a_bad_stress_flag_as_a_usage_failure(capsys):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "mj_eval_heap.py"
    spec = importlib.util.spec_from_file_location("mj_eval_heap_stress", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit):
        mod.parse_args(["--heap_k", "5", "--warp_back"])
    assert 'EVAL_FAIL {"reason": "usage"' in capsys.readouterr().out
