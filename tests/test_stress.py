import argparse

import numpy as np
import pytest

from maker_arm_lab.mj.stress import Stress, add_stress_arguments, degrade, stress_from_args


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


def _textured():
    return np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)


def test_degrade_with_the_knobs_off_is_the_same_object():
    img = np.zeros((224, 224, 3), np.uint8)
    assert degrade(img, Stress()) is img


def test_one_stop_up_brightens_mid_grey_to_about_175():
    out = degrade(np.full((8, 8, 3), 128, np.uint8), Stress(exposure_stops=1.0))
    assert 170 <= int(out[0, 0, 0]) <= 180


def test_jpeg_and_blur_change_a_textured_frame_a_little():
    img = _textured()
    for s in (Stress(jpeg_quality=85), Stress(blur_sigma=1.0)):
        out = degrade(img, s)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert 0 < np.abs(out.astype(int) - img).mean() < 80


def test_blur_sigma_scales_with_the_frame_width():
    big = np.kron(_textured(), np.ones((2, 2, 1), np.uint8))
    small = degrade(_textured(), Stress(blur_sigma=1.0)).astype(int)
    out = degrade(big, Stress(blur_sigma=1.0)).astype(int)
    # sigma 2 at 448 px is the same blur as sigma 1 at 224 px, up to resampling
    assert np.abs(out[::2, ::2] - small).mean() < 12
