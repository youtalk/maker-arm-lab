"""The overhead depth noise model (design section 9)."""

import numpy as np

from maker_arm_lab.mj import depth


def test_depth_becomes_uint16_millimeters_in_constant_blocks():
    true = np.full((224, 224), 0.9, dtype=np.float32)
    bright = np.full((224, 224, 3), 200, dtype=np.uint8)
    out = depth.noisy_depth_mm(true, bright, np.random.default_rng(0))
    assert out.shape == (224, 224) and out.dtype == np.uint16
    assert (out > 0).all()
    assert np.abs(out.astype(int) - 900).max() <= 5 * depth.DEPTH_NOISE_STD_M * 1000
    block = out[: depth.DEPTH_BLOCK_PX, : depth.DEPTH_BLOCK_PX]
    assert (block == block[0, 0]).all()
    assert out.std() > 0.5  # the noise is there


def test_dark_blocks_drop_out_as_holes_and_bright_ones_do_not():
    true = np.full((224, 224), 0.9, dtype=np.float32)
    dark = np.zeros((224, 224, 3), dtype=np.uint8)
    out = depth.noisy_depth_mm(true, dark, np.random.default_rng(1))
    fraction = float((out == 0).mean())
    assert 0.2 < fraction < 0.4
    bright = np.full((224, 224, 3), 200, dtype=np.uint8)
    assert (depth.noisy_depth_mm(true, bright, np.random.default_rng(1)) > 0).all()


def test_depth_clips_to_the_uint16_range():
    far = np.full((224, 224), 70.0, dtype=np.float32)
    bright = np.full((224, 224, 3), 200, dtype=np.uint8)
    assert depth.noisy_depth_mm(far, bright, np.random.default_rng(0)).max() == depth.DEPTH_MAX_MM
    near = np.zeros((224, 224), dtype=np.float32)
    assert depth.noisy_depth_mm(near, bright, np.random.default_rng(0)).min() >= 1
