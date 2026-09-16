"""Overhead depth noise model with block-correlated noise and drop-out holes (design section 9).

Adds Gaussian noise at the block level (4×4 pixels per block, one noise sample per block)
and drops out dark surfaces (low RGB luminance) with a fixed probability. Every noise
constant below is a placeholder that MS5 measures on the real iPhone LiDAR mounted on the rig.
"""

from __future__ import annotations

import numpy as np

DEPTH_NOISE_STD_M = 0.003  # INITIAL: MS5 measures the iPhone LiDAR noise at 1 m
DEPTH_BLOCK_PX = 4  # INITIAL: the LiDAR grid at this field of view, 56 x 56 cells
DEPTH_HOLE_LUMINANCE = 40  # INITIAL: dark surfaces drop out
DEPTH_HOLE_PROB = 0.3  # INITIAL: per dark block
DEPTH_MAX_MM = 65535


def noisy_depth_mm(depth_m: np.ndarray, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(H, W) uint16 millimeters from a rendered depth and the RGB frame it pairs with.

    H and W must be multiples of DEPTH_BLOCK_PX (reshape raises ValueError otherwise).
    """
    h, w = depth_m.shape
    b = DEPTH_BLOCK_PX
    blocks = depth_m.reshape(h // b, b, w // b, b).mean(axis=(1, 3))
    blocks = blocks + rng.normal(0.0, DEPTH_NOISE_STD_M, blocks.shape)
    luminance = rgb.astype(np.float32).mean(axis=2).reshape(h // b, b, w // b, b).mean(axis=(1, 3))
    holes = (luminance < DEPTH_HOLE_LUMINANCE) & (rng.random(blocks.shape) < DEPTH_HOLE_PROB)
    mm = np.clip(np.rint(blocks * 1000.0), 1, DEPTH_MAX_MM)
    mm[holes] = 0
    return np.repeat(np.repeat(mm, b, axis=0), b, axis=1).astype(np.uint16)
