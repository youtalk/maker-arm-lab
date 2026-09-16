"""Heap mode of MjLegoSortEnv and the recorder's heap datasets (design sections 6, 8, 9)."""

import json
import random

import h5py
import mujoco
import numpy as np
import pytest  # noqa: F401 -- used by the env tests a later task appends to this file
import torch

from maker_arm_lab.core import layout, parts  # noqa: F401 -- layout: see pytest note above
from maker_arm_lab.core.heap_oracle import STOW_HALF_EXTENTS  # noqa: F401 -- see pytest note above
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj import overlay  # noqa: F401 -- see pytest note above
from maker_arm_lab.mj.context_names import CONTEXT_BOX_NAMES, CONTEXT_DIMS
from maker_arm_lab.mj.env import MjLegoSortEnv  # noqa: F401 -- see pytest note above
from maker_arm_lab.mj.recorder import ShardRecorder

PARTS = [
    parts.make_part("brick", 2, 4),
    parts.make_part("plate", 1, 2),
    parts.make_part("tile", 1, 1),
    parts.make_part("slope", 1, 2),
    parts.make_part("brick", 1, 1),
    parts.make_part("plate", 2, 2),
]


def _draw(seed: int = 5):
    draw = sample_run_draw(random.Random(seed))
    draw["slot_colors"] = (0, 1, 2, 3)
    return draw


def _hold_action(env) -> torch.Tensor:
    q = env.scene["robot"].data.joint_pos[:, :6].clone().float()
    return torch.cat([q, torch.ones(env.num_envs, 1)], dim=1)


def _can_render() -> bool:
    try:
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        mujoco.Renderer(model, 8, 8).close()
        return True
    except Exception:  # noqa: BLE001
        return False


def test_the_recorder_writes_the_heap_datasets(tmp_path):
    rec = ShardRecorder(tmp_path / "s.hdf5", 1, export_all=True, cameras=True, depth=True)
    state = {
        "robot_joint_position": np.zeros(8, np.float32),
        "robot_joint_velocity": np.zeros(8, np.float32),
        "robot_root_pose": np.zeros(7, np.float32),
        "robot_root_velocity": np.zeros(6, np.float32),
        "brick_root_pose": np.zeros((2, 7), np.float32),
        "brick_root_velocity": np.zeros((2, 6), np.float32),
        "bin_layout": np.zeros(5, np.float32),
        "destination_bin": np.array([2], np.int64),
    }
    rec.begin(0, state, parts=["brick_2x4", "tile_1x1"])
    blank = np.zeros((224, 224, 3), np.uint8)
    for _ in range(3):
        rec.pre_step(
            0,
            np.zeros(8),
            np.zeros(7),
            1,
            0,
            blank,
            blank,
            context_camera=np.zeros(6),
            context_privileged=np.zeros(66),
            actor=1,
            depth=np.full((224, 224), 900, np.uint16),
            target_part=0,
            context_box=np.array([0.44, -0.07, 0.0], np.float32),
        )
        rec.post_step(0, np.zeros(8), False)
    assert rec.end(0, False)
    rec.close()
    with h5py.File(tmp_path / "s.hdf5", "r") as f:
        g = f["data/demo_0"]
        assert g["obs/images/overhead_depth"].shape == (3, 224, 224)
        assert g["obs/images/overhead_depth"].dtype == np.uint16
        assert int(g["obs/images/overhead_depth"][0, 10, 10]) == 900
        assert g["obs/target_part"][:].tolist() == [0, 0, 0]
        assert g["obs/target_part"].dtype == np.int16
        assert g["obs/context_box"].shape == (3, 3)
        assert json.loads(g["obs/context_box"].attrs["names"]) == list(CONTEXT_BOX_NAMES)
        assert json.loads(g.attrs["parts"]) == ["brick_2x4", "tile_1x1"]
        assert g["initial_state/destination_bin"][:].tolist() == [2]
    assert CONTEXT_DIMS["camera_box"] == 9
