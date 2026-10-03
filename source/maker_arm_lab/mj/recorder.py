"""HDF5 shard writer with the layout ``lerobot_stack/convert_to_lerobot.py`` reads.

One shard per generation run, ``data/demo_N`` per exported episode, the same datasets Isaac
Lab's recorder wrote for MS1 and MS2 (the converter and ``extract_replay_bundle.py`` are the
consumers and are unchanged):

    attrs: success (bool), num_samples (int)
    actions (T,7) f32            raw pre-clamp oracle command, column 6 the binary gripper
    joint_targets (T,8) f32      the post-clamp targets applied (6 arm + 2 jaws)
    obs/joint_pos_abs (T,8) f32  pre-step joint positions, URDF order
    obs/images/wrist_rgb, obs/images/overhead_rgb (T,224,224,3) u8   pre-step frames
    obs/cycle_phase (T,) i8, obs/cycle_brick (T,) i8                  the rig's stash;
        cycle_brick is i16 on a heap of more than MAX_INT8_PART parts (MS7f, 150-part heaps)
    obs/clamped (T,) bool        the Rust clamp changed this step's target
    obs/context_camera (T,6) f32, obs/context_privileged (T,66) f32   the two context
        channels (MS4 levers 1 and 3), always recorded; each carries a JSON `names` attr
    obs/actor (T,) i16           who commanded the frame: 1 the oracle (batch loop), k the
        k-th DAgger takeover of the episode, negative when that takeover's cycle did not
        stow its brick
    obs/images/overhead_depth (T,224,224) u16   MS6 heap shards only: millimeters, 0 = hole
    obs/target_part (T,) i16     the part the overhead marker points at, -1 for none
    obs/context_box (T,3) f32    destination bin x, y, yaw (MS6), with a JSON `names` attr
    initial_state/articulation/robot/{joint_position,joint_velocity,root_pose,root_velocity}
    initial_state/rigid_object/brick_N/{root_pose,root_velocity}     one leading axis of 1
    initial_state/bin_layout (1,5) f32: bin_x and the four slot y values of this episode's
        layout
    initial_state/destination_bin (1,) i64      MS6 heap shards only; (K,) one box per part
        with desk boxes (MS7a)
    initial_state/desk_box_poses (1,M,3) f32    desk box shards only: each box's x, y, yaw
    initial_state/desk_box_sizes (1,M,3) f32    box draw shards only (MS7e): each box's
        interior; initial_state/box_count (1,) i64 the boxes on the desk (the rest are parked)
        and initial_state/box_draw_rounds (1,) i64 the draw's extra size-draw rounds
    initial_state/class_to_box (C,) i64         desk box shards only: the box of each class,
        in the order of its JSON `names` attr (the env's class_names, the four
        core.parts.BASE_CLASSES unless an MS7e switch is on)
    initial_state/class_box_mask (C,M) bool     MS7f --categories only: every box of each
        category (the first is class_to_box's), in the same `names` order
    attrs: parts (JSON list of part names, MS6 heap shards only)
    attrs (file level): gripper_mode ("binary" | "continuous", MS6 phase B design 5.4)

Frames are appended per control step. Images go to resizable datasets under ``inflight/``
in chunks of ``flush_every`` frames, so RAM stays flat whatever the episode length; at the
end of the episode the datasets are MOVED (an HDF5 link rename) into the demo group when
the episode exports and deleted otherwise. Deleted datasets leave their space unreclaimed
inside the file, which is fine for a shard that is converted and removed run by run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

from maker_arm_lab.core.parts import BASE_CLASSES
from maker_arm_lab.mj.context_names import (
    CONTEXT_BOX_NAMES,
    CONTEXT_CAMERA_NAMES,
    CONTEXT_PRIVILEGED_NAMES,
    CONTEXT_TARGET_NAMES,
)

IMAGE_KEYS = ("wrist_rgb", "overhead_rgb")
MAX_INT8_PART = 127
"""The largest part index obs/cycle_brick holds as int8, the schema every shard up to MS7e
used; a larger heap (the MS7f heap size, design section 5) writes it as int16 (``wide_ids``)."""
DEPTH_KEY = "overhead_depth"


class ShardRecorder:
    def __init__(
        self,
        path: Path | str,
        num_envs: int,
        *,
        export_all: bool = False,
        cameras: bool = True,
        flush_every: int = 256,
        depth: bool = False,
        gripper_mode: str = "binary",
        class_names: Sequence[str] = BASE_CLASSES,
        wide_ids: bool = False,
    ) -> None:
        self.path = Path(path)
        self._brick_dtype = np.int16 if wide_ids else np.int8
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._class_names = tuple(class_names)
        self._file = h5py.File(self.path, "w")
        self._data = self._file.create_group("data")
        self._inflight = self._file.create_group("inflight")
        self._file.attrs["gripper_mode"] = gripper_mode
        self.num_envs = num_envs
        self.export_all = export_all
        self.cameras = cameras
        self.depth = depth and cameras
        self.image_keys = IMAGE_KEYS + ((DEPTH_KEY,) if self.depth else ())
        self._parts: list[list[str] | None] = [None] * num_envs
        self.flush_every = flush_every
        self.exported_successful_episode_count = 0
        self.exported_episode_count = 0
        self._total_frames = 0
        self._scalars: list[dict[str, list]] = [self._empty() for _ in range(num_envs)]
        self._images: list[dict[str, list[np.ndarray]]] = [
            {k: [] for k in self.image_keys} for _ in range(num_envs)
        ]
        self._image_ds: list[dict[str, h5py.Dataset] | None] = [None] * num_envs
        self._initial: list[dict[str, np.ndarray] | None] = [None] * num_envs
        self._open = [False] * num_envs

    @staticmethod
    def _empty() -> dict[str, list]:
        return {
            "actions": [],
            "joint_targets": [],
            "joint_pos_abs": [],
            "cycle_phase": [],
            "cycle_brick": [],
            "clamped": [],
            "context_camera": [],
            "context_privileged": [],
            "actor": [],
            "target_part": [],
            "context_box": [],
            "context_target": [],
        }

    # -- episode lifecycle --------------------------------------------------

    def begin(
        self,
        env_index: int,
        initial_state: dict[str, np.ndarray],
        parts: list[str] | None = None,
    ) -> None:
        """Start buffering env ``env_index``'s next episode from ``initial_state``."""
        self._drop_inflight(env_index)
        self._scalars[env_index] = self._empty()
        self._images[env_index] = {k: [] for k in self.image_keys}
        self._initial[env_index] = {k: np.asarray(v) for k, v in initial_state.items()}
        self._parts[env_index] = list(parts) if parts is not None else None
        if self.cameras:
            group = self._inflight.create_group(f"env_{env_index}")
            self._image_ds[env_index] = {}
            for key in self.image_keys:
                shape = (224, 224) if key == DEPTH_KEY else (224, 224, 3)
                dtype = np.uint16 if key == DEPTH_KEY else np.uint8
                self._image_ds[env_index][key] = group.create_dataset(
                    key,
                    shape=(0, *shape),
                    maxshape=(None, *shape),
                    dtype=dtype,
                    chunks=(1, *shape),
                    compression="lzf",
                )
        self._open[env_index] = True

    def pre_step(
        self,
        env_index: int,
        joint_pos: np.ndarray,
        action: np.ndarray,
        phase: int,
        brick: int,
        wrist: np.ndarray | None,
        overhead: np.ndarray | None,
        context_camera: np.ndarray | None = None,
        context_privileged: np.ndarray | None = None,
        actor: int = 1,
        depth: np.ndarray | None = None,
        target_part: int = -1,
        context_box: np.ndarray | None = None,
        context_target: np.ndarray | None = None,
    ) -> None:
        s = self._scalars[env_index]
        s["joint_pos_abs"].append(np.asarray(joint_pos, dtype=np.float32).copy())
        s["actions"].append(np.asarray(action, dtype=np.float32).copy())
        s["cycle_phase"].append(np.int8(phase))
        s["cycle_brick"].append(self._brick_dtype(brick))
        if context_camera is not None:
            s["context_camera"].append(np.asarray(context_camera, dtype=np.float32).copy())
        if context_privileged is not None:
            s["context_privileged"].append(np.asarray(context_privileged, dtype=np.float32).copy())
        s["actor"].append(np.int16(actor))
        s["target_part"].append(np.int16(target_part))
        if context_box is not None:
            s["context_box"].append(np.asarray(context_box, dtype=np.float32).copy())
        if context_target is not None:
            s["context_target"].append(np.asarray(context_target, dtype=np.float32).copy())
        if self.cameras:
            imgs = self._images[env_index]
            imgs["wrist_rgb"].append(np.asarray(wrist, dtype=np.uint8).copy())
            imgs["overhead_rgb"].append(np.asarray(overhead, dtype=np.uint8).copy())
            if self.depth:
                imgs[DEPTH_KEY].append(np.asarray(depth, dtype=np.uint16).copy())
            if len(imgs["wrist_rgb"]) >= self.flush_every:
                self._flush_images(env_index)

    def post_step(self, env_index: int, joint_targets: np.ndarray, clamped: bool) -> None:
        s = self._scalars[env_index]
        s["joint_targets"].append(np.asarray(joint_targets, dtype=np.float32).copy())
        s["clamped"].append(bool(clamped))

    def negate_actor(self, env_index: int, ordinal: int) -> None:
        """Mark the buffered frames of takeover ``ordinal`` as a cycle that did not stow its
        brick (MS4 c2 design 6.4): the converter's ``--dagger`` keeps positive actors only."""
        s = self._scalars[env_index]
        s["actor"] = [np.int16(-v) if v == ordinal else v for v in s["actor"]]

    def end(self, env_index: int, success: bool) -> bool:
        """Close env ``env_index``'s episode; returns whether it reached the shard."""
        if not self._open[env_index]:
            return False
        self._open[env_index] = False
        s = self._scalars[env_index]
        steps = min(len(s["actions"]), len(s["joint_targets"]))
        if not (success or self.export_all) or steps == 0:
            self._drop_inflight(env_index)
            return False
        self._flush_images(env_index)
        group = self._data.create_group(f"demo_{self.exported_episode_count}")
        group.attrs["success"] = bool(success)
        group.attrs["num_samples"] = steps
        group.create_dataset("actions", data=np.stack(s["actions"][:steps]))
        group.create_dataset("joint_targets", data=np.stack(s["joint_targets"][:steps]))
        obs = group.create_group("obs")
        obs.create_dataset("joint_pos_abs", data=np.stack(s["joint_pos_abs"][:steps]))
        obs.create_dataset("cycle_phase", data=np.asarray(s["cycle_phase"][:steps], dtype=np.int8))
        brick = np.asarray(s["cycle_brick"][:steps], dtype=self._brick_dtype)
        obs.create_dataset("cycle_brick", data=brick)
        obs.create_dataset("clamped", data=np.asarray(s["clamped"][:steps], dtype=bool))
        obs.create_dataset("actor", data=np.asarray(s["actor"][:steps], dtype=np.int16))
        obs.create_dataset("target_part", data=np.asarray(s["target_part"][:steps], dtype=np.int16))
        for key, names in (
            ("context_camera", CONTEXT_CAMERA_NAMES),
            ("context_privileged", CONTEXT_PRIVILEGED_NAMES),
            ("context_box", CONTEXT_BOX_NAMES),
            ("context_target", CONTEXT_TARGET_NAMES),
        ):
            if s[key]:
                ds = obs.create_dataset(key, data=np.stack(s[key][:steps]))
                ds.attrs["names"] = json.dumps(list(names))
        if self.cameras:
            images = obs.create_group("images")
            for key, ds in self._image_ds[env_index].items():
                if ds.shape[0] != steps:
                    ds.resize(steps, axis=0)
                self._file.move(ds.name, f"{images.name}/{key}")
            del self._inflight[f"env_{env_index}"]
            self._image_ds[env_index] = None
        init = group.create_group("initial_state")
        robot = init.create_group("articulation").create_group("robot")
        state = self._initial[env_index]
        for key in ("joint_position", "joint_velocity", "root_pose", "root_velocity"):
            robot.create_dataset(key, data=state[f"robot_{key}"][None].astype(np.float32))
        if "bin_layout" in state:
            init.create_dataset("bin_layout", data=state["bin_layout"][None].astype(np.float32))
        if "destination_bin" in state:
            init.create_dataset("destination_bin", data=state["destination_bin"].astype(np.int64))
        if "desk_box_poses" in state:
            init.create_dataset(
                "desk_box_poses", data=state["desk_box_poses"][None].astype(np.float32)
            )
        if "desk_box_sizes" in state:  # MS7e box draw
            init.create_dataset(
                "desk_box_sizes", data=state["desk_box_sizes"][None].astype(np.float32)
            )
            for key in ("box_count", "box_draw_rounds"):
                init.create_dataset(key, data=state[key].astype(np.int64))
        if "class_to_box" in state:
            ds = init.create_dataset("class_to_box", data=state["class_to_box"].astype(np.int64))
            ds.attrs["names"] = json.dumps(list(self._class_names))
        if "class_box_mask" in state:
            ds = init.create_dataset("class_box_mask", data=state["class_box_mask"].astype(bool))
            ds.attrs["names"] = json.dumps(list(self._class_names))
        if self._parts[env_index] is not None:
            group.attrs["parts"] = json.dumps(self._parts[env_index])
        rigid = init.create_group("rigid_object")
        for b in range(state["brick_root_pose"].shape[0]):
            brick = rigid.create_group(f"brick_{b}")
            brick.create_dataset(
                "root_pose", data=state["brick_root_pose"][b][None].astype(np.float32)
            )
            brick.create_dataset(
                "root_velocity", data=state["brick_root_velocity"][b][None].astype(np.float32)
            )
        self.exported_episode_count += 1
        self._total_frames += steps
        if success:
            self.exported_successful_episode_count += 1
        return True

    def close(self) -> None:
        for i in range(self.num_envs):
            self._drop_inflight(i)
        self._data.attrs["total"] = self._total_frames
        if "inflight" in self._file:
            del self._file["inflight"]
        self._file.close()

    # -- internals -------------------------------------------------------------

    def _flush_images(self, env_index: int) -> None:
        if not self.cameras or self._image_ds[env_index] is None:
            return
        for key, ds in self._image_ds[env_index].items():
            pending = self._images[env_index][key]
            if not pending:
                continue
            block = np.stack(pending)
            start = ds.shape[0]
            ds.resize(start + block.shape[0], axis=0)
            ds[start:] = block
            pending.clear()

    def _drop_inflight(self, env_index: int) -> None:
        name = f"env_{env_index}"
        if name in self._inflight:
            del self._inflight[name]
        self._image_ds[env_index] = None
        for pending in self._images[env_index].values():
            pending.clear()
