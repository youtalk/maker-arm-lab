"""HDF5 shard writer with the layout ``lerobot_stack/convert_to_lerobot.py`` reads.

One shard per generation run, ``data/demo_N`` per exported episode, the same datasets Isaac
Lab's recorder wrote for MS1 and MS2 (the converter and ``extract_replay_bundle.py`` are the
consumers and are unchanged):

    attrs: success (bool), num_samples (int)
    actions (T,7) f32            raw pre-clamp oracle command, column 6 the binary gripper
    joint_targets (T,8) f32      the post-clamp targets applied (6 arm + 2 jaws)
    obs/joint_pos_abs (T,8) f32  pre-step joint positions, URDF order
    obs/images/wrist_rgb, obs/images/overhead_rgb (T,224,224,3) u8   pre-step frames
    obs/cycle_phase (T,) i8, obs/cycle_brick (T,) i8                  the rig's stash
    obs/clamped (T,) bool        the Rust clamp changed this step's target
    obs/context_camera (T,6) f32, obs/context_privileged (T,66) f32   the two context
        channels (MS4 levers 1 and 3), always recorded; each carries a JSON `names` attr
    obs/actor (T,) i16           who commanded the frame: 1 the oracle (batch loop), k the
        k-th DAgger takeover of the episode, negative when that takeover's cycle did not
        stow its brick
    initial_state/articulation/robot/{joint_position,joint_velocity,root_pose,root_velocity}
    initial_state/rigid_object/brick_N/{root_pose,root_velocity}     one leading axis of 1
    initial_state/bin_layout (1,5) f32: bin_x and the four slot y values of this episode's
        layout

Frames are appended per control step. Images go to resizable datasets under ``inflight/``
in chunks of ``flush_every`` frames, so RAM stays flat whatever the episode length; at the
end of the episode the datasets are MOVED (an HDF5 link rename) into the demo group when
the episode exports and deleted otherwise. Deleted datasets leave their space unreclaimed
inside the file, which is fine for a shard that is converted and removed run by run.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from maker_arm_lab.mj.context_names import CONTEXT_CAMERA_NAMES, CONTEXT_PRIVILEGED_NAMES

IMAGE_KEYS = ("wrist_rgb", "overhead_rgb")


class ShardRecorder:
    def __init__(
        self,
        path: Path | str,
        num_envs: int,
        *,
        export_all: bool = False,
        cameras: bool = True,
        flush_every: int = 256,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._data = self._file.create_group("data")
        self._inflight = self._file.create_group("inflight")
        self.num_envs = num_envs
        self.export_all = export_all
        self.cameras = cameras
        self.flush_every = flush_every
        self.exported_successful_episode_count = 0
        self.exported_episode_count = 0
        self._total_frames = 0
        self._scalars: list[dict[str, list]] = [self._empty() for _ in range(num_envs)]
        self._images: list[dict[str, list[np.ndarray]]] = [
            {k: [] for k in IMAGE_KEYS} for _ in range(num_envs)
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
        }

    # -- episode lifecycle --------------------------------------------------

    def begin(self, env_index: int, initial_state: dict[str, np.ndarray]) -> None:
        """Start buffering env ``env_index``'s next episode from ``initial_state``."""
        self._drop_inflight(env_index)
        self._scalars[env_index] = self._empty()
        self._images[env_index] = {k: [] for k in IMAGE_KEYS}
        self._initial[env_index] = {k: np.asarray(v) for k, v in initial_state.items()}
        if self.cameras:
            group = self._inflight.create_group(f"env_{env_index}")
            self._image_ds[env_index] = {
                key: group.create_dataset(
                    key,
                    shape=(0, 224, 224, 3),
                    maxshape=(None, 224, 224, 3),
                    dtype=np.uint8,
                    chunks=(1, 224, 224, 3),
                    compression="lzf",
                )
                for key in IMAGE_KEYS
            }
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
    ) -> None:
        s = self._scalars[env_index]
        s["joint_pos_abs"].append(np.asarray(joint_pos, dtype=np.float32).copy())
        s["actions"].append(np.asarray(action, dtype=np.float32).copy())
        s["cycle_phase"].append(np.int8(phase))
        s["cycle_brick"].append(np.int8(brick))
        if context_camera is not None:
            s["context_camera"].append(np.asarray(context_camera, dtype=np.float32).copy())
        if context_privileged is not None:
            s["context_privileged"].append(np.asarray(context_privileged, dtype=np.float32).copy())
        s["actor"].append(np.int16(actor))
        if self.cameras:
            imgs = self._images[env_index]
            imgs["wrist_rgb"].append(np.asarray(wrist, dtype=np.uint8).copy())
            imgs["overhead_rgb"].append(np.asarray(overhead, dtype=np.uint8).copy())
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
        obs.create_dataset("cycle_brick", data=np.asarray(s["cycle_brick"][:steps], dtype=np.int8))
        obs.create_dataset("clamped", data=np.asarray(s["clamped"][:steps], dtype=bool))
        obs.create_dataset("actor", data=np.asarray(s["actor"][:steps], dtype=np.int16))
        for key, names in (
            ("context_camera", CONTEXT_CAMERA_NAMES),
            ("context_privileged", CONTEXT_PRIVILEGED_NAMES),
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
