"""Per-step state/action trace for an eval rollout.

Kit-free, like core.gate, so the buffering has CPU tests.

Exists to answer a question the gate's counters cannot: WHERE in state space
does a failing rollout go? Open-loop agreement with the oracle on the frames a
policy was trained on says nothing about where its closed-loop rollout drifts,
and testing that means comparing the states a rollout actually visits against
the states the dataset contains. That needs the trajectory, not a summary of it.

Rows are appended only for envs still running: ManagerBasedRLEnv auto-resets a
terminated env and the loop keeps stepping until all envs are done, so a
finished env would otherwise append frames from an unrelated episode.
"""

from pathlib import Path

import numpy as np


class RolloutTrace:
    """Accumulates (step, state, raw action, clamped action) per env, in RAM.

    Deliberately unbuffered-to-disk: 5 envs x 5,400 steps x 22 floats is under
    3 MB, so a single npz per env at the end beats streaming complexity.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self._steps: list[list[int]] = [[] for _ in range(num_envs)]
        self._state: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self._raw: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self._clamped: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self._step_index = 0

    def append(
        self,
        state: np.ndarray,
        action_raw: np.ndarray,
        action_clamped: np.ndarray,
        active: np.ndarray,
    ) -> None:
        for name, array in (("state", state), ("action_raw", action_raw), ("active", active)):
            if array.shape[0] != self.num_envs:
                raise ValueError(f"{name} has batch {array.shape[0]}, expected {self.num_envs}")
        for i in range(self.num_envs):
            if not bool(active[i]):
                continue
            self._steps[i].append(self._step_index)
            self._state[i].append(np.asarray(state[i], dtype=np.float32))
            self._raw[i].append(np.asarray(action_raw[i], dtype=np.float32))
            self._clamped[i].append(np.asarray(action_clamped[i], dtype=np.float32))
        self._step_index += 1

    def save(self, directory: Path | str, stem: str) -> list[Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(self.num_envs):
            path = directory / f"{stem}_env{i}_trace.npz"
            # The widths below only shape the file an env that NEVER ran produces
            # (`_stack`); a populated env keeps whatever the caller appended. 8 is the
            # articulation's joint count (6 arm + 2 jaws), NOT the (7,) width of the
            # LeRobot `observation.state` -- an eval script feeding `append()` the
            # policy-side state instead would need this to follow.
            np.savez_compressed(
                path,
                step=np.asarray(self._steps[i], dtype=np.int32),
                state=_stack(self._state[i], 8),
                action_raw=_stack(self._raw[i], 7),
                action_clamped=_stack(self._clamped[i], 7),
            )
            paths.append(path)
        return paths


def _stack(rows: list[np.ndarray], width: int) -> np.ndarray:
    """np.stack refuses an empty list; an env that never ran still needs a file."""
    if not rows:
        return np.zeros((0, width), dtype=np.float32)
    return np.stack(rows)
