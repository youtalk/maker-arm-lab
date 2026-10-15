"""The MS7i subtask id of each frame of a teacher root, for lerobot_stack/prompt_codes.py.

lerobot_stack's venv has no maker_arm_rs and cannot import core, so prompt_codes.py hands this
venv a root's frames as an ``.npz`` in frame order: ``q`` (N, 6) the URDF joints
(observation.state[:, :6]), ``opening`` (N,) the left jaw's opening fraction
(observation.state[:, 6]), ``cmd`` (N,) the action's jaw column (the continuous gap command, m),
``width`` (N,) the served target width (environment_state[14]) and ``episode`` (N,) the episode
index, each episode's frames contiguous. One teacher episode is one cycle (MS7i design rev3 5.3),
so ``core.hold.subtask_ids`` runs over each episode with one cycle ordinal: the gap is
``JAW_FULL_GAP * opening``, the previous frame's served command ``commanded_gap(cmd,
"continuous")`` (none on an episode's first frame) and the pick point z ``pick_points.py``'s FK.

Usage: ``.venv/bin/python scripts/subtask_ids.py IN.npz OUT.npy`` writes the (N,) int8 ids.
Prints exactly one terminal token line, ``SUBTASK_OK rows=N counts=a,b,c,d`` (frames a subtask,
in ``SUBTASKS`` order) or ``SUBTASK_FAIL ...``, and exits through ``os._exit``.
"""

from __future__ import annotations

import os
import sys
import traceback


def subtask_ids(frames) -> list[int]:
    """The id of each frame of ``frames`` (the ``.npz`` arrays), episode by episode."""
    import numpy as np

    from maker_arm_lab.core import hold
    from maker_arm_lab.robots.frames import JAW_FULL_GAP
    from pick_points import pick_points

    episode = np.asarray(frames["episode"])
    starts = np.r_[0, np.flatnonzero(np.diff(episode)) + 1] if len(episode) else np.zeros(0, int)
    if len(np.unique(episode)) != len(starts):
        raise ValueError("an episode's frames are not contiguous")
    z = pick_points(np.asarray(frames["q"], dtype=np.float64))[:, 2].tolist()
    gap = (JAW_FULL_GAP * np.asarray(frames["opening"], dtype=np.float64)).tolist()
    served = [hold.commanded_gap(float(c), "continuous") for c in frames["cmd"]]
    width = np.asarray(frames["width"], dtype=np.float64).tolist()
    ids = []
    for s, e in zip(starts, [*starts[1:], len(episode)]):
        prev = [None, *served[s : e - 1]]
        ids += hold.subtask_ids(gap[s:e], prev, z[s:e], width[s:e], [1] * (e - s))
    return ids


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 2:
            raise ValueError("usage: subtask_ids.py IN.npz OUT.npy")
        import numpy as np

        from maker_arm_lab.core.hold import SUBTASKS

        ids = np.asarray(subtask_ids(np.load(argv[0])), dtype=np.int8)
        np.save(argv[1], ids)
        counts = np.bincount(ids, minlength=len(SUBTASKS))
    except Exception as err:  # noqa: BLE001 -- every failure must become a token
        print(f"SUBTASK_FAIL {type(err).__name__}: {err}", flush=True)
        traceback.print_exc()
        return 1
    print(f"SUBTASK_OK rows={len(ids)} counts={','.join(map(str, counts))}", flush=True)
    return 0


if __name__ == "__main__":
    os._exit(main())
