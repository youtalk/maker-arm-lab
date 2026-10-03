"""Pick points of joint rows, for lerobot_stack/reopen_mask.py (MS7g design 8, ledger G10).

lerobot_stack's venv has no maker_arm_rs, so the corpus mask asks this venv for its FK. Reads an
(N, 6) ``.npy`` of URDF joint coordinates and writes the (N, 3) pick points: FK of grasp_center
plus ``PICK_OFFSET_MEASURED`` in the tool frame, ``core.loop.commanded_pick`` batched.

Usage: ``.venv/bin/python scripts/pick_points.py IN.npy OUT.npy``. Prints exactly one terminal
token line, ``PICK_OK rows=N`` or ``PICK_FAIL ...``, and exits through ``os._exit``.
"""

from __future__ import annotations

import os
import sys
import traceback

CHUNK = 100_000  # rows per fk_batch call: bounds the Python lists the binding builds


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 2:
            raise ValueError("usage: pick_points.py IN.npy OUT.npy")
        import numpy as np
        import torch

        from maker_arm_lab.core.global_ik import GlobalIK
        from maker_arm_lab.robots.frames import PICK_OFFSET_MEASURED

        q = np.load(argv[0]).reshape(-1, 6)
        ik = GlobalIK()
        offset = torch.tensor(PICK_OFFSET_MEASURED, dtype=torch.float64)
        picks = [np.zeros((0, 3))]
        for start in range(0, len(q), CHUNK):
            pos, rot = ik.fk_batch(torch.from_numpy(q[start : start + CHUNK]))
            picks.append((pos + rot @ offset).numpy())
        np.save(argv[1], np.concatenate(picks))
    except Exception as err:  # noqa: BLE001 -- every failure must become a token
        print(f"PICK_FAIL {type(err).__name__}: {err}", flush=True)
        traceback.print_exc()
        return 1
    print(f"PICK_OK rows={len(q)}", flush=True)
    return 0


if __name__ == "__main__":
    os._exit(main())
