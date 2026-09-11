"""MuJoCo backend for the LegoSort simulation (MS3).

Same scene, same oracle, same dataset schema as the Isaac Lab task, at about two orders of
magnitude more env-steps per second. ``scene`` builds the model from the vendored URDF and
``core.layout``; ``env`` is the duck-typed environment ``OracleRig``, the batch loop and the
eval loop drive; ``recorder`` writes the HDF5 shards ``lerobot_stack/convert_to_lerobot.py``
reads. Rendering goes through EGL unless the caller chose another backend first.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
