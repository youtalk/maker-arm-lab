"""Pick MuJoCo's EGL backend before any test module imports ``mujoco``.

``maker_arm_lab.mj`` sets the same default, but a test module that imports ``mujoco`` first
gets GLFW, and on a host with no display a failed GLFW context aborts the process on the next
render probe. An explicit ``MUJOCO_GL`` (CI sets ``disable``) still wins.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
