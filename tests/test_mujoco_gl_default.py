"""The suite picks EGL when the caller names no MuJoCo GL backend.

``test_mj_heap_env.py`` imports ``mujoco`` before ``maker_arm_lab.mj`` sets its EGL default, so
without ``tests/conftest.py`` MuJoCo falls back to GLFW. On a host with no display (the Jetson
Thor) the first GLFW context fails with a ``FatalError``, and the second ``_can_render`` probe
then aborts the whole process in a C++ static-initialization guard instead of skipping.
"""

import ctypes.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(ctypes.util.find_library("EGL") is None, reason="no libEGL on this host")
def test_collecting_the_render_probes_without_a_backend_does_not_abort():
    env = {k: v for k, v in os.environ.items() if k not in ("MUJOCO_GL", "DISPLAY")}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_mj_heap_env.py",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
