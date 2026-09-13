"""Quaternion helpers, torch-only, in the (w, x, y, z) order Isaac Lab and MuJoCo share.

``quat_apply`` is ``isaaclab.utils.math.quat_apply`` re-implemented so that the oracle runtime
imports no isaaclab. The stub in ``tests/test_oracle_runtime_frames.py`` carried this same
formula for years; it is now the one copy every backend uses.
"""

import math

import torch


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` (..., 3) by ``quat`` (..., 4) in wxyz order, broadcasting on the batch."""
    q = quat.reshape(-1, 4)
    v = vec.reshape(-1, 3)
    if q.shape[0] == 1 and v.shape[0] > 1:
        q = q.expand(v.shape[0], 4)
    if v.shape[0] == 1 and q.shape[0] > 1:
        v = v.expand(q.shape[0], 3)
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return (v + w * t + torch.linalg.cross(xyz, t, dim=-1)).reshape(vec.shape)


def quat_from_rpy_deg(rpy_deg) -> tuple[float, float, float, float]:
    """(w, x, y, z) from roll/pitch/yaw in degrees, ZYX convention (URDF-style fixed axes)."""
    r, p, y = (math.radians(v) / 2.0 for v in rpy_deg)
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def quat_mul(a, b) -> tuple[float, float, float, float]:
    """Hamilton product ``a * b`` of two (w, x, y, z) tuples: rotate by ``b``, then by ``a``."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def compose_offset_rot(rot, drot_deg) -> tuple[float, float, float, float]:
    """Left-multiply a small rpy jitter (degrees) onto a camera offset quaternion."""
    return quat_mul(quat_from_rpy_deg(drot_deg), rot)


def rpy_from_quat(quat) -> tuple[float, float, float]:
    """Roll, pitch, yaw in radians (ZYX, the inverse of ``quat_from_rpy_deg``) from (w, x, y, z)."""
    w, x, y, z = quat
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw
