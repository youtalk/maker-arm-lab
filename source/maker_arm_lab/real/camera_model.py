"""Pinhole cameras and the warps to the views the policy was trained on (MS5 design section
4.3 and P2). numpy only.

Conventions. A camera pose is world from camera in the ROS optical frame (x right, y down,
z forward), as ``core.cameras.OVERHEAD_CAM_ROT`` is. Pixel coordinates are continuous with the
origin at the top-left corner of the image, the ARKit and Conduit convention and
``core.overlay.project_point``'s: the centre of pixel (0, 0) is (0.5, 0.5).

The policy saw 224x224 frames from the simulator's cameras. A real frame reaches it through a
``ViewWarp``: for every output pixel, the source pixel that shows the same thing, sampled
bilinearly and averaged over a supersampling grid. The overhead warp is the desk-plane
homography from the real camera to the simulator's nominal overhead camera: exact on the
plane ``z = plane_z``, off it by the parallax of a part's height between the two camera
centres, and good for any mount yaw and a small tilt. The wrist warp is the rotation
homography from the real wrist camera to the simulated one, exact at every depth when the two
share a centre. A view the source frame does not cover stays black; ``coverage`` says how
much of it the frame covers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from maker_arm_lab.core import cameras

SIM_SIZE = cameras.IMAGE_WIDTH


@dataclass(frozen=True)
class Pinhole:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_k(cls, width: int, height: int, k: Sequence[float]) -> Pinhole:
        """From a row-major 3x3 ``k`` (``sensor_msgs/CameraInfo.k``) with zero skew."""
        k = [float(v) for v in k]
        if len(k) != 9 or k[1] != 0.0 or k[3] != 0.0 or k[6:] != [0.0, 0.0, 1.0]:
            raise ValueError(f"not a zero-skew pinhole K: {k}")
        return cls(int(width), int(height), k[0], k[4], k[2], k[5])

    @classmethod
    def sim(cls, fovy_deg: float, size: int = SIM_SIZE) -> Pinhole:
        """A MuJoCo camera's: square pixels, the principal point at the centre."""
        f = 0.5 * size / math.tan(math.radians(fovy_deg) / 2.0)
        return cls(size, size, f, f, 0.5 * size, 0.5 * size)

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def fov_deg(self) -> tuple[float, float]:
        """(horizontal, vertical) field of view, degrees."""
        return (
            math.degrees(2.0 * math.atan(0.5 * self.width / self.fx)),
            math.degrees(2.0 * math.atan(0.5 * self.height / self.fy)),
        )


def quat_matrix(quat_wxyz) -> np.ndarray:
    """The (3, 3) rotation of a (w, x, y, z) quaternion, normalized first."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_quat(rot) -> tuple[float, float, float, float]:
    """(w, x, y, z) with w >= 0 of a rotation matrix (Shepperd's method)."""
    m = np.asarray(rot, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    q = np.asarray(q) / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return tuple(float(v) for v in q)


@dataclass(frozen=True)
class Pose:
    """World from camera: ``rot`` (3, 3), whose columns are the camera axes in the world, and
    ``pos`` (3,)."""

    rot: np.ndarray
    pos: np.ndarray

    @classmethod
    def from_pos_quat(cls, pos, quat_wxyz) -> Pose:
        return cls(quat_matrix(quat_wxyz), np.asarray(pos, dtype=np.float64))

    def to_camera(self, point) -> np.ndarray:
        return self.rot.T @ (np.asarray(point, dtype=np.float64) - self.pos)

    def quat(self) -> tuple[float, float, float, float]:
        return matrix_quat(self.rot)


def project(camera: Pinhole, pose: Pose, point) -> tuple[float, float] | None:
    """Pixel (u, v) of a world point, or None at or behind the camera."""
    x, y, z = pose.to_camera(point)
    if z <= 1e-6:
        return None
    return (camera.cx + camera.fx * x / z, camera.cy + camera.fy * y / z)


def sim_overhead() -> tuple[Pinhole, Pose]:
    """The simulator's nominal overhead camera: the view a warped real frame emulates."""
    pose = Pose.from_pos_quat(cameras.OVERHEAD_CAM_POS, cameras.OVERHEAD_CAM_ROT)
    return Pinhole.sim(cameras.OVERHEAD_FOV_DEG), pose


def _plane_to_image(camera: Pinhole, pose: Pose, plane_z: float) -> np.ndarray:
    """The 3x3 map from plane coordinates (a, b, 1), world point (a, b, plane_z), to pixels."""
    columns = np.column_stack(
        [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), np.array((0.0, 0.0, plane_z)) - pose.pos]
    )
    return camera.matrix() @ pose.rot.T @ columns


def plane_homography(
    src: Pinhole, src_pose: Pose, dst: Pinhole, dst_pose: Pose, plane_z: float = 0.0
) -> np.ndarray:
    """H with ``src_pixel ~ H @ dst_pixel`` for the points of the world plane z = plane_z."""
    return _plane_to_image(src, src_pose, plane_z) @ np.linalg.inv(
        _plane_to_image(dst, dst_pose, plane_z)
    )


def rotation_homography(src: Pinhole, dst: Pinhole, rot_src_from_dst=None) -> np.ndarray:
    """H with ``src_pixel ~ H @ dst_pixel`` for two cameras that share a centre;
    ``rot_src_from_dst`` turns dst camera axes into src camera axes (None: aligned)."""
    rot = np.eye(3) if rot_src_from_dst is None else np.asarray(rot_src_from_dst, dtype=float)
    return src.matrix() @ rot @ np.linalg.inv(dst.matrix())


class ViewWarp:
    """A ``size`` x ``size`` view of a ``source_size`` (width, height) frame through a
    homography, sampled at ``supersample`` x ``supersample`` points per output pixel."""

    def __init__(self, homography, source_size, size: int = SIM_SIZE, supersample: int = 2):
        self.h = np.asarray(homography, dtype=np.float64)
        self.source_size = (int(source_size[0]), int(source_size[1]))
        self.size, self.supersample = int(size), int(supersample)
        n = self.size * self.supersample
        centres = (np.arange(n) + 0.5) / self.supersample
        u, v = np.meshgrid(centres, centres)
        src = self.h @ np.stack([u.ravel(), v.ravel(), np.ones(n * n)])
        with np.errstate(divide="ignore", invalid="ignore"):
            su, sv = src[0] / src[2], src[1] / src[2]
        su[src[2] <= 0.0], sv[src[2] <= 0.0] = -1e9, -1e9
        self._su, self._sv = su.reshape(n, n), sv.reshape(n, n)
        w, h = self.source_size
        inside = (self._su >= 0) & (self._su <= w) & (self._sv >= 0) & (self._sv <= h)
        self.coverage = float(inside.mean())
        """The share of the view the source frame covers; the rest is black."""
        self.source_span = (
            float(max(np.ptp(self._su[inside]), np.ptp(self._sv[inside]))) if inside.any() else 0.0
        )
        """The view's extent in source pixels, the larger axis."""

    def jpeg_reduce(self) -> int:
        """The largest JPEG DCT reduction (1, 2, 4 or 8) that keeps at least one source pixel
        per sample point across the view."""
        need = self.size * self.supersample
        return next((r for r in (8, 4, 2) if self.source_span / r >= need), 1)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """The (size, size, C) uint8 view of an (H, W, C) uint8 frame. ``image`` may be a
        uniformly reduced decode of the source frame (a JPEG draft): the sample points scale
        with its width."""
        h_img, w_img = image.shape[:2]
        scale = w_img / self.source_size[0]
        x = self._su * scale - 0.5
        y = self._sv * scale - 0.5
        x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        fx, fy = (x - x0)[..., None], (y - y0)[..., None]
        valid = (x >= -0.5) & (x <= w_img - 0.5) & (y >= -0.5) & (y <= h_img - 0.5)
        xa, xb = np.clip(x0, 0, w_img - 1), np.clip(x0 + 1, 0, w_img - 1)
        ya, yb = np.clip(y0, 0, h_img - 1), np.clip(y0 + 1, 0, h_img - 1)

        def at(yy, xx):
            return image[yy, xx].astype(np.float32)

        top = at(ya, xa) * (1.0 - fx) + at(ya, xb) * fx
        bottom = at(yb, xa) * (1.0 - fx) + at(yb, xb) * fx
        out = top * (1.0 - fy) + bottom * fy
        out[~valid] = 0.0
        s, n = self.supersample, self.size
        out = out.reshape(n, s, n, s, -1).mean(axis=(1, 3))
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def overhead_warp(camera: Pinhole, pose: Pose, plane_z: float = 0.0) -> ViewWarp:
    """The real overhead frame to the simulator's nominal overhead view, on the desk plane."""
    sim, sim_pose = sim_overhead()
    h = plane_homography(camera, pose, sim, sim_pose, plane_z)
    return ViewWarp(h, (camera.width, camera.height))


def wrist_warp(camera: Pinhole, rot_real_from_sim=None) -> ViewWarp:
    """The real wrist frame to the simulated wrist view (``core.cameras.WRIST_FOV_DEG``)."""
    h = rotation_homography(camera, Pinhole.sim(cameras.WRIST_FOV_DEG), rot_real_from_sim)
    return ViewWarp(h, (camera.width, camera.height))
