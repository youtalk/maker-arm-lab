"""Target marker overlay: project a world point through a camera and stamp it on an image.

Expects `project_point` to receive the camera position and rotation from MjData.cam_xpos and
MjData.cam_xmat, read after `mj_camlight`. These fields propagate the camera state through the scene
to the rendering pipeline.
"""

from __future__ import annotations

import math

import numpy as np

MARKER_RADIUS_PX = 5
MARKER_RING_PX = 1


def project_point(cam_pos, cam_mat, fovy_deg: float, width: int, height: int, point):
    """Pixel (u, v) of a world point through a MuJoCo camera, or None behind the camera.

    MuJoCo cameras look along their -z with +y up; ``fovy_deg`` is the vertical field of view.
    """
    rotation = np.asarray(cam_mat, dtype=np.float64).reshape(3, 3)
    local = rotation.T @ (np.asarray(point, dtype=np.float64) - np.asarray(cam_pos))
    if local[2] >= -1e-6:
        return None
    focal = 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)
    u = 0.5 * width + focal * local[0] / -local[2]
    v = 0.5 * height - focal * local[1] / -local[2]
    return (float(u), float(v))


def stamp_marker(image: np.ndarray, u: float, v: float) -> None:
    """Draw a white disc with a black ring at (u, v), in place, clipped to the image."""
    h, w = image.shape[:2]
    cu, cv = round(u), round(v)
    r = MARKER_RADIUS_PX + MARKER_RING_PX
    if cu < -r or cv < -r or cu >= w + r or cv >= h + r:
        return
    y0, y1 = max(0, cv - r), min(h, cv + r + 1)
    x0, x1 = max(0, cu - r), min(w, cu + r + 1)
    ys, xs = np.ogrid[y0:y1, x0:x1]
    d2 = (ys - cv) ** 2 + (xs - cu) ** 2
    view = image[y0:y1, x0:x1]
    view[d2 <= r * r] = 0
    view[d2 <= MARKER_RADIUS_PX * MARKER_RADIUS_PX] = 255
