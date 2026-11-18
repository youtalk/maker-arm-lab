"""AprilTag 36h11 boards on the iPad: the overhead calibration, the screen pose and the base
registration (MS5 sensing). numpy and the stdlib, plus OpenCV (the ``real`` group), which
this module imports only inside its functions.

Frames. The screen frame is ``layouts.Screen``'s: the origin at the screen centre, x along +u,
y along -v, z out of the screen. A tag's corners are the outer edge of its black border, in
OpenCV's ArUco order (top-left, top-right, bottom-right, bottom-left as the tag is shown).
``screen_in_camera`` returns the screen frame in the camera frame, a ``Pose`` whose ``rot`` and
``pos`` take screen points to camera points; ``solve_pose`` and ``calibrate`` return a camera
pose in the world as everywhere else (``Pose`` = world from camera).

Pixels. The lab's convention puts the centre of pixel (0, 0) at (0.5, 0.5); OpenCV's puts it at
(0, 0). ``detect_tags`` and ``solve_pose`` are the adapters: they add and remove the half pixel,
and nothing else here does.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from maker_arm_lab.real.camera_model import Pinhole, Pose, project
from maker_arm_lab.real.layouts import IPAD_AIR_11, Screen

TAG_CELLS = 8
"""A 36h11 tag is 6 data cells and a 1-cell black border on each side."""
MIN_TAGS = 4
"""Tags a placement must show for ``scripts/real_calibrate_overhead.py``."""


@dataclass(frozen=True)
class TagBoard:
    cell_px: int
    tags: dict[int, tuple[int, int]]
    """Each tag's top-left pixel on the screen (u, v), lab convention."""
    screen: Screen

    @property
    def size_m(self) -> float:
        return TAG_CELLS * self.cell_px * self.screen.mm_per_px / 1000.0


def _grid(cell_px: int, cols: int, rows: int, gap_px: int, screen: Screen) -> TagBoard:
    side = TAG_CELLS * cell_px
    u0 = (screen.width_px - cols * side - (cols - 1) * gap_px) // 2
    v0 = (screen.height_px - rows * side - (rows - 1) * gap_px) // 2
    tags = {
        r * cols + c: (u0 + c * (side + gap_px), v0 + r * (side + gap_px))
        for r in range(rows)
        for c in range(cols)
    }
    return TagBoard(cell_px, tags, screen)


IPAD_BOARD = _grid(64, 3, 2, 128, IPAD_AIR_11)
"""Six 49.26 mm tags in a 3x2 grid, 128 px (12.2 mm) apart."""


def _dictionary(cv2):
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def board_png(board: TagBoard) -> np.ndarray:
    """The (height, width, 3) uint8 image to show on the screen, one to one."""
    import cv2

    s = board.screen
    img = np.full((s.height_px, s.width_px, 3), 255, np.uint8)
    side = TAG_CELLS * board.cell_px
    for tag_id, (u, v) in board.tags.items():
        tag = cv2.aruco.generateImageMarker(_dictionary(cv2), tag_id, side, borderBits=1)
        img[v : v + side, u : u + side] = tag[..., None]
    return img


def tag_corners_screen(board: TagBoard, tag_id: int) -> np.ndarray:
    """(4, 2) corners of a tag in the screen frame, m, in ArUco order."""
    u, v = board.tags[tag_id]
    side = TAG_CELLS * board.cell_px
    px = ((u, v), (u + side, v), (u + side, v + side), (u, v + side))
    return np.array([board.screen.from_px(a, b) for a, b in px])


def detect_tags(rgb: np.ndarray) -> dict[int, np.ndarray]:
    """{id: (4, 2) corners} in the lab's pixel convention, of an (H, W, 3) uint8 RGB frame."""
    import cv2

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(_dictionary(cv2), params)
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) + 0.5 for i, c in zip(ids.ravel(), corners)}


def solve_pose(
    camera: Pinhole, image_points: np.ndarray, world_points: np.ndarray
) -> tuple[Pose, float]:
    """The camera in the frame of ``world_points`` (N, 3), from their pixels (N, 2) in the lab
    convention, and the RMS reprojection error, px. Planar points use IPPE."""
    import cv2

    img = np.asarray(image_points, np.float64).reshape(-1, 1, 2) - 0.5
    obj = np.asarray(world_points, np.float64).reshape(-1, 1, 3)
    k = np.array(camera.ros_k()).reshape(3, 3)
    planar = bool(np.allclose(obj[:, 0, 2], obj[0, 0, 2]))
    flag = cv2.SOLVEPNP_IPPE if planar else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=flag)
    if not ok:
        raise ValueError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, k, None, rvec, tvec)
    proj = cv2.projectPoints(obj, rvec, tvec, k, None)[0]
    rms = float(np.sqrt(((proj - img) ** 2).sum(axis=-1).mean()))
    r = cv2.Rodrigues(rvec)[0]
    return Pose(r.T, -r.T @ tvec.ravel()), rms


def screen_in_camera(
    camera: Pinhole, corners: dict[int, np.ndarray], board: TagBoard
) -> tuple[Pose, float]:
    """The screen frame in the camera frame (camera from screen), and the RMS error, px."""
    ids = sorted(corners)
    img = np.concatenate([corners[i] for i in ids])
    xyz = np.concatenate(
        [np.column_stack([tag_corners_screen(board, i), np.zeros(4)]) for i in ids]
    )
    inv, rms = solve_pose(camera, img, xyz)
    return Pose(inv.rot.T, -inv.rot.T @ inv.pos), rms


def split_ids(ids: Sequence[int]) -> tuple[list[int], list[int]]:
    """The calibration ids (even positions) and the check ids (odd) of the sorted ``ids``."""
    s = sorted(ids)
    return s[0::2], s[1::2]


def mean_pose(poses: Sequence[Pose]) -> Pose:
    """The mean position and the rotation nearest the summed rotation matrices."""
    u, _, vt = np.linalg.svd(sum(p.rot for p in poses))
    rot = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    return Pose(rot, np.mean([p.pos for p in poses], axis=0))


def _xyz(board: TagBoard, tag_id: int) -> np.ndarray:
    return np.column_stack([tag_corners_screen(board, tag_id), np.zeros(4)])


def _on_plane(camera: Pinhole, pose: Pose, px: np.ndarray, z: float) -> np.ndarray:
    """The world points at height ``z`` that the pixels (N, 2) of a camera at ``pose`` see."""
    ray = np.column_stack([(px[:, 0] - camera.cx) / camera.fx, (px[:, 1] - camera.cy) / camera.fy])
    world = np.column_stack([ray, np.ones(len(px))]) @ pose.rot.T
    return pose.pos + world * ((z - pose.pos[2]) / world[:, 2:3])


@dataclass(frozen=True)
class CalibResult:
    pose: Pose
    plane_rms_mm: float
    check_rms_px: float
    check_rms_mm: float
    report: dict = field(default_factory=dict)
    """JSON-ready; ``check_pairs`` is [[true xyz, measured xyz]] of the check corners in the
    base frame, the rows a caller can project into any view."""


def calibrate(
    camera: Pinhole, placements: list[dict[int, np.ndarray]], board: TagBoard, reference: Pose
) -> CalibResult:
    """The camera in the base frame from iPad placements. ``placements[0]`` is the iPad on its
    taped spot, whose screen frame in the base frame is ``reference``; the rest lie flat
    anywhere in view. One plane fits every tag corner: its normal is the base z axis and its
    distance from the camera sets the height above ``reference``'s z. Base x and y follow
    placement 0's screen axes, and its screen centre sits at ``reference``'s position."""
    first_ids = sorted(i for i in placements[0] if i in board.tags)
    if len(first_ids) < 2:
        raise ValueError("placement 0 needs two tags: one to solve and one to check")
    use_ids, check_ids = split_ids(first_ids)
    poses, rms_px, points = [], [], []
    for n, corners in enumerate(placements):
        use = {i: corners[i] for i in (use_ids if n == 0 else sorted(corners)) if i in board.tags}
        pose, rms = screen_in_camera(camera, use, board)
        poses.append(pose)
        rms_px.append(rms)
        points += [(pose.rot @ _xyz(board, i).T).T + pose.pos for i in use]
    pts = np.concatenate(points)
    centre = pts.mean(axis=0)
    normal = np.linalg.svd(pts - centre)[2][2]
    up = normal if normal[2] < 0 else -normal  # base z in the camera frame
    plane_rms = float(np.sqrt((((pts - centre) @ up) ** 2).mean()) * 1000.0)
    r0, t0 = poses[0].rot, poses[0].pos
    e1 = r0[:, 0] - (r0[:, 0] @ up) * up
    e1 /= np.linalg.norm(e1)
    turn = np.cross(up, e1)
    phi = np.arctan2(reference.rot[1, 0], reference.rot[0, 0])  # screen x's angle from base x
    bx = e1 * np.cos(phi) - turn * np.sin(phi)
    rot = np.column_stack([bx, np.cross(up, bx), up]).T  # camera axes in the base frame
    on_plane = t0 - ((t0 - centre) @ up) * up
    pose = Pose(rot, reference.pos - rot @ on_plane)
    true, seen = [], []
    for i in check_ids:
        true.append((reference.rot @ _xyz(board, i).T).T + reference.pos)
        seen.append(placements[0][i])
    true, seen = np.concatenate(true), np.concatenate(seen)
    back = [project(camera, pose, p) for p in true]
    check_px = float(np.sqrt(((np.array(back) - seen) ** 2).sum(axis=1).mean()))
    measured = _on_plane(camera, pose, seen, float(reference.pos[2]))
    check_mm = float(np.sqrt(((measured - true) ** 2).sum(axis=1).mean()) * 1000.0)
    report = {
        "placements": len(placements),
        "screen_rms_px": rms_px,
        "camera_height_m": float(pose.pos[2] - reference.pos[2]),
        "calibration_ids": use_ids,
        "check_ids": check_ids,
        "check_pairs": [[a.tolist(), b.tolist()] for a, b in zip(true, measured)],
    }
    return CalibResult(pose, plane_rms, check_px, check_mm, report)


def _kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t with ``b ~ a @ R.T + t``, no scale."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    r = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return r, cb - r @ ca


def register_base(
    q: np.ndarray, tag_cam: np.ndarray, ik, guess: Pose, iters: int = 50
) -> tuple[Pose, np.ndarray, float, np.ndarray]:
    """The camera in the base frame and the tag offset in the ``grasp_center`` frame, from N
    joint vectors (N, 6) and the gripper tag's centre in the camera frame at each (N, 3).
    Two closed-form steps alternate: with the offset fixed, Kabsch aligns the camera points
    with ``fk(q) + R_fk @ offset``; with the pose fixed, least squares gives the offset. It
    stops when the RMS changes by under 1e-6 m. Returns (pose, offset, rms_mm, per-pose mm);
    the pose starts at ``guess``, which only has to be near enough for the alternation."""
    q = np.asarray(q, np.float64).reshape(-1, 6)
    cam = np.asarray(tag_cam, np.float64).reshape(-1, 3)
    fk = [ik.fk(qi) for qi in q]
    p = np.array([a.numpy() for a, _ in fk])
    r = np.array([b.numpy() for _, b in fk])
    rot, pos = guess.rot, guess.pos
    prev = np.inf
    for _ in range(iters):
        rhs = (cam @ rot.T + pos - p).reshape(-1)  # R_fk @ off = rot @ cam + pos - p
        off = np.linalg.lstsq(r.reshape(-1, 3), rhs, rcond=None)[0]
        rot, pos = _kabsch(cam, p + r @ off)
        err = np.linalg.norm(cam @ rot.T + pos - (p + r @ off), axis=1)
        now = float(np.sqrt((err**2).mean()))
        done = abs(prev - now) < 1e-6
        prev = now
        if done:
            break
    return Pose(rot, pos), off, now * 1000.0, err * 1000.0


class FrameTap:
    """A Conduit subscriber's callbacks: the first camera info and every JPEG since ``start``.
    No ROS import: ``real.ros.ConduitSubscriber`` calls these on its thread, tests call them
    directly."""

    def __init__(self):
        self.camera: Pinhole | None = None
        self.frames: list = []
        self._lock = threading.Lock()
        self._on = False

    def on_info(self, info) -> None:
        if self.camera is None:
            self.camera = Pinhole.from_ros_k(info.width, info.height, info.k)

    def on_jpeg(self, frame) -> None:
        with self._lock:
            if self._on:
                self.frames.append(frame)

    def start(self) -> None:
        with self._lock:
            self.frames, self._on = [], True

    def stop(self) -> list:
        with self._lock:
            self._on = False
            return self.frames
