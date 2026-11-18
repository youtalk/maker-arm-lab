"""MS5 design section 4.2: the perception worker's geometry, numpy and the stdlib only."""

import ast
import math
import random

import mujoco
import numpy as np
import pytest

from helpers import ROOT, load_script
from maker_arm_lab.core import cameras, parts
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.core.quat import quat_mul
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.mj.stress import Stress
from maker_arm_lab.real.camera_model import Pinhole, quat_matrix

RULES = ROOT / "perception_stack" / "rules.py"
rules = load_script(RULES)
proto = load_script(ROOT / "perception_stack" / "perception_protocol.py")

CAM = Pinhole(640, 480, 415.7, 415.7, 320.0, 240.0)
K = (CAM.fx, 0.0, CAM.cx, 0.0, CAM.fy, CAM.cy, 0.0, 0.0, 1.0)
POS = np.array([0.20, 0.0, 0.80])
ROT = quat_matrix(cameras.OVERHEAD_CAM_ROT)


def rotated_rect_points(length, width, yaw, centre=(0.0, 0.0), n=15):
    a = np.linspace(-0.5, 0.5, n)
    gx, gy = np.meshgrid(a * length, a * width)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([gx.ravel() * c - gy.ravel() * s, gx.ravel() * s + gy.ravel() * c], 1) + centre


def scene(objects, base_mm=800):
    """A planar depth of a desk at z = 0 with flat-topped ``objects`` (height, (cx, cy, yaw,
    length, width)) applied in order, seen from POS looking down."""
    depth = np.full((480, 640), base_mm, np.uint16)
    ray = rules.deproject(np.full((480, 640), 1000, np.uint16), K, ROT, POS)[..., :2] - POS[:2]
    for height, (cx, cy, yaw, length, width) in objects:
        d = 0.8 - height
        dx, dy = POS[0] + ray[..., 0] * d - cx, POS[1] + ray[..., 1] * d - cy
        c, s = math.cos(yaw), math.sin(yaw)
        hit = (np.abs(dx * c + dy * s) <= length / 2) & (np.abs(dy * c - dx * s) <= width / 2)
        depth[hit] = round(d * 1000)
    return depth


def test_the_classes_are_the_protocols():
    assert rules.PART_CLASSES == proto.PART_CLASSES


def test_deprojecting_a_flat_desk_gives_z_zero():
    depth = scene([])
    depth[10, 10] = 0
    world = rules.deproject(depth, K, ROT, POS)
    assert world.shape == (480, 640, 3) and world.dtype == np.float32
    assert np.isnan(world[10, 10]).all()
    z = np.delete(world[..., 2].ravel(), 10 * 640 + 10)
    assert np.abs(z).max() < 1e-4
    # the principal pixel looks straight down at the camera's xy
    assert world[240, 320, :2] == pytest.approx(POS[:2], abs=0.002)
    # project is deproject's inverse, in the lab's half-pixel convention
    pix = rules.project(world[100:102, 200, :].astype(float), K, ROT, POS)
    assert pix == pytest.approx(np.array([[200.5, 100.5], [200.5, 101.5]]), abs=0.01)


def test_one_blob_per_part_and_none_for_the_floor():
    objects = [(0.0096, (0.15, -0.03, 0.3, 0.032, 0.016)), (0.0096, (0.25, 0.04, 0, 0.016, 0.016))]
    world = rules.deproject(scene(objects), K, ROT, POS)
    region = (0.05, 0.35, -0.1, 0.1)
    blobs = rules.blobs_above(world, 0.0, region)
    assert len(blobs) == 2
    assert rules.blobs_above(world, 0.0, (0.05, 0.2, -0.1, 0.0)) != []
    assert len(rules.blobs_above(world, 0.0, (0.05, 0.2, -0.1, 0.0))) == 1  # the region clips
    assert rules.blobs_above(rules.deproject(scene([]), K, ROT, POS), 0.0, region) == []


def test_the_rectangle_of_a_turned_1x4_gives_its_yaw_and_extents():
    cx, cy, yaw, length, width = rules.min_area_rect(rotated_rect_points(0.032, 0.008, 0.6))
    assert yaw == pytest.approx(0.6, abs=0.02) and length == pytest.approx(0.032, abs=0.001)
    assert width == pytest.approx(0.008, abs=0.001) and (cx, cy) == pytest.approx((0, 0), abs=1e-6)


@pytest.mark.parametrize("yaw", [-1.4, -0.2, 0.0, 1.2, math.pi / 2])
def test_the_yaw_stays_in_the_half_open_range(yaw):
    out = rules.min_area_rect(rotated_rect_points(0.032, 0.008, yaw, (0.1, 0.2)))[2]
    assert -math.pi / 2 < out <= math.pi / 2
    assert math.sin(out - yaw) == pytest.approx(0.0, abs=0.02)


def test_a_mask_with_no_valid_depth_still_gives_a_detection_from_the_floor():
    xy = rotated_rect_points(0.032, 0.016, 0.0)
    d = rules.part_detection(xy, np.full(len(xy), np.nan), floor_z=0.01)
    assert d["centre"][2] == pytest.approx(0.01 + rules.BRICK_HEIGHT_M / 2) and d["score"] == 0.0
    assert d["cls"] == "wide" and d["size"][2] == rules.BRICK_HEIGHT_M


def test_a_detection_carries_pose_size_and_score():
    xy = rotated_rect_points(0.032, 0.008, 0.6, (0.2, 0.05))
    heights = np.full(len(xy), 0.0096)
    heights[:10] = np.nan
    d = rules.part_detection(xy, heights, floor_z=0.0)
    assert d["quat"] == pytest.approx([math.cos(0.3), 0, 0, math.sin(0.3)], abs=0.01)
    assert d["size"] == pytest.approx([0.032, 0.008, 0.0096], abs=0.001)
    assert d["centre"][:2] == pytest.approx([0.2, 0.05], abs=0.001)
    assert d["cls"] == "narrow" and d["score"] == pytest.approx(1 - 10 / len(xy))


def test_the_classes_follow_the_catalog_rule():
    assert rules.classify((0.032, 0.008, 0.0096), 0.0) == "narrow"
    assert rules.classify((0.032, 0.016, 0.0096), 0.0) == "wide"
    assert rules.classify((0.016, 0.016, 0.0192), 0.0) == "tall"
    assert rules.classify((0.024, 0.016, 0.0192), 0.9) == "slope"


def test_a_ramp_scores_one_and_a_flat_top_zero():
    along = np.linspace(-0.016, 0.016, 20)
    assert rules.slope_score(along, np.full(20, 0.0096)) == pytest.approx(0.0, abs=1e-9)
    assert rules.slope_score(along, 0.0096 * (along + 0.016) / 0.032) == pytest.approx(
        1.0, abs=0.15
    )
    assert rules.slope_score(along, np.full(20, np.nan)) == 0.0


BANDS = [(0.05, 0.25, 0.14, 0.26)]
BOX = (0.15, 0.20, 0.3, 0.095, 0.095)  # outer: 0.09 interior plus two 5 mm walls


def box_scene(extra=()):
    ring = [(0.04, BOX), (0.0, (0.15, 0.20, 0.3, 0.085, 0.085))]
    return rules.deproject(scene(ring + list(extra)), K, ROT, POS)


def test_a_box_in_the_band_is_found_and_a_dark_part_in_the_tray_is_not():
    part = (0.0096, (0.20, 0.0, 0.0, 0.032, 0.016))
    world = box_scene([part])
    luma = np.full(world.shape[:2], 200.0)
    dx, dy = world[..., 0] - 0.15, world[..., 1] - 0.20
    luma[(np.abs(dx) < 0.05) & (np.abs(dy) < 0.05)] = 30.0  # the box is black
    luma[np.abs(world[..., 1]) < 0.03] = 20.0  # so is the part's side of the tray
    boxes = rules.box_detections(world, luma, 0.0, BANDS, (0.09, 0.09, 0.04))
    assert len(boxes) == 1 and math.dist(boxes[0][:2], (0.15, 0.20)) < 0.005
    # a bright box, a box of the wrong size and a dark part in the band are not boxes
    assert (
        rules.box_detections(world, np.full(luma.shape, 200.0), 0.0, BANDS, (0.09, 0.09, 0.04))
        == []
    )
    assert rules.box_detections(world, luma, 0.0, BANDS, (0.14, 0.14, 0.04)) == []
    small = rules.deproject(scene([(0.0096, (0.2, 0.2, 0, 0.032, 0.016))]), K, ROT, POS)
    assert rules.box_detections(small, np.zeros(luma.shape), 0.0, BANDS, (0.09, 0.09, 0.04)) == []


def test_the_rules_import_only_numpy_and_the_stdlib():
    tree = ast.parse(RULES.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module.split(".")[0])
    import sys

    assert names <= set(sys.stdlib_module_names) | {"numpy"}


def _can_render() -> bool:
    try:
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        mujoco.Renderer(model, 8, 8).close()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _can_render(), reason="no MuJoCo GL context on this host")
def test_rendered_parts_are_detected_within_the_nominal_knobs():
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    catalog = [
        parts.make_part("brick", 1, 4),
        parts.make_part("brick", 2, 3),
        parts.make_part("brick", 2, 4),
        parts.make_part("slope", 1, 2),
        parts.make_part("brick", 1, 2),
    ]
    env = MjLegoSortEnv(
        1,
        draw,
        parts=catalog,
        cameras=True,
        tray=True,
        episode_length_s=1.0,
        stress=Stress(sparse_gap=0.035),
    )
    env.reset(seed=5)
    m, d, h = env.models[0], env.datas[0], env.handles
    cam = m.camera("overhead").id
    m.cam_pos[cam] = POS
    m.cam_quat[cam] = quat_mul(cameras.OVERHEAD_CAM_ROT, cameras.ROS_TO_MUJOCO_CAM)
    m.cam_fovy[cam] = math.degrees(2 * math.atan(240 / CAM.fy))
    # The arm's home pose hides the tray from this camera. Render the arm into an unseen group.
    arm = {m.body(n).id for n in ("base_link", "link_002", "link_003", "link_004", "link_005")}
    arm |= {m.body(n).id for n in ("link_006", "link_007", "gripper_left", "gripper_right")}
    arm |= {m.body("grasp_center").id}
    for g in np.flatnonzero(np.isin(m.geom_bodyid, list(arm))):
        m.geom_group[g] = 5
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 480, 640)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0  # a mirrored part is not a mask
    r.update_scene(d, camera=cam)
    r.enable_depth_rendering()
    depth_m = r.render().copy()
    r.disable_depth_rendering()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=cam)
    seg = r.render().copy()
    r.close()
    world = rules.deproject(np.rint(depth_m * 1000).astype(np.uint16), K, ROT, POS)
    geom = seg[..., 0] * (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
    checked = 0
    for b, part in enumerate(catalog):
        w, x, y, z = env.scene[f"brick_{b}"].data.root_quat_w[0].tolist()
        if 1.0 - 2.0 * (x * x + y * y) < math.cos(math.radians(10.0)):
            continue  # a part on its side or edge is a different shape from above
        checked += 1
        # the segmentation has a few stray edge pixels: the part is its largest blob
        masked = np.where(np.isin(geom, list(h.body_geoms[b]))[..., None], world, np.nan)
        blobs = rules.blobs_above(masked, 0.0, (-1, 1, -1, 1), min_cells=20)
        assert blobs, part.name
        cells = max(blobs, key=len)
        xy, heights = masked[cells[:, 0], cells[:, 1], :2], masked[cells[:, 0], cells[:, 1], 2]
        det = rules.part_detection(xy, heights, 0.0)
        pos = env.scene[f"brick_{b}"].data.root_pos_w[0].numpy()
        truth = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        assert math.dist(det["centre"][:2], pos[:2]) < 0.003, part.name
        fold = math.pi / 2 if part.nx == part.ny else math.pi
        got = 2 * math.atan2(det["quat"][3], det["quat"][0])
        err = (got - truth + fold / 2) % fold - fold / 2
        assert abs(math.degrees(err)) < 4, part.name
        assert det["cls"] == parts.part_class(part), part.name
    assert checked >= 3
