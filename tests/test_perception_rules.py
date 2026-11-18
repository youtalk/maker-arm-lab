"""MS5 design section 4.2: the perception worker's geometry, numpy and the stdlib only."""

import ast
import math
import random
import sys

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


def test_the_mask_footprint_limit_is_the_largest_base_class_part():
    base = [p for p in parts._replayable() if parts.part_class(p) in parts.BASE_CLASSES]
    longest = max(max(p.size[:2]) for p in base)
    widest = max(min(p.size[:2]) for p in base)
    assert rules.PART_MAX_FOOTPRINT_M == pytest.approx((longest, widest))


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
INTERIOR = (0.09, 0.09, 0.04)


def luma_scene(boxes, plane_z=0.04, k=K, size=(480, 640), pos=POS):
    """A bright desk with solid black boxes (cx, cy, yaw, length, width), ``plane_z`` tall, cast
    from a camera at ``pos``: it sees each box's top and the outer walls that face it."""
    h, w = size
    ray = rules.deproject(np.full((h, w), 1000, np.uint16), k, ROT, pos).astype(float) - pos
    luma = np.full((h, w), 200.0)
    for cx, cy, yaw, length, width in boxes:
        c, s = math.cos(yaw), math.sin(yaw)
        o = pos - np.array([cx, cy, 0.0])
        t0, t1 = np.full((h, w), -np.inf), np.full((h, w), np.inf)
        slabs = (
            (o[0] * c + o[1] * s, ray[..., 0] * c + ray[..., 1] * s, length / 2),
            (-o[0] * s + o[1] * c, -ray[..., 0] * s + ray[..., 1] * c, width / 2),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            for oc, rc, lo, hi in (
                *((a, b, -e, e) for a, b, e in slabs),
                (o[2], ray[..., 2], 0.0, plane_z),
            ):
                a, b = (lo - oc) / rc, (hi - oc) / rc
                t0, t1 = np.maximum(t0, np.minimum(a, b)), np.minimum(t1, np.maximum(a, b))
        luma[(t1 >= t0) & (t1 > 0)] = 30.0
    return luma


def boxes_in(luma, depth=None, interior=INTERIOR, **kw):
    world = None if depth is None else rules.deproject(depth, K, ROT, POS)
    return rules.box_detections(luma, K, ROT, POS, 0.0, BANDS, interior, world, **kw)


def test_a_black_box_with_no_depth_is_found_from_its_outline():
    box = (0.15, 0.20, 0.3, 0.095, 0.095)
    found = boxes_in(luma_scene([box]), np.zeros((480, 640), np.uint16))
    assert len(found) == 1 and math.dist(found[0][:2], box[:2]) < 0.005
    # a long box shows its heading
    long_box = (0.15, 0.20, 0.5, 0.12, 0.08)
    found = boxes_in(luma_scene([long_box]), interior=(0.11, 0.07, 0.04))
    assert math.dist(found[0][:2], long_box[:2]) < 0.005
    assert abs(math.degrees(found[0][2] - 0.5)) < 2


def test_a_box_with_a_valid_depth_rim_is_found_and_a_flat_dark_mat_is_not():
    luma = luma_scene([BOX])
    ring = [(0.04, BOX), (0.0, (0.15, 0.20, 0.3, 0.085, 0.085))]
    assert len(boxes_in(luma, scene(ring))) == 1
    assert boxes_in(luma, scene([])) == []  # dark, but nothing rises above the desk


def test_a_dark_part_in_the_tray_or_in_a_band_is_not_a_box():
    tray_part = luma_scene([(0.20, 0.0, 0.0, 0.032, 0.016)], plane_z=0.0096)
    assert boxes_in(tray_part) == []
    assert boxes_in(luma_scene([(0.2, 0.2, 0.0, 0.032, 0.016)], plane_z=0.0096)) == []


def test_the_wall_parameter_sets_the_outer_size_the_rule_wants():
    luma = luma_scene([(0.15, 0.20, 0.0, 0.078, 0.078)])
    assert len(boxes_in(luma, interior=(0.075, 0.075, 0.04), wall_m=0.0015)) == 1
    small = luma_scene([(0.15, 0.20, 0.0, 0.060, 0.060)])
    assert boxes_in(small, interior=(0.075, 0.075, 0.04), wall_m=0.005) == []


HI_K = (1450.0, 0.0, 960.0, 0.0, 1450.0, 720.0, 0.0, 0.0, 1.0)  # the iPhone's 1920 x 1440
HI_POS = np.array([0.16, 0.0, 0.80])


@pytest.mark.parametrize(
    "box",
    [
        (0.13, 0.18, 0.0),
        (0.13, -0.18, 0.0),
        (0.13, 0.22, 0.0),
        (0.13, -0.22, 0.0),
        (0.05, 0.22, 0.0),
        (0.21, -0.22, math.radians(30.0)),
        (0.05, 0.18, math.radians(-40.0)),
    ],
)
def test_a_tall_box_is_centred_on_its_walls_not_on_its_parallax(box):
    """A 0.075 m box seen from 0.8 m shows its near outer wall, and its far rim projects onto
    the desk past the band: the box must still read within 3 mm and 2 degrees."""
    interior, wall = (0.075, 0.075, 0.075), 0.0015
    outer = interior[0] + 2 * wall
    luma = luma_scene([(*box, outer, outer)], interior[2], HI_K, (1440, 1920), HI_POS)
    half = math.hypot(interior[0], interior[1]) / 2  # the worker's band margin
    bands = [(0.05 - half, 0.21 + half, 0.18 - half, 0.22 + half)]
    bands.append((bands[0][0], bands[0][1], -bands[0][3], -bands[0][2]))
    found = rules.box_detections(luma, HI_K, ROT, HI_POS, 0.0, bands, interior, wall_m=wall)
    assert len(found) == 1
    assert math.dist(found[0][:2], box[:2]) < 0.003
    assert abs(math.degrees((found[0][2] - box[2] + math.pi / 4) % (math.pi / 2) - math.pi / 4)) < 2


def test_a_mask_with_no_valid_depth_is_placed_by_the_floor_plane_ray():
    truth = (0.17, -0.02, 0.4, 0.032, 0.016)
    depth = scene([(rules.BRICK_HEIGHT_M, truth)])
    rows, cols = np.nonzero(depth != 800)
    pixels = np.stack([rows, cols], axis=-1)
    xy = rules.mask_xy(pixels, np.zeros(len(pixels)), K, ROT, POS, rules.BRICK_HEIGHT_M)
    det = rules.part_detection(xy, np.full(len(xy), np.nan), 0.0)
    assert det["centre"][:2] == pytest.approx(truth[:2], abs=0.002)
    assert det["centre"][2] == pytest.approx(rules.BRICK_HEIGHT_M / 2)
    # with valid depth the same pixels deproject directly
    seen = rules.mask_xy(pixels, depth[rows, cols], K, ROT, POS, 0.0)
    assert seen.mean(axis=0) == pytest.approx(truth[:2], abs=0.002)


def test_the_rules_import_only_numpy_and_the_stdlib():
    tree = ast.parse(RULES.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module.split(".")[0])
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
