"""Pin the vendored Maker Arm URDF (maker-arm-sdk urdf/maker_arm @ b30d05a).

Every number here is read off the vendored file at vendoring time and cross-checked against
the upstream README (six revolute arm joints, symmetric sliding gripper 0..0.0524125 m per
jaw, `grasp_center` at the inner-face centroid). An upstream revision that changes any of
them must fail here, not inside Isaac Lab.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = REPO_ROOT / "assets" / "vendor" / "maker_arm_description"
URDF = VENDOR_DIR / "robot.urdf"

ARM_JOINTS = [
    "link_002_joint",
    "link_003_joint",
    "link_004_joint",
    "link_005_joint",
    "link_006_joint",
    "link_007_joint",
]
EXPECTED_LIMITS = {
    "link_002_joint": (-2.8, 2.8),
    "link_003_joint": (-3.14, 0.0),
    "link_004_joint": (0.0, 3.14),
    "link_005_joint": (-1.6, 1.73),
    "link_006_joint": (-1.57, 1.57),
    "link_007_joint": (-2.7, 2.7),
}
GRIPPER_TRAVEL = 0.0524125


@pytest.fixture(scope="module")
def root():
    assert URDF.exists(), f"{URDF} missing -- run scripts/vendor_maker_arm_description.sh"
    return ET.parse(URDF).getroot()


def joints_by_name(root):
    return {j.get("name"): j for j in root.findall("joint")}


def test_robot_name_and_link_count(root):
    assert root.get("name") == "maker_arm_gripper_v1"
    assert len(root.findall("link")) == 10


def test_arm_joint_chain_and_limits(root):
    joints = joints_by_name(root)
    parent = "base_link"
    for name in ARM_JOINTS:
        j = joints[name]
        assert j.get("type") == "revolute"
        assert j.find("parent").get("link") == parent
        child = j.find("child").get("link")
        limit = j.find("limit")
        assert (float(limit.get("lower")), float(limit.get("upper"))) == EXPECTED_LIMITS[name]
        parent = child
    assert parent == "link_007"


def test_gripper_is_symmetric_prismatic_with_mimic(root):
    joints = joints_by_name(root)
    left, right = joints["gripper_left_joint"], joints["gripper_right_joint"]
    for j in (left, right):
        assert j.get("type") == "prismatic"
        assert j.find("parent").get("link") == "link_007"
        limit = j.find("limit")
        assert float(limit.get("lower")) == 0.0
        assert float(limit.get("upper")) == GRIPPER_TRAVEL
    assert left.find("mimic") is None
    mimic = right.find("mimic")
    assert mimic is not None and mimic.get("joint") == "gripper_left_joint"
    assert float(mimic.get("multiplier", "1")) == 1.0
    lax = [float(v) for v in left.find("axis").get("xyz").split()]
    rax = [float(v) for v in right.find("axis").get("xyz").split()]
    assert lax[1] == pytest.approx(1.0, abs=1e-6)
    assert rax[1] == pytest.approx(-1.0, abs=1e-6)


def test_grasp_center_is_fixed_on_link_007_along_minus_z(root):
    j = joints_by_name(root)["grasp_center_joint"]
    assert j.get("type") == "fixed"
    assert j.find("parent").get("link") == "link_007"
    assert j.find("child").get("link") == "grasp_center"
    xyz = [float(v) for v in j.find("origin").get("xyz").split()]
    assert abs(xyz[0]) < 1e-6 and abs(xyz[1]) < 1e-6
    assert xyz[2] == pytest.approx(-0.169638704618, abs=1e-9)


def test_base_link_is_y_up_cad_frame(root):
    # The first arm joint carries the CAD's Y-up correction (rpy -pi/2 about x); the derived
    # Z-up URDF (assets/urdf_prep.py) relies on this exact value.
    rpy = [
        float(v) for v in joints_by_name(root)["link_002_joint"].find("origin").get("rpy").split()
    ]
    assert rpy == pytest.approx([-1.5707963, 0.0, 0.0])


def test_every_mesh_is_present(root):
    files = {m.get("filename") for m in root.iter("mesh")}
    # 52 unique mesh filenames, measured on the pinned b30d05a revision (not guessed): each
    # of the 52 files is referenced by both a visual and a collision element, so root.iter
    # over all <mesh> tags returns 102 elements before de-duplication by filename.
    assert len(files) == 52
    for f in files:
        assert (VENDOR_DIR / f).is_file(), f
