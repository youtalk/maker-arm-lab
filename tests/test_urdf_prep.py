import math
import xml.etree.ElementTree as ET

import pytest

from maker_arm_lab.assets import urdf_prep


def test_arm_joint_limits_in_kinematic_order():
    limits = urdf_prep.arm_joint_limits()
    assert len(limits) == 6
    assert limits[0] == (-2.8, 2.8)
    assert limits[2] == (0.0, 3.14)
    assert limits[5] == (-2.7, 2.7)


def test_gripper_travel():
    assert urdf_prep.gripper_travel() == pytest.approx(0.0524125)


def test_write_zup_urdf_adds_mount_and_absolute_meshes(tmp_path):
    dst = urdf_prep.write_zup_urdf(urdf_prep.VENDORED_URDF, tmp_path / "robot_zup.urdf")
    root = ET.parse(dst).getroot()
    links = [link.get("name") for link in root.findall("link")]
    assert links[0] == "mount_link" and len(links) == 11
    joints = {j.get("name"): j for j in root.findall("joint")}
    mount = joints["mount_joint"]
    assert mount.get("type") == "fixed"
    assert mount.find("parent").get("link") == "mount_link"
    assert mount.find("child").get("link") == "base_link"
    rpy = [float(v) for v in mount.find("origin").get("rpy").split()]
    assert rpy == pytest.approx([math.pi / 2, 0.0, 0.0])
    # The arm chain is untouched.
    assert joints["link_002_joint"].find("origin").get("rpy") == "-1.5707963 0 0"
    for mesh in root.iter("mesh"):
        assert mesh.get("filename").startswith("/"), mesh.get("filename")
    # And the vendored file itself was not modified.
    assert "mount_link" not in urdf_prep.VENDORED_URDF.read_text()


def test_write_zup_urdf_custom_yaw(tmp_path):
    dst = urdf_prep.write_zup_urdf(
        urdf_prep.VENDORED_URDF, tmp_path / "r.urdf", mount_rpy=(math.pi / 2, 0.0, math.pi)
    )
    root = ET.parse(dst).getroot()
    mount = {j.get("name"): j for j in root.findall("joint")}["mount_joint"]
    assert mount.find("origin").get("rpy").split()[2] == repr(math.pi)
