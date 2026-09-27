"""MS5 design section 10: ``real/ros.py`` is the only module that imports ROS 2, scripts load
it only inside functions, and the CPU suite never imports it. Source is parsed, not imported."""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROS_TOP = {"rclpy", "sensor_msgs", "std_msgs", "builtin_interfaces"}
ROS_PY = REPO / "source" / "maker_arm_lab" / "real" / "ros.py"


def _files(*tops):
    for top in tops:
        for path in (REPO / top).rglob("*.py"):
            if ".venv" not in path.parts and "__pycache__" not in path.parts:
                yield path


def _imports(nodes):
    for node in nodes:
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_only_real_ros_imports_ros_2():
    offenders = [
        f"{path.relative_to(REPO)}: {name}"
        for path in _files("source", "scripts", "tests", "lerobot_stack", "perception_stack")
        if path != ROS_PY
        for name in _imports(ast.walk(ast.parse(path.read_text())))
        if name.split(".")[0] in ROS_TOP
    ]
    assert not offenders, offenders


def test_scripts_load_real_ros_only_inside_functions():
    offenders = [
        str(path.relative_to(REPO))
        for path in _files("scripts")
        if "maker_arm_lab.real.ros" in set(_imports(ast.parse(path.read_text()).body))
    ]
    assert not offenders, offenders


def test_no_test_imports_real_ros():
    offenders = [
        str(path.relative_to(REPO))
        for path in _files("tests")
        if "maker_arm_lab.real.ros" in set(_imports(ast.walk(ast.parse(path.read_text()))))
    ]
    assert not offenders, offenders


def test_the_topics_are_the_conduit_contract():
    tree = ast.parse(ROS_PY.read_text())
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("NAMESPACE", "TOPICS")
    }
    assert values["NAMESPACE"] == "/conduit"
    assert values["TOPICS"] == {
        "jpeg": "camera/wide/image_raw/compressed",
        "info": "camera/wide/camera_info",
        "depth": "camera/wide/depth/image_raw",
        "depth_info": "camera/wide/depth/camera_info",
    }
