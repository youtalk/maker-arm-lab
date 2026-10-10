"""MS5 design section 4.2: the perception worker's wire format, numpy and pyzmq only."""

import ast
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import zmq

from maker_arm_lab.core.parts import BASE_CLASSES

PROTOCOL = Path(__file__).resolve().parents[1] / "perception_stack" / "perception_protocol.py"
spec = importlib.util.spec_from_file_location("perception_protocol_under_test", PROTOCOL)
proto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proto)

GOOD = {
    "centre": [0.2, 0.0, 0.006],
    "quat": [1.0, 0.0, 0.0, 0.0],
    "size": [0.0318, 0.0158, 0.0096],
    "cls": "wide",
    "score": 0.9,
}


def test_the_classes_are_the_catalogs():
    assert proto.PART_CLASSES == BASE_CLASSES


def test_a_message_round_trips_with_its_arrays():
    ctx = zmq.Context.instance()
    a, b = ctx.socket(zmq.PAIR), ctx.socket(zmq.PAIR)
    a.bind("inproc://perception-protocol-test")
    b.connect("inproc://perception-protocol-test")
    camera = proto.camera_header(
        (1.0, 0, 2.0, 0, 1.0, 3.0, 0, 0, 1), (4, 3), (0, 0, 1), (1, 0, 0, 0)
    )
    depth = np.arange(12, dtype=np.uint16).reshape(3, 4)
    jpeg = np.frombuffer(b"\xff\xd8\xff\xd9", dtype=np.uint8)
    proto.send_message(a, {"type": "parts", "camera": camera}, {"depth": depth, "rgb_jpeg": jpeg})
    header, arrays = proto.recv_message(b)
    a.close(0)
    b.close(0)
    assert header["type"] == "parts" and header["camera"]["size"] == [4, 3]
    assert header["camera"]["pose"]["quat"] == [1.0, 0.0, 0.0, 0.0]
    assert np.array_equal(arrays["depth"], depth) and arrays["rgb_jpeg"].tobytes() == jpeg.tobytes()


def test_detections_are_validated():
    proto.validate_detection(GOOD)
    for bad in (
        {**GOOD, "cls": "blue"},
        {**GOOD, "centre": [0.2, 0.0]},
        {**GOOD, "size": [0.03, math.nan, 0.01]},
        {k: v for k, v in GOOD.items() if k != "quat"},
    ):
        with pytest.raises(ValueError):
            proto.validate_detection(bad)


def test_the_wire_imports_only_numpy_and_the_stdlib():
    body = ast.parse(PROTOCOL.read_text()).body
    names = {a.name for n in body if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in body if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= {"io", "json", "math", "numpy", "__future__"}, names
