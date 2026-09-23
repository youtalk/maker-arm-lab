"""lerobot_stack has its own venv and cannot import maker_arm_lab; keep its pinned constants
equal to the URDF's by reading the source text."""

import re
from pathlib import Path

import pytest

from maker_arm_lab.assets import urdf_prep

CONVERT = Path(__file__).resolve().parents[1] / "lerobot_stack" / "convert_to_lerobot.py"
PROTOCOL = Path(__file__).resolve().parents[1] / "lerobot_stack" / "policy_protocol.py"


def test_gripper_travel_pin_matches_urdf():
    text = CONVERT.read_text()
    value = float(re.search(r"^GRIPPER_TRAVEL = ([0-9.]+)", text, re.MULTILINE).group(1))
    assert value == urdf_prep.gripper_travel()


def test_arm_joint_names_pin_matches_urdf():
    text = CONVERT.read_text()
    for name in urdf_prep.ARM_JOINT_NAMES:
        assert f'"{name}"' in text


def test_the_converter_names_both_gripper_contracts():
    text = CONVERT.read_text()
    assert '"gripper_gap_m"' in text and 'GRIPPER_MODES = ("binary", "continuous")' in text


def test_the_converter_drops_the_corrective_phases_by_their_ids():
    from maker_arm_lab.core.oracle import PHASE_TO_INT

    text = CONVERT.read_text()
    pinned = re.search(r"^CORRECTIVE_INTS = \(([0-9, ]+)\)", text, re.MULTILINE).group(1)
    names = [n for n in PHASE_TO_INT if n.startswith(("PUSH_", "TIP_"))]
    assert [int(v) for v in pinned.split(",")] == [PHASE_TO_INT[n] for n in names]


def test_protocol_state_is_seven_wide():
    if not PROTOCOL.exists():
        pytest.skip("lerobot_stack/policy_protocol.py is ported by Task E2")
    assert '"state": ((7,), "float32")' in PROTOCOL.read_text()
