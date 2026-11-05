"""lerobot_stack has its own venv and cannot import maker_arm_lab; keep its pinned constants
equal to the URDF's by reading the source text."""

import re
from pathlib import Path

from maker_arm_lab.assets import urdf_prep

CONVERT = Path(__file__).resolve().parents[1] / "lerobot_stack" / "convert_to_lerobot.py"
PROTOCOL = Path(__file__).resolve().parents[1] / "lerobot_stack" / "policy_protocol.py"


def test_the_pi05_jaw_latch_reads_the_target_width_and_the_full_gap():
    from maker_arm_lab.mj import context_names as cn
    from maker_arm_lab.robots.frames import JAW_FULL_GAP

    text = (CONVERT.parent / "pi05_server.py").read_text()
    names = cn.CONTEXT_CAMERA_NAMES + cn.CONTEXT_BOX_NAMES + cn.CONTEXT_TARGET_NAMES
    assert len(names) == cn.CONTEXT_DIMS["camera_box_target"]  # the served 16-wide context
    assert f"\nTARGET_WIDTH = {7 + names.index('target_width')}\n" in text
    assert "\nJAW_FULL_GAP = 2 * 0.0524125\n" in text and JAW_FULL_GAP == 2 * 0.0524125
    # the close rule of train_pi05.py --close_weight (MS7k L1) reads the same full gap
    assert "\nJAW_FULL_GAP = 2 * 0.0524125  #" in (CONVERT.parent / "train_pi05.py").read_text()


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
    assert '"state": ((7,), "float32")' in PROTOCOL.read_text()


def test_the_part_classifier_names_the_ms7f_categories_in_order():
    from maker_arm_lab.core.parts import PART_CATEGORIES

    text = (CONVERT.parent / "train_part_classifier.py").read_text()
    pinned = re.search(r"^CATEGORIES = \(([^)]*)\)", text, re.MULTILINE).group(1)
    assert tuple(v.strip().strip('"') for v in pinned.split(",")) == PART_CATEGORIES


def test_the_corpus_mask_names_lift_and_reopen_by_their_ids():
    from maker_arm_lab.core.oracle import PHASE_TO_INT

    text = (CONVERT.parent / "reopen_mask.py").read_text()
    lift, reopen = re.search(r"^LIFT_INT, REOPEN_INT = (\d+), (\d+)", text, re.MULTILINE).groups()
    assert (int(lift), int(reopen)) == (PHASE_TO_INT["LIFT"], PHASE_TO_INT["REOPEN"])


def test_the_served_context_roll_index_is_the_camera_roll():
    from maker_arm_lab.mj.context_names import CONTEXT_CAMERA_NAMES

    text = (CONVERT.parent / "policy_server.py").read_text()
    index = int(re.search(r"^CONTEXT_ROLL = (\d+)", text, re.MULTILINE).group(1))
    assert CONTEXT_CAMERA_NAMES.index("overhead_cam_roll") == index


def test_the_own_frames_labels_copy_the_subtasks_and_the_census_rules():
    import ast

    from maker_arm_lab.core import hold, loop, oracle

    text = CONVERT.read_text()

    def pinned(name):
        found = re.search(rf"^{name} = (.+)$", text, re.MULTILINE).group(1)
        return ast.literal_eval(found.split("  #")[0])

    assert pinned("SUBTASKS") == hold.SUBTASKS
    assert pinned("LIFT_MIN_RISE") == oracle.LIFT_MIN_RISE
    assert pinned("CENSUS_MIN_HELD_STEPS") == loop.CENSUS_MIN_HELD_STEPS
    failing, boxes = pinned("FAILING_PLACES"), pinned("BOX_PLACES")
    assert sorted(failing + boxes) == sorted(loop.RELEASE_PLACES)
    assert set(boxes) == {"right_box", "wrong_box"}


def test_run_a_s_phrases_follow_the_subtask_ids():
    from maker_arm_lab.core.hold import SUBTASKS

    text = (CONVERT.parent / "train_pi05.py").read_text()
    pinned = re.search(r"^PHRASES = \(([^)]*)\)", text, re.MULTILINE).group(1)
    phrases = tuple(v.strip().strip('"') for v in pinned.split(","))
    meaning = {  # MS7i design rev1 5.2: each subtask's phrase
        "reach": "reach the marked part",
        "grasp": "close on the part",
        "carry": "carry it to its box",
        "release": "release it",
    }
    assert phrases == tuple(meaning[name] for name in SUBTASKS)
