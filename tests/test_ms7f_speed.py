"""MS7f track S (design section 8): the teacher's pace and the serving waits. With the default
values the walk and the waits are the MS7e ones."""

import inspect
import sys
from pathlib import Path

import pytest

from maker_arm_lab.core import trajectory
from maker_arm_lab.tasks.manager_based.lego_sort import oracle_runtime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mj_eval_heap

SCRIPT = Path(mj_eval_heap.__file__).read_text()


def test_the_default_pace_is_the_ms7e_walk():
    assert mj_eval_heap.parse_args([]).transport_speed == trajectory.TRANSPORT_SPEED == 4.0
    default = inspect.signature(oracle_runtime.OracleRig).parameters["transport_speed"].default
    assert default == trajectory.TRANSPORT_SPEED
    assert SCRIPT.count("transport_speed=args.transport_speed") == 1
    # The MS7e walk at its pace: (waypoints, control steps) of three legs, read at 29a36ef.
    legs = ({"distance": 0.12}, {"distance": 0.30, "tilt_change": 1.4}, {"joint_change": 1.2})
    steps = [oracle_runtime.leg_resolution(300, 1.0, speed_scale=4.0, **leg) for leg in legs]
    assert steps == [(40, 22), (100, 50), (60, 30)]


def test_a_faster_pace_shortens_the_walk_and_a_pace_under_zero_is_refused(capsys):
    assert mj_eval_heap.parse_args(["--transport_speed", "6"]).transport_speed == 6.0
    fast = oracle_runtime.leg_resolution(300, 1.0, distance=0.30, tilt_change=1.4, speed_scale=6)
    assert fast == (100, 34)
    for bad in ("0", "-1"):
        with pytest.raises(SystemExit):
            mj_eval_heap.parse_args(["--transport_speed", bad])
        assert "EVAL_FAIL" in capsys.readouterr().out
