import math

import pytest

from maker_arm_lab.core.quat import quat_from_rpy_deg, rpy_from_quat


@pytest.mark.parametrize("rpy", [(0.0, 0.0, 0.0), (10.0, -20.0, 135.0), (-80.0, 5.0, -170.0)])
def test_rpy_round_trips_through_the_quaternion(rpy):
    back = rpy_from_quat(quat_from_rpy_deg(rpy))
    assert back == pytest.approx(tuple(math.radians(v) for v in rpy), abs=1e-9)
