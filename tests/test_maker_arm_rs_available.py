"""The profile is the sim's single source of gains and caps (spec section 4.6):
if maker_arm_rs is missing, nothing downstream may silently fall back."""

import maker_arm_rs


def test_profile_has_seven_joints():
    assert len(maker_arm_rs.profile()["joints"]) == 7


def test_clamp_command_is_exposed():
    p = maker_arm_rs.profile()
    cmds = [(0.0, 0.0, 1.0, 1.0, 0.0)] * 7
    out, _clamped = maker_arm_rs.clamp_command(cmds, p)
    assert len(out) == 7
