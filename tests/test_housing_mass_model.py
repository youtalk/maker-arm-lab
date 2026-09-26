"""The housing mass model asset (tracking design section 3): the bench's model error."""

import json

from maker_arm_lab.core import dynamics_model as dm

DESK_POSE = (0.0, -2.03, 1.71, -1.17, 0.0, 0.0)


def test_the_asset_carries_its_provenance_and_every_body():
    data = json.loads(dm.HOUSING_ASSET.read_text())
    for key in ("source", "rules", "urdf_pin", "script", "generated"):
        assert key in data["provenance"], key
    assert set(data["bodies"]) == set(dm.ARM_LINKS) | set(dm.JAW_LINKS)
    assert all(b["mass"] > 0.0 for b in data["bodies"].values())


def test_housing_gravity_at_the_desk_pose_matches_the_study():
    # The study: 5.36 N m at j2 and 3.36 N m at j3, 13 % and 15 % above the URDF.
    g = dm.dynamics(dm.housing_bodies()).gravity(DESK_POSE)
    assert abs(abs(g[1]) - 5.36) < 0.15 and abs(abs(g[2]) - 3.36) < 0.15, g
    u = dm.dynamics(dm.urdf_bodies()).gravity(DESK_POSE)
    assert abs(g[1]) > abs(u[1]) and abs(g[2]) > abs(u[2])
