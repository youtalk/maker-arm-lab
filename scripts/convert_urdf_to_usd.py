"""Convert the vendored Maker Arm URDF to USD through Isaac Lab's UrdfConverter.

Run with:  $ISAACLAB_PATH/isaaclab.sh -p scripts/convert_urdf_to_usd.py
Prints CONVERT_OK <usd path> or CONVERT_FAIL; exits 0/1 via os._exit (Kit's teardown does
not reliably preserve exit codes -- see smoke_usd.py).

What is converted is not the vendored file but the Z-up derived copy written by
maker_arm_lab.assets.urdf_prep.write_zup_urdf into the build directory (plan refinement 1).
Mimic joints become independent joints, so both gripper jaws are driven to the same target
by the env's BinaryJointPositionAction, matching the two-finger structure the oracle's pinch
test reasons about. That takes `convert_mimic_joints_to_normal_joints=False`, NOT True:
Isaac Lab's cfg field is the inverse of its own name. `UrdfConverter._get_urdf_import_config`
passes it straight through as `import_config.set_parse_mimic(...)`, and the URDF importer's
own tooltip for that switch reads "If false, creates a PhysX Mimic Joint to enforce the mimic
joint behavior. Otherwise, the joint is treated as a regular joint." Measured on the live
stage, Task D3: with True, `gripper_right_joint` carried a `PhysxMimicJointAPI:rotY` (gearing
-1, referenceJointAxis `rotX` -- rotational tokens on a prismatic pair), no
`PhysicsDriveAPI:linear` at all, limits padded to [-0.0105, 0.0629] and its velocity limit
raised to 1e6, and it fought the actuator's PD drive instead of tracking it. With False both
jaws are ordinary driven prismatic joints with identical authored properties.

Fixed joints are kept so `grasp_center` (and `mount_link`) survive as bodies.
"""

import argparse
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=Path("assets/usd/maker_arm.usd"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

from maker_arm_lab.assets import urdf_prep


def run_conversion() -> bool:
    out = args.out.resolve()
    derived = urdf_prep.write_zup_urdf(urdf_prep.VENDORED_URDF, out.parent / "robot_zup.urdf")
    cfg = UrdfConverterCfg(
        asset_path=str(derived),
        usd_dir=str(out.parent),
        usd_file_name=out.name,
        fix_base=True,
        merge_fixed_joints=False,
        # False really does mean "convert the mimic to a normal joint" -- see the docstring.
        convert_mimic_joints_to_normal_joints=False,
        # The converter's cache key hashes the URDF bytes, not the meshes; a re-vendor can
        # change meshes without changing the URDF, so always regenerate.
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=10.0),
        ),
    )
    converter = UrdfConverter(cfg)
    produced = Path(converter.usd_path)
    if not produced.is_file():
        print(f"converter reported {produced}, which does not exist", flush=True)
        return False
    print(f"CONVERT_OK {produced}", flush=True)
    return True


try:
    ok = run_conversion()
except Exception:  # noqa: BLE001 -- any failure must become CONVERT_FAIL, not a crash.
    traceback.print_exc()
    ok = False

if not ok:
    print("CONVERT_FAIL", flush=True)
os._exit(0 if ok else 1)
