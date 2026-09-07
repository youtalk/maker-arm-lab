# maker-arm-lab

Research layer for the MakerMods Maker Arm: system identification, model
learning, and gated milestone experiments (MA0–MA5). Hardware access goes
through the [maker-arm-rs](https://github.com/youtalk/maker-arm-rs) SDK's
`maker_arm_rs` Python module.

Milestone reports live in `docs/`. Current: MA0 (bench prep).

## Simulation asset build

The simulator loads `assets/usd/maker_arm.usd`, which is **generated and git-ignored**:

```bash
$ISAACLAB_PATH/isaaclab.sh -p scripts/convert_urdf_to_usd.py --headless
```

Run this once after cloning, and again whenever `scripts/convert_urdf_to_usd.py` or the
vendored URDF changes. Nothing checks freshness for you: a stale USD loads without complaint
and then misbehaves physically rather than failing.
