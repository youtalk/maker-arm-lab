#!/usr/bin/env bash
# Vendor makermods-robotics/maker-arm-sdk's urdf/maker_arm at a pinned commit.
# Re-vendor: bump PIN_SHA, re-run, then `uv run pytest tests/test_urdf_validation.py`.
set -euo pipefail

REPO_URL="git@github.com:makermods-robotics/maker-arm-sdk.git"
PIN_SHA="b30d05a23d72e8c155a8e00f807aba8e8c705f68"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/assets/vendor/maker_arm_description"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone --quiet --filter=blob:none "$REPO_URL" "$TMP/sdk"
git -C "$TMP/sdk" checkout --quiet "$PIN_SHA"

rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "$TMP/sdk/urdf/maker_arm/." "$DEST/"
cp "$TMP/sdk/LICENSE" "$DEST/LICENSE"

{
  echo "# Vendored: maker_arm_description"
  echo
  echo "- Source: https://github.com/makermods-robotics/maker-arm-sdk, directory \`urdf/maker_arm\`"
  echo "- Pinned commit: \`$PIN_SHA\` (\"Add Maker Arm URDF with symmetric sliding gripper\", 2026-09-05)"
  echo "- License: Apache-2.0 (\`LICENSE\`, copied verbatim from the upstream repository root)"
  echo "- Vendored on: $(date -u +%Y-%m-%d) with \`scripts/vendor_maker_arm_description.sh\`"
  echo "- The files are used unmodified. The Z-up derived copy the simulator loads is"
  echo "  generated into \`assets/usd/\` by \`source/maker_arm_lab/assets/urdf_prep.py\`."
  echo
  echo "## Known limitations (from upstream README.md and revision_report.json)"
  echo
  echo "- Not validated for RL dynamics: inertias are CAD density estimates; moving-jaw"
  echo "  mass/inertia are redistributed, not measured."
  echo "- Gripper effort (20 N) and velocity (0.1 m/s) are placeholders; the motor-angle to"
  echo "  jaw-gap transmission is uncalibrated (two-point observation only)."
  echo "- Arm joint zero alignment and actuator limits are unverified against the arm."
  echo "- No camera frame. \`part_003\` is missing from the export."
  echo "- Base frame is the CAD's Y-up frame."
  echo
  echo "## SHA-256"
  echo
  echo '```'
  (cd "$DEST" && find . -type f ! -name VENDOR.md | sort | xargs sha256sum)
  echo '```'
} > "$DEST/VENDOR.md"

echo "OK: vendored maker_arm_description @ $PIN_SHA into $DEST"
