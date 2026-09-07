#!/usr/bin/env bash
# Install pinned Isaac Sim + Isaac Lab into $ISAACLAB_PATH (idempotent).
set -euo pipefail

# Captured before the Isaac Lab bootstrap below does `cd "$ISAACLAB_PATH"` and stays
# there for the rest of the script. $0 is relative when invoked as `bash
# scripts/setup_isaac_env.sh`, so resolving it after that `cd` would resolve against
# the wrong directory -- and since IsaacLab also has its own `scripts/` subdirectory,
# it would silently resolve back to $ISAACLAB_PATH instead of this repo's root.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/src/IsaacLab}"
ISAACLAB_TAG="v2.3.2"
ISAACSIM_PIP="isaacsim[all,extscache]==5.1.0"
PYTHON_VERSION="3.11"

# Isaac Sim requires accepting NVIDIA's Omniverse license (EULA) before it can be
# imported. When unaccepted in a non-interactive shell, isaacsim's bootstrap_kernel()
# wraps the EULA prompt's EOFError in sys.exit(...), which raises SystemExit and
# escapes isaaclab.sh's own `except EOFError:` safety net -- surfacing as an opaque
# crash ("Unable to bootstrap inner kit kernel: EOF when reading a line") deep inside
# `./isaaclab.sh --install`. Fail fast, before doing any work, with a clear message
# instead. This script must never export OMNI_KIT_ACCEPT_EULA=YES itself -- accepting
# the license is a decision only the person running this script can make.
if [ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]; then
    echo "FAIL: Isaac Sim requires accepting NVIDIA's Omniverse license (EULA)." >&2
    echo "To indicate acceptance, export OMNI_KIT_ACCEPT_EULA=YES and re-run this script:" >&2
    echo "  export OMNI_KIT_ACCEPT_EULA=YES" >&2
    exit 1
fi

if [ ! -d "$ISAACLAB_PATH" ]; then
    git clone --branch "$ISAACLAB_TAG" --depth 1 \
        https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_PATH"
fi

# The pin is the entire point of this script. Whether $ISAACLAB_PATH was just cloned
# above or already existed from a previous run, verify it is actually checked out at
# $ISAACLAB_TAG before doing anything else -- otherwise a pre-existing checkout at some
# other tag would silently pass through and this script would still print "OK: Isaac
# Lab $ISAACLAB_TAG ready" at the end, over a tree that is not actually pinned to that
# tag. This is exactly the false-success failure mode the `import isaaclab` check
# below exists to prevent on the venv side.
if [ ! -d "$ISAACLAB_PATH/.git" ]; then
    echo "FAIL: $ISAACLAB_PATH exists but is not a git checkout (no .git found)." >&2
    exit 1
fi
FOUND_TAG="$(git -C "$ISAACLAB_PATH" describe --tags --exact-match 2>/dev/null || true)"
if [ "$FOUND_TAG" != "$ISAACLAB_TAG" ]; then
    FOUND_REF="$(git -C "$ISAACLAB_PATH" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "FAIL: $ISAACLAB_PATH is not checked out at $ISAACLAB_TAG" \
        "(found: ${FOUND_TAG:-untagged commit} @ $FOUND_REF)." >&2
    echo "Point ISAACLAB_PATH at a fresh directory, or check out $ISAACLAB_TAG there yourself." >&2
    exit 1
fi

cd "$ISAACLAB_PATH"
if [ -d .venv ]; then
    # Same reasoning as the tag check above: a pre-existing .venv at the wrong Python
    # version must fail loudly here, not be silently reused and then declared "ready".
    FOUND_PY="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [ "$FOUND_PY" != "$PYTHON_VERSION" ]; then
        echo "FAIL: $ISAACLAB_PATH/.venv is Python ${FOUND_PY:-unknown}," \
            "not the pinned $PYTHON_VERSION." >&2
        echo "Remove $ISAACLAB_PATH/.venv to let this script recreate it," \
            "or point ISAACLAB_PATH elsewhere." >&2
        exit 1
    fi
else
    uv venv --seed --python "$PYTHON_VERSION" .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install "$ISAACSIM_PIP" --extra-index-url https://pypi.nvidia.com

# Isaac Lab's own `isaaclab` core package (source/isaaclab) depends on flatdict==4.0.1,
# a legacy sdist whose setup.py needs "pkg_resources" at build time without declaring
# it as a build dependency. The setuptools version `uv venv --seed` provisions here
# (verified: 83.0.0) no longer bundles pkg_resources, and no standalone pkg_resources
# package exists on PyPI to add instead, so uv's isolated build environment for
# flatdict fails with "ModuleNotFoundError: No module named 'pkg_resources'" -- and
# `isaaclab.sh --install` does not propagate that failure, so it reports success even
# though the core `isaaclab` package never actually installed. Pin the setuptools
# version resolved *inside* uv's isolated build environments (verified working:
# 75.8.0) via UV_BUILD_CONSTRAINT; this does not touch this venv's own installed
# setuptools or any of the Isaac Sim / Isaac Lab / Python pins above.
BUILD_CONSTRAINT_FILE="$(mktemp)"
# Armed here, not where $RS_TMP is created below, so that a failure in either
# `./isaaclab.sh --install` or the `import isaaclab` check right after it -- both real,
# previously-encountered failure points, per the flatdict/pkg_resources comment above --
# still cleans up $BUILD_CONSTRAINT_FILE. $RS_TMP does not exist yet at this point in the
# script, so it is expanded defensively (`${RS_TMP:-}`): under `set -u`, an unset `$RS_TMP`
# would otherwise make the trap body itself error out before reaching the first `rm`.
# `rm -rf ""` is a silent no-op, so this is a safe skip rather than a failure when $RS_TMP
# is still unset, and it never alters the script's own exit status either way.
trap 'rm -f "$BUILD_CONSTRAINT_FILE"; rm -rf "${RS_TMP:-}"' EXIT
echo "setuptools==75.8.0" >"$BUILD_CONSTRAINT_FILE"
export UV_BUILD_CONSTRAINT="$BUILD_CONSTRAINT_FILE"
./isaaclab.sh --install none
unset UV_BUILD_CONSTRAINT

# `isaaclab.sh --install` can report success even when a source package failed to
# build (see above), so verify the core package actually imports before declaring
# success -- printing "OK" over a broken environment would be worse than failing loudly.
if ! python -c "import isaaclab" >/dev/null 2>&1; then
    echo "FAIL: isaaclab package did not install correctly into $ISAACLAB_PATH/.venv" \
        "-- re-run with 'bash -x' or inspect the './isaaclab.sh --install none' output above" >&2
    exit 1
fi

# --- maker_arm_rs (PyO3 wheel) into the Isaac venv --------------------------------------
# The sim reads the actuator profile and the command clamp from maker-arm-rs at a pinned
# commit (spec section 4.6). Built with maturin from a throwaway clone; needs cargo on PATH.
MAKER_ARM_RS_REPO="https://github.com/youtalk/maker-arm-rs"
MAKER_ARM_RS_PIN="8f511c81f983323a1d33a3a213c18d5341566750"
if ! command -v cargo >/dev/null 2>&1; then
    echo "FAIL: cargo not found -- install rustup (https://rustup.rs) to build maker_arm_rs." >&2
    exit 1
fi
RS_TMP="$(mktemp -d)"
git clone --quiet --filter=blob:none "$MAKER_ARM_RS_REPO" "$RS_TMP/maker-arm-rs"
git -C "$RS_TMP/maker-arm-rs" checkout --quiet "$MAKER_ARM_RS_PIN"
pip install "maturin>=1.7"
(cd "$RS_TMP/maker-arm-rs/bindings/maker-arm-py" && maturin build --release --out "$RS_TMP/wheels")
pip install --force-reinstall "$RS_TMP"/wheels/maker_arm_rs-*.whl
python -c "import maker_arm_rs; assert len(maker_arm_rs.profile()['joints']) == 7"

# --- this repo, editable, so isaaclab.sh -p scripts/… can import maker_arm_lab ------------
LAB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
pip install -e "$LAB_ROOT"
python -c "import maker_arm_lab.assets.urdf_prep"

echo "OK: Isaac Lab $ISAACLAB_TAG at $ISAACLAB_PATH, maker_arm_rs @ $MAKER_ARM_RS_PIN, maker_arm_lab editable"
echo "Use: export ISAACLAB_PATH=$ISAACLAB_PATH; \"\$ISAACLAB_PATH/isaaclab.sh\" -p scripts/<script>.py --headless"
