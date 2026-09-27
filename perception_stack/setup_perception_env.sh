#!/usr/bin/env bash
# The perception worker's venv (MS5 design section 4.2): CUDA torch and the segmentation
# model, apart from the lab's uv venv and the LeRobot venv. It targets the Jetson Thor, where
# the worker runs beside the loop (MS5 design decision M8).
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname -m)" != "aarch64" || ! -e /etc/nv_tegra_release ]]; then
    echo "setup_perception_env.sh: the worker venv targets the Jetson Thor" >&2
    exit 1
fi
uv venv .venv --python 3.12
# The same torch as lerobot_stack/setup_lerobot_env.sh, from the PyTorch cu130 index.
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.10.0" "torchvision==0.25.0"
# SAM 2.1 from its repository at main on 2026-09-26. Its setup.py imports torch, so it builds
# without isolation against the torch above, and without its optional CUDA extension (a
# connected-components post-process the worker does not use).
uv pip install --python .venv/bin/python setuptools wheel
# This host's ~/.gitconfig rewrites every https://github.com/ URL to git@github.com: (an
# `insteadOf`, for pushing over SSH from an interactive checkout); this account's SSH key is
# not registered with GitHub, so the rewrite turns the anonymous, read-only clone below into a
# publickey failure. GIT_CONFIG_GLOBAL=/dev/null drops just that rewrite for this one git
# subprocess so the git+https URL is fetched as written; it does not touch the user's gitconfig.
# The repository's setup.py declares NAME = "SAM-2" (PEP 503 normalizes to "sam-2"); the
# import package is still `sam2` (its actual package directory), so only the requirement
# spec's name below has to say "sam-2" for uv to accept the built metadata.
GIT_CONFIG_GLOBAL=/dev/null SAM2_BUILD_CUDA=0 uv pip install --python .venv/bin/python \
    --no-build-isolation \
    "sam-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4" \
    "huggingface_hub>=0.23" "pillow>=10" "pyzmq>=26" pytest
.venv/bin/python -c "
import torch, sam2
assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (11, 0)
print('torch', torch.__version__, 'sam2 ok')
"
