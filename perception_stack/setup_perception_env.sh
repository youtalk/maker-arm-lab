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
# A JetPack shell profile exports LD_LIBRARY_PATH=/usr/local/cuda/lib64. That directory takes
# precedence over the torch wheels' RUNPATH, so JetPack's CUDA 13.0 cuBLAS shadows the venv's
# pip cuBLAS 13.1 and every GEMM raises CUBLAS_STATUS_INVALID_VALUE. The check below runs with
# it unset; the venv's python has to run the same way (env -u LD_LIBRARY_PATH).
if [[ ":${LD_LIBRARY_PATH:-}:" == *cuda* ]]; then
    echo "setup_perception_env.sh: warning: LD_LIBRARY_PATH=$LD_LIBRARY_PATH holds a CUDA" \
        "library directory, which shadows the venv's pip CUDA libraries and breaks every" \
        "CUDA op. This setup unsets it; run the venv's python with env -u LD_LIBRARY_PATH." >&2
    unset LD_LIBRARY_PATH
fi
# --clear rebuilds the venv on every run, as lerobot_stack/setup_lerobot_env.sh does: a re-run
# after a partial failure starts clean and ends with exactly the pins below. uv's cache (the
# SAM 2 wheel built at the pinned commit included) makes the rebuild a relink.
uv venv --clear .venv --python 3.12
# The same torch as lerobot_stack/setup_lerobot_env.sh, from the PyTorch cu130 index.
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.10.0" "torchvision==0.25.0"
# SAM 2.1 from its repository at main on 2026-09-26. Its setup.py imports torch, so it builds
# without isolation against the torch above, and without its optional CUDA extension (a
# connected-components post-process the worker does not use).
uv pip install --python .venv/bin/python setuptools wheel
# A global git `insteadOf` rewrite of https://github.com/ to SSH (git@github.com:) turns the
# anonymous, read-only clone below into an SSH fetch, which fails without a GitHub-registered
# key. GIT_CONFIG_GLOBAL=/dev/null makes this one uv command (and the git it runs) read no
# global git config at all, the rewrite included, so the git+https URL is fetched as written.
# It does not touch ~/.gitconfig; a proxy or CA setting kept only there is dropped too.
# The repository's setup.py declares NAME = "SAM-2" (PEP 503 normalizes to "sam-2"); the
# import package is still `sam2` (its actual package directory), so only the requirement
# spec's name below has to say "sam-2" for uv to accept the built metadata.
GIT_CONFIG_GLOBAL=/dev/null SAM2_BUILD_CUDA=0 uv pip install --python .venv/bin/python \
    --no-build-isolation \
    "sam-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4" \
    "huggingface_hub>=0.23" "pillow>=10" "pyzmq>=26" pytest
.venv/bin/python -c "
import torch, sam2
assert torch.cuda.is_available(), 'no CUDA device'
assert torch.cuda.get_device_capability() == (11, 0), torch.cuda.get_device_capability()
x = torch.ones(256, 256, device='cuda')
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'check', float((x @ x).sum()),
      'sam2 ok')
"
