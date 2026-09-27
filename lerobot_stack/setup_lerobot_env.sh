#!/usr/bin/env bash
# LeRobot conversion venv, deliberately separate from the Isaac Sim venv.
# lerobot 0.5.1 requires Python >= 3.12 (the Isaac venv is 3.11).
set -euo pipefail
cd "$(dirname "$0")"
uv venv .venv --python 3.12
# MS5 serves the policy on the Jetson Thor (aarch64, CUDA 13, sm_110), whose GPU PyPI's
# aarch64 torch is not built for. The Jetson AI Lab SBSA index is; 2.10.0 and 0.25.0 are the
# newest there inside lerobot 0.5.1's bounds (torch < 2.11, torchvision < 0.26). They go in
# first, so the lerobot install below finds them satisfied and keeps them.
THOR_INDEX="https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple/"
on_thor=false
if [[ "$(uname -m)" == "aarch64" && -e /etc/nv_tegra_release ]]; then
    on_thor=true
    uv pip install --python .venv/bin/python --index-url "$THOR_INDEX" \
        "torch==2.10.0" "torchvision==0.25.0"
fi
# wandb arrives transitively with lerobot, but train_act.sh enables it by default,
# so it is named here to keep that a declared dependency rather than one a future
# lerobot bump could quietly drop.
uv pip install --python .venv/bin/python "lerobot==0.5.1" "h5py>=3.10" "wandb>=0.24" pytest pyzmq
.venv/bin/python -c "import lerobot, wandb; from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('lerobot', lerobot.__version__, 'wandb', wandb.__version__)"
if $on_thor; then
    .venv/bin/python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA device'
assert torch.cuda.get_device_capability() == (11, 0), torch.cuda.get_device_capability()
x = torch.ones(256, 256, device='cuda')
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'check', float((x @ x).sum()))
"
fi
