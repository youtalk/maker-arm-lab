#!/usr/bin/env bash
# LeRobot conversion venv, deliberately separate from the Isaac Sim venv.
# lerobot 0.5.1 requires Python >= 3.12 (the Isaac venv is 3.11).
set -euo pipefail
cd "$(dirname "$0")"
uv venv .venv --python 3.12
# wandb arrives transitively with lerobot, but train_act.sh enables it by default,
# so it is named here to keep that a declared dependency rather than one a future
# lerobot bump could quietly drop.
uv pip install --python .venv/bin/python "lerobot==0.5.1" "h5py>=3.10" "wandb>=0.24" pytest pyzmq
.venv/bin/python -c "import lerobot, wandb; from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('lerobot', lerobot.__version__, 'wandb', wandb.__version__)"
