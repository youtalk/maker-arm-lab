#!/usr/bin/env bash
# pi0.5 venv (MS7h design 7.3), separate from the ACT venv (.venv): lerobot 0.5.1's pi extra adds
# transformers 5.3.0 and scipy, which .venv does not carry. Same Python and torch source as
# setup_lerobot_env.sh: the PyPI torch wheel lerobot 0.5.1 resolves to (2.10, CUDA 12.8) includes
# sm_120, so it runs on the RTX 5090. pyzmq serves policy_protocol.py from the shared server.
set -euo pipefail
cd "$(dirname "$0")"
uv venv .venv-pi --python 3.12
uv pip install --python .venv-pi/bin/python "lerobot[pi]==0.5.1" pytest pyzmq
.venv-pi/bin/python -c "import lerobot, transformers, torch; from lerobot.policies.pi05.modeling_pi05 import PI05Policy; print('lerobot', lerobot.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__, torch.cuda.get_arch_list())"
