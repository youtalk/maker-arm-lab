#!/usr/bin/env bash
# pi0.5 venv (MS7j), separate from the ACT venv (.venv, lerobot 0.5.1). lerobot 0.6.1 with the pi,
# training and peft extras. 0.6.1 moved the embedding scale into transformers, so only transformers
# 5.4 or 5.5 gives the right numbers. torch below 2.12 keeps a wheel with sm_120 for the RTX 5090.
# pyzmq serves policy_protocol.py from the shared server.
set -euo pipefail
cd "$(dirname "$0")"
uv venv .venv-pi --python 3.12
uv pip install --python .venv-pi/bin/python "lerobot[pi,training,peft]==0.6.1" pytest pyzmq
.venv-pi/bin/python -c "
import lerobot, peft, torch, transformers
from packaging.version import Version as V
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
print('lerobot', lerobot.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__, torch.cuda.get_arch_list())
raise SystemExit(0 if V('5.4.0') <= V(transformers.__version__) < V('5.6.0') and V(torch.__version__) < V('2.12') else 'transformers must be in [5.4.0, 5.6.0) and torch below 2.12')
"
