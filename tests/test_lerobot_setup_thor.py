"""MS5 design P1: the LeRobot venv's Thor path pins a CUDA 13 torch inside lerobot's bounds."""

import re
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "lerobot_stack" / "setup_lerobot_env.sh"
# PyTorch's cu130 index: its aarch64 wheels carry sm_110 kernels and pull their CUDA
# libraries as pip dependencies (the Jetson AI Lab SBSA build needs system NVPL and cuDSS).
THOR_INDEX = "https://download.pytorch.org/whl/cu130"


def _pin(text: str, name: str) -> tuple[int, ...]:
    found = re.search(rf'"{name}==([0-9.]+)"', text)
    assert found, f"{name} is not pinned"
    return tuple(int(v) for v in found.group(1).split("."))


def test_the_thor_branch_installs_torch_from_the_cu130_index_before_lerobot():
    text = SCRIPT.read_text()
    assert THOR_INDEX in text
    assert text.index(THOR_INDEX) < text.index('"lerobot==0.5.1"')
    assert "/etc/nv_tegra_release" in text


def test_the_pins_sit_inside_lerobot_0_5_1s_bounds():
    text = SCRIPT.read_text()
    assert (2, 7) <= _pin(text, "torch") < (2, 11)
    assert (0, 22) <= _pin(text, "torchvision") < (0, 26)


def test_the_thor_branch_checks_cuda_on_sm_110():
    text = SCRIPT.read_text()
    assert "torch.cuda.is_available()" in text and "(11, 0)" in text


def test_the_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
