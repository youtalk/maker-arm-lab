"""MS5 design P1: both Thor venvs pin one CUDA 13 torch inside lerobot's bounds."""

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


PERCEPTION = Path(__file__).resolve().parents[1] / "perception_stack" / "setup_perception_env.sh"
BOTH = (SCRIPT, PERCEPTION)


def _pin_text(text: str, name: str) -> str:
    found = re.findall(rf'"{name}==([0-9.]+)"', text)
    assert len(found) == 1, f"{name} is pinned {len(found)} times"
    return found[0]


def test_the_perception_venv_pins_the_same_torch_from_the_same_index():
    lerobot, perception = SCRIPT.read_text(), PERCEPTION.read_text()
    for name in ("torch", "torchvision"):
        assert _pin_text(perception, name) == _pin_text(lerobot, name)
    assert THOR_INDEX in perception
    indexes = [
        set(re.findall(r"https://download\.pytorch\.org/whl/\w+", t)) for t in (lerobot, perception)
    ]
    assert indexes[0] == indexes[1] == {THOR_INDEX}


def test_the_perception_script_parses():
    subprocess.run(["bash", "-n", str(PERCEPTION)], check=True)


def test_both_scripts_rebuild_the_venv_on_a_rerun():
    for script in BOTH:
        venv = [ln for ln in script.read_text().splitlines() if ln.startswith("uv venv")]
        assert venv == ["uv venv --clear .venv --python 3.12"], script.name


def test_both_scripts_warn_about_and_unset_a_cuda_ld_library_path():
    for script in BOTH:
        text = script.read_text()
        guard = text.index('[[ ":${LD_LIBRARY_PATH:-}:" == *cuda* ]]')
        assert text.index("warning: LD_LIBRARY_PATH", guard) < text.index(
            "unset LD_LIBRARY_PATH", guard
        )
        assert "env -u LD_LIBRARY_PATH" in text
        # the unset precedes the venv's own checks, so setup succeeds from a JetPack shell
        assert text.index("unset LD_LIBRARY_PATH") < text.index('.venv/bin/python -c "')


def test_both_checks_run_a_cuda_matmul_on_sm_110():
    for script in BOTH:
        text = script.read_text()
        assert "(11, 0)" in text and "x @ x" in text, script.name
