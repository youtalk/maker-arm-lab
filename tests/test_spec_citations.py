"""Pin the MS1 citation sweep: no dangling OpenYAM-style ``section 5.x`` spec references.

The MS1 design spec's section 5 is an unsubdivided numbered list, so ``section 5 item N`` is
the only valid sub-reference; the dotted forms were OpenYAM's numbering carried in
by the port. A citation qualified as ``MS2 spec section 5.x`` names a spec that does have
those sections and is allowed. ``docs/ms1-maker-arm-sim.md`` section 7 records the sweep this test protects.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SWEPT_DIRS = ("source", "scripts", "lerobot_stack", "tests")
SUFFIXES = {".py", ".sh"}
SKIP_PARTS = {".venv", "__pycache__"}
# The MS2 design spec and the MS2 report (docs/ms2-act-gate.md) both HAVE a subdivided
# section 5, and the scripts cite them as "MS2 spec section 5.x" / "MS2 report section 5.x"
# on purpose; only an unqualified dotted citation is dangling.
DANGLING = re.compile(r"(?<!MS2 spec )(?<!MS2 report )section 5\.\d")


def _swept_files() -> list[Path]:
    files = []
    for top in SWEPT_DIRS:
        for path in (REPO / top).rglob("*"):
            if path.suffix in SUFFIXES and not SKIP_PARTS.intersection(path.parts):
                files.append(path)
    return sorted(files)


def test_sweep_covers_the_known_files():
    names = {p.name for p in _swept_files()}
    for expected in ("oracle.py", "gate.py", "mit_law.py", "generate_batch.py", "train_act.sh"):
        assert expected in names


@pytest.mark.parametrize("path", _swept_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_no_dangling_section_5_x_citation(path: Path):
    hits = [
        f"{path.relative_to(REPO)}:{n}: {line.strip()}"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if DANGLING.search(line)
    ]
    assert not hits, "\n".join(hits)
