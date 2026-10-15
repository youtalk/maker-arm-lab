"""Shared test helpers: the repository root and the script loader."""

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_script(script: str | Path) -> ModuleType:
    """Import a script file as a fresh module named after its file stem. A bare name is a file
    under ``scripts/``. The module is not registered in ``sys.modules``."""
    path = ROOT / "scripts" / f"{script}.py" if isinstance(script, str) else script
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
