"""Research layer for the MakerMods Maker Arm (see the maker-arm-rs SDK)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("maker-arm-lab")
except PackageNotFoundError:
    # Package is not installed (e.g. running from a source checkout without
    # `uv sync` / `pip install`); fall back to an explicit "unknown" marker
    # rather than letting the exception escape at import time.
    __version__ = "unknown"
