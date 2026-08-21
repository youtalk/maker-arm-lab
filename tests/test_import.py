from importlib.metadata import version

import maker_arm_lab


def test_version_matches_installed_metadata():
    """__version__ must agree with the installed distribution metadata.

    This catches a real packaging mismatch (e.g. a stale build, or the
    package resolving to the wrong installed distribution) rather than
    just restating a hardcoded literal.
    """
    assert maker_arm_lab.__version__ == version("maker-arm-lab")
