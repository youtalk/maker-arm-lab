"""The Maker Arm's frame conventions plus its (isaaclab) ArticulationCfg.

This package is split across an isaaclab boundary that its own contents straddle:
``frames`` is pure Python and is what the CPU test suite (which has no isaaclab
installed) imports directly, while ``maker_arm`` builds ``MAKER_ARM_CFG`` out of
``isaaclab.sim`` / ``isaaclab.assets`` types and can only be imported where isaaclab
exists.

A plain ``from .maker_arm import *`` at module scope here would import isaaclab as a
side effect of importing this package at all -- which breaks not just "``import
maker_arm_lab.robots``" but also "``from maker_arm_lab.robots import frames``", because
Python always runs a package's ``__init__.py`` before handing back a submodule, whether
it is reached via ``from pkg import submodule`` or ``import pkg.submodule``. Since the
CPU test suite needs exactly that submodule import to work without isaaclab, the
isaaclab-dependent names cannot be pulled in eagerly.

So: the isaaclab-free names are re-exported eagerly below, and the
isaaclab-dependent names (``MAKER_ARM_CFG``, ``MAKER_ARM_USD_PATH``,
``GRIPPER_EFFORT_LIMIT_N``, ``ARM_ARMATURE``, ``ARM_ARMATURE_UNVERIFIED``) are resolved
lazily through the module-level
``__getattr__`` (PEP 562): touching one of them imports ``.maker_arm`` -- and only then
pulls in isaaclab -- on first access, while ``from maker_arm_lab.robots import frames``
never touches it at all.
"""

# There is no fallback for the profile (see core.profile): a missing maker_arm_rs must break
# `import maker_arm_lab.robots` immediately, not only when MAKER_ARM_CFG is first touched, which
# is what the lazy __getattr__ below would otherwise silently allow. Do not delete this as
# "unused" -- pulling in maker_arm_rs itself, not anything from it, is the point. It imports no
# isaaclab, so it does not undermine the lazy split for the isaaclab-dependent names.
import maker_arm_rs  # noqa: F401 -- fail fast at import, not deferred to first attribute access.

from .frames import *  # noqa: F403 -- isaaclab-free; safe to re-export eagerly

_LAZY_NAMES = frozenset(
    {
        "MAKER_ARM_CFG",
        "MAKER_ARM_USD_PATH",
        "GRIPPER_EFFORT_LIMIT_N",
        "ARM_ARMATURE",
        "ARM_ARMATURE_UNVERIFIED",
    }
)


def __getattr__(name: str):
    if name in _LAZY_NAMES:
        from . import maker_arm

        return getattr(maker_arm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
