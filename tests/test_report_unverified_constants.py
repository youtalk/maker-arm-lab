"""Couple the milestone report's divergence list to the source's ``UNVERIFIED`` flags.

``docs/ms1-maker-arm-sim.md`` section 4.3 is the artifact the October hardware bring-up
works from, and it is prose. The ``<NAME>_UNVERIFIED`` flags in ``source/`` are the
greppable half; nothing otherwise stops a future edit from adding a flagged constant
without telling the bring-up about it, and the report would then go stale independently
of the code it describes. This test is that guard: it fails in CI rather than in October.

WHAT IS COVERED, and what deliberately is not. The direction pinned here is source ->
report: a flag added to ``source/`` whose constant the report does not name fails. The
reverse -- a flag REMOVED from source while the report still lists it -- is not pinned,
because doing so needs a literal roster of the flags in a third place, which is one more
site to update on every legitimate change; and a report that still names a constant
somebody has since measured on the real arm is stale in the SAFE direction, unlike one
that omits a constant nobody has measured at all. What IS pinned is that the section is
found and that the scan is non-empty, so this test cannot pass vacuously -- the failure
mode this milestone has caught five times.

The scan uses ``ast`` rather than a grep so that a mention of a flag inside a comment, a
docstring or a string literal cannot produce a false positive. That is what makes the
coupling cheap enough to keep: it needs no allow-list and no babysitting.
"""

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_DIR = _REPO_ROOT / "source"
_REPORT = _REPO_ROOT / "docs" / "ms1-maker-arm-sim.md"

# The section the bring-up reads. Renaming it must fail loudly here rather than silently
# reducing this test to a tautology over an empty string.
_SECTION_HEADING = "### 4.3 Sim/real divergences and unverified constants"
_FLAG_SUFFIX = "_UNVERIFIED"


def _unverified_constants() -> dict[str, str]:
    """Constant name -> the source file whose ``<NAME>_UNVERIFIED`` flag declares it.

    Only module-level assignments count. ``WRIST_CAM_UNVERIFIED`` yields ``WRIST_CAM``,
    which is the prefix the report spells out as ``WRIST_CAM_POS_IN_WRIST`` /
    ``WRIST_CAM_ROT_IN_WRIST``; ``ARM_ARMATURE_UNVERIFIED`` yields ``ARM_ARMATURE``.
    """
    found: dict[str, str] = {}
    for path in sorted(_SOURCE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if target.id.endswith(_FLAG_SUFFIX) and len(target.id) > len(_FLAG_SUFFIX):
                    name = target.id[: -len(_FLAG_SUFFIX)]
                    found[name] = str(path.relative_to(_REPO_ROOT))
    return found


def _divergence_section() -> str:
    text = _REPORT.read_text(encoding="utf-8")
    if _SECTION_HEADING not in text:
        raise AssertionError(
            f"{_REPORT.relative_to(_REPO_ROOT)} has no section headed {_SECTION_HEADING!r}. "
            "That section is what the October hardware bring-up works from; if it moved or "
            "was renamed, update _SECTION_HEADING here so this coupling keeps its teeth."
        )
    start = text.index(_SECTION_HEADING)
    end = text.index("\n## ", start)  # the next top-level section; '### ' cannot match
    return text[start:end]


def test_the_scan_and_the_report_section_are_both_non_empty():
    """Neither half may be empty, or the coupling below would hold vacuously."""
    section = _divergence_section()
    assert len(section.splitlines()) > 5, "the divergence section is too short to be the real one"
    assert _unverified_constants(), (
        f"no module-level '<NAME>{_FLAG_SUFFIX}' flags found under source/. Either every "
        "unverified constant was measured and removed -- in which case update the report's "
        "section 4.3 and this test together -- or the scan is broken."
    )


def test_every_unverified_constant_is_named_in_the_report_divergence_list():
    section = _divergence_section()
    missing = {
        name: where for name, where in _unverified_constants().items() if name not in section
    }
    assert not missing, (
        "these constants are flagged UNVERIFIED in source but are not named in "
        f"docs/ms1-maker-arm-sim.md's {_SECTION_HEADING!r}: "
        + ", ".join(f"{name} ({where})" for name, where in sorted(missing.items()))
        + ". That section is the list the October hardware bring-up re-measures from, so a "
        "constant missing from it is one nobody will know to measure. Add it there, with "
        "what it is bounded by and what a measured value has to clear."
    )
