"""Tier 0 — the reserved ``dawis.selection`` namespace (#48).

Atom selection is the step that turns atoms into source models (#4). Its
criteria and API are specced on #17 and its interactive interface on #18. The
namespace is no longer empty — #125 put the criterion protocol in it — but
what it holds is still only what a landed ticket built.

Two things are checked. First, that the namespace is importable, still says
where the spec lives — a stub whose pointer has rotted is worse than no stub
— and exports what it claims to. Second, that every subpackage under ``dawis``
is declared in
``pyproject.toml``: setuptools is configured with an explicit package list
here, so a subpackage missing from it exists in the source tree and vanishes
from the installed wheel. That check is not specific to this ticket, but this
is the ticket that adds the eighth entry to that list.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import dawis.selection

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / 'dawis'


def declared_packages() -> set[str]:
    """Read the explicit setuptools package list out of ``pyproject.toml``."""
    with open(REPO_ROOT / 'pyproject.toml', 'rb') as f:
        pyproject = tomllib.load(f)
    return set(pyproject['tool']['setuptools']['packages'])


def source_packages() -> set[str]:
    """Every importable package in the source tree, as dotted names."""
    return {
        '.'.join(init.parent.relative_to(REPO_ROOT).parts)
        for init in PACKAGE_ROOT.rglob('__init__.py')
    }


def test_selection_namespace_is_importable():
    assert dawis.selection.__doc__ is not None


def test_selection_docstring_points_at_the_specs():
    doc = dawis.selection.__doc__
    assert '#17' in doc, 'the criteria and API spec'
    assert '#18' in doc, 'the interactive interface spec'


def test_selection_holds_only_what_has_landed():
    """#125 added the criterion protocol; #126 and #127 add the rest."""
    modules = sorted(m.name for m in (PACKAGE_ROOT / 'selection').glob('*.py'))
    assert modules == ['__init__.py', 'criteria.py']
    assert dawis.selection.__all__ == ['Criterion', 'split']
    for name in dawis.selection.__all__:
        assert getattr(dawis.selection, name, None) is not None, name


def test_every_source_package_is_declared_in_pyproject():
    assert source_packages() - declared_packages() == set()


def test_every_declared_package_exists_in_the_source_tree():
    assert declared_packages() - source_packages() == set()
