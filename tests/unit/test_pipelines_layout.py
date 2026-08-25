"""Tier 0 — the layout rules the ``dawis.pipelines`` carve rests on (#49).

Two rules, both from the target architecture (#4), both stated in prose in the
package docstrings and neither previously executable.

**Blocks never import pipelines.** A block that imported the pipeline would be
a block that cannot be run without the whole solver behind it, and the toolbox
would collapse back into the flat package it came from. The edge runs one way:
pipelines may import any block.

**No algorithm mathematics lives in a pipeline.** A pipeline holds the order
the steps run in and the conditions the loop stops under; every number it
produces is produced by a call into a block. The proxy checked here is that a
pipeline module needs no numerical library — a file that reaches for numpy is
a file doing arithmetic.

The full import-graph walk over all three of #4's dependency rules is #59;
this covers the two rules this ticket is accountable for.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import dawis
import dawis.pipelines

PACKAGE_ROOT = Path(dawis.__file__).resolve().parent
PIPELINES_ROOT = PACKAGE_ROOT / 'pipelines'

#: Libraries a pipeline has no business importing, because importing one means
#: doing arithmetic in a file whose job is composition.
NUMERICAL_LIBRARIES = {'numpy', 'scipy', 'skimage', 'photutils'}


def imported_modules(path: Path) -> set[str]:
    """Every module name a source file imports, as written."""
    names = set()
    with warnings.catch_warnings():
        # Legacy docstrings in the package carry stray backslashes. Parsing
        # them is not the subject of this test.
        warnings.simplefilter('ignore', SyntaxWarning)
        tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def package_modules() -> list[Path]:
    """Every source file in the installed package."""
    return sorted(PACKAGE_ROOT.rglob('*.py'))


def test_no_block_imports_a_pipeline():
    """The dependency edge runs one way, and only one.

    The shim ``dawis.synthesis_by_analysis`` used to be a deliberate exception
    here — the old flat module's address, kept reachable while callers moved.
    The contract ticket (#50) deleted it, and the package ``__init__`` now
    imports the entry point from :mod:`dawis.pipelines` directly, so it takes
    the exemption over: an ``__init__`` that re-exports the public surface is
    the package speaking for itself, not a block reaching for a pipeline.
    """
    offenders = {
        str(path.relative_to(PACKAGE_ROOT))
        for path in package_modules()
        if PIPELINES_ROOT not in path.parents
        and path != PACKAGE_ROOT / '__init__.py'
        and any(name.startswith('dawis.pipelines') for name in imported_modules(path))
    }

    assert offenders == set(), (
        'these blocks import dawis.pipelines, which reverses the one '
        'dependency edge the layout rests on'
    )


def test_a_pipeline_imports_no_numerical_library():
    """No numpy in a pipeline means no arithmetic in a pipeline."""
    for path in sorted(PIPELINES_ROOT.rglob('*.py')):
        roots = {name.split('.')[0] for name in imported_modules(path)}
        assert not roots & NUMERICAL_LIBRARIES, (
            '%s imports a numerical library, so algorithm mathematics has '
            'come back into a pipeline' % path.name
        )


def test_the_entry_point_lives_in_pipelines_and_answers_to_its_old_name():
    """The carve moved the function; every caller's address still resolves."""
    assert dawis.synthesis_by_analysis is dawis.pipelines.synthesis_by_analysis
    assert dawis.synthesis_by_analysis.__module__ == \
        'dawis.pipelines.synthesis_by_analysis'
