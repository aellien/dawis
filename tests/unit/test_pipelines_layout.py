"""Tier 0 — the layout rules the ``dawis.pipelines`` carve rests on (#49).

**No algorithm mathematics lives in a pipeline.** A pipeline holds the order
the steps run in and the conditions the loop stops under; every number it
produces is produced by a call into a block. The proxy checked here is that a
pipeline module needs no numerical library — a file that reaches for numpy is
a file doing arithmetic.

The other rule this ticket was accountable for — that no block imports a
pipeline — moved to ``test_import_graph.py`` with #58, which walks the source
tree once for all three of #4's dependency rules. It is stated there with the
matplotlib and ray rules it belongs beside, not here.
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


def test_a_pipeline_imports_no_numerical_library():
    """No numpy in a pipeline means no arithmetic in a pipeline."""
    for path in sorted(PIPELINES_ROOT.rglob('*.py')):
        roots = {name.split('.')[0] for name in imported_modules(path)}
        assert not roots & NUMERICAL_LIBRARIES, (
            '%s imports a numerical library, so algorithm mathematics has '
            'come back into a pipeline' % path.name
        )


def test_the_entry_points_live_in_pipelines():
    """The package's front door is the pipelines package's, and only that.

    #54 replaced the fifteen-keyword ``synthesis_by_analysis`` with the pair
    below. The module keeps the old name — it is what the *pipeline* is
    called — but the function is gone, and no second address for either entry
    point survives.
    """
    assert dawis.run is dawis.pipelines.run
    assert dawis.run_fits is dawis.pipelines.run_fits
    assert dawis.run.__module__ == 'dawis.pipelines.synthesis_by_analysis'
    assert dawis.run_fits.__module__ == 'dawis.pipelines.synthesis_by_analysis'
    assert not hasattr(dawis, 'synthesis_by_analysis')
