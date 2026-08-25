"""Tier 0 — the contract half of expand-contract (#50).

The carve tickets moved every function into a subpackage and left a shim
behind at each old address, so callers could move one at a time. This ticket
is the other half: the shims are deleted and the package ``__init__`` states
its public surface as a list instead of accumulating it out of star imports.

Three things are pinned here, one per acceptance criterion that can be read
off the source tree.

**No shim survives.** A shim that outlives the contract is a second address
for every name in the package, and the layout of #4 stops being the layout.

**The surface is explicit.** ``__init__`` may not star-import. A star import
over a flat module bound whatever that module imported — ``dawis.np``,
``dawis.pdb``, ``dawis.h5py`` were all reachable — so the public surface was
whatever the imports happened to leave lying around.

**The layout is the one specced.** One subpackage per block, and only the
shared vocabulary at the top level.

The dependency rules over that layout are checked in
``test_pipelines_layout.py`` and ``test_viz.py``, and #58 generalises them.
"""

from __future__ import annotations

import ast
import types
from pathlib import Path

import dawis

PACKAGE_ROOT = Path(dawis.__file__).resolve().parent

#: The flat modules the carve emptied. Each held a shim from the ticket that
#: moved its code until this one; none may be left.
SHIMS = {
    'anscombe_transforms',
    'atrous',
    'datacube',
    'deconvolution',
    'detect_and_deblend',
    'gif',
    'hard_threshold',
    'inpainting',
    'label_regions',
    'make_interscale_trees',
    'make_regions',
    'restore_objects',
    'store_objects',
    'synthesis_by_analysis',
    'table_noise_stds',
    'wavelet_transforms',
}

#: The blocks of #4. ``selection`` is reserved and empty; ``viz`` is real but
#: deliberately not imported by the package.
BLOCKS = {
    'detection',
    'io',
    'pipelines',
    'restoration',
    'selection',
    'trees',
    'viz',
    'wavelets',
}

#: What may sit at the top level beside the blocks: the vocabulary every block
#: shares, and the two seams that are not blocks. ``config`` joined them with
#: #53 and ``results`` with #54; ``cli`` of #4 lands in Phase B.
TOP_LEVEL_MODULES = {
    '__init__',
    'config',
    'convergence',
    'cubes',
    'exceptions',
    'parallel',
    'results',
}


def test_no_shim_module_is_left():
    """Every old flat address is gone, not merely emptied."""
    survivors = sorted(
        name for name in SHIMS if (PACKAGE_ROOT / ('%s.py' % name)).exists()
    )

    assert survivors == [], (
        'these shims outlived the contract, so every name in them has two '
        'addresses: %s' % ', '.join(survivors)
    )


def test_the_package_init_holds_no_star_import():
    """The surface is named, not accumulated."""
    tree = ast.parse((PACKAGE_ROOT / '__init__.py').read_text())
    starred = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == '*' for alias in node.names)
    ]

    assert starred == [], (
        'dawis/__init__.py star-imports %s, so its public surface is whatever '
        'those modules happen to import' % ', '.join(starred)
    )


def test_the_export_list_names_exactly_the_public_surface():
    """``__all__`` and what ``import dawis`` binds are the same set.

    Both directions matter. A name in ``__all__`` that is not bound breaks
    ``from dawis import *``; a public name bound but not listed is surface the
    package never meant to promise — which is how ``dawis.np`` happened.
    """
    listed = set(dawis.__all__)
    bound = {
        name
        for name, value in vars(dawis).items()
        if not name.startswith('_') and not isinstance(value, types.ModuleType)
    }

    assert sorted(listed - bound) == [], 'listed in __all__ but not bound'
    assert sorted(bound - listed) == [], 'bound on the package but not in __all__'


def test_no_third_party_module_is_bound_on_the_package():
    """The leaks the star imports carried are gone.

    Submodules of ``dawis`` are bound by the import machinery and are meant to
    be; anything else is a library that leaked in through someone's imports.
    """
    leaked = sorted(
        name
        for name, value in vars(dawis).items()
        if isinstance(value, types.ModuleType)
        and not value.__name__.startswith('dawis')
    )

    assert leaked == [], (
        'these libraries are reachable as dawis.<name>: %s' % ', '.join(leaked)
    )


def test_the_layout_is_the_one_specced():
    """One package per block, and only shared vocabulary at the top."""
    packages = {
        path.parent.name
        for path in PACKAGE_ROOT.glob('*/__init__.py')
    }
    modules = {path.stem for path in PACKAGE_ROOT.glob('*.py')}

    assert packages == BLOCKS
    assert modules == TOP_LEVEL_MODULES
