#!/usr/bin/env python
"""Generate one entry of the simulation gallery (#68, spec #67).

A hand-run script, like ``benchmarks/run_benchmark.py``. It is not a pytest
tier and it is not run in CI. It runs inside the ``dawis-sim`` conda
environment, which is the only place GalSim is installed; DAWIS itself never
runs there.

One invocation builds one entry: a simulated observation, both truth images,
the PSF kernel, the per-source parameter table and the provenance manifest,
all in one self-describing directory under the data root.

Usage
-----
Create the environment once::

    conda env create -f simulations/environment.yml

Build the entry #68 delivers::

    conda run -n dawis-sim python simulations/generate.py \\
        --tier full --class sersic_n4 --regime balanced --realisation 0

Build it somewhere else for inspection, and rebuild it after a recipe change::

    conda run -n dawis-sim python simulations/generate.py \\
        --tier full --class sersic_n4 --regime balanced --realisation 0 \\
        --out /tmp/gallery --force

Scope
-----
#68 is the tracer bullet. ``--all``, which #67 specifies as the way to build
all 72 entries, is not wired up here: 71 of those 72 entries need axis values
no ticket has decided yet, and the recipe refuses them. The flag arrives with
the ticket that decides them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run as a script rather than imported as a module, so the package that holds
# the three layers has to be findable from the file's own location.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulations import recipe as recipe_module  # noqa: E402
from simulations.write import is_already_built, write_entry  # noqa: E402


def build_entry(
    tier: str,
    source_class: str,
    regime: str,
    realisation: int,
    root: Path,
    force: bool = False,
) -> tuple[Path, bool]:
    """Build one entry, or skip it if it is already on disk.

    Returns the entry directory and whether it was actually built, so a caller
    reporting the outcome never has to re-derive the entry's name.

    GalSim is imported inside the function rather than at module level so that
    ``--help`` and the argument errors work in any environment. Everything past
    that import needs ``dawis-sim``.
    """
    recipe = recipe_module.build_recipe(
        tier=tier,
        source_class=source_class,
        regime=regime,
        realisation=realisation,
    )
    directory = recipe.entry_directory(root)

    if is_already_built(recipe, root) and not force:
        return directory, False

    from simulations.render import render

    rendered = render(recipe)
    return write_entry(recipe, rendered, root), True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--tier',
        required=True,
        choices=sorted(recipe_module.TIERS),
        help='image size and grid shape',
    )
    parser.add_argument(
        '--class',
        dest='source_class',
        required=True,
        choices=recipe_module.SOURCE_CLASSES,
        help='source class',
    )
    parser.add_argument(
        '--regime',
        required=True,
        choices=recipe_module.KNOWN_REGIME_NAMES,
        help='noise regime',
    )
    parser.add_argument(
        '--realisation', required=True, type=int, help='noise realisation index'
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=recipe_module.default_output_root(),
        help='output root (default: %(default)s)',
    )
    parser.add_argument(
        '--force', action='store_true', help='rebuild an entry that already exists'
    )
    arguments = parser.parse_args(argv)

    try:
        directory, built = build_entry(
            tier=arguments.tier,
            source_class=arguments.source_class,
            regime=arguments.regime,
            realisation=arguments.realisation,
            root=arguments.out,
            force=arguments.force,
        )
    except (
        recipe_module.AxisNotBuiltError,
        recipe_module.GridTooTightError,
        recipe_module.SourceClippedError,
    ) as error:
        parser.exit(2, 'generate.py: %s\n' % error)

    if not built:
        print('already built, skipping: %s' % directory)
        return 0

    print('built %s' % directory)
    for name in recipe_module.ENTRY_FILE_NAMES:
        path = directory / name
        print('  %-20s %8.1f kB' % (name, path.stat().st_size / 1024))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
