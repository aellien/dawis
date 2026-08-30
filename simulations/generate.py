#!/usr/bin/env python
"""Generate entries of the simulation gallery (#69, spec #67).

A hand-run script, like ``benchmarks/run_benchmark.py``. It is not a pytest
tier and it is not run in CI. It runs inside the ``dawis-sim`` conda
environment, which is the only place GalSim is installed; DAWIS itself never
runs there.

One entry is a simulated observation, both truth images, the PSF kernel, the
per-source parameter table and the provenance manifest, all in one
self-describing directory under the data root.

Usage
-----
Create the environment once::

    conda env create -f simulations/environment.yml

Build one entry::

    conda run -n dawis-sim python simulations/generate.py \\
        --tier full --class sersic_n4 --regime balanced --realisation 0

Build every entry the recipe knows how to build::

    conda run -n dawis-sim python simulations/generate.py --all

Build somewhere else for inspection, and rebuild after a recipe change::

    conda run -n dawis-sim python simulations/generate.py --all \\
        --out /tmp/gallery --force

Resuming, and staleness
-----------------------
Generation is idempotent. An entry whose directory already holds a manifest
matching the recipe is skipped, so an interrupted ``--all`` resumes where it
stopped rather than starting over.

An entry whose manifest describes a *different* recipe is **stale**, and the
generator stops rather than skipping it or overwriting it. Skipping would
leave last month's ground truth on disk under a name that says it is this
month's; overwriting would be a rebuild nobody asked for. ``--force`` is how a
rebuild is asked for.

Selecting entries
-----------------
Without ``--all``, all four axes are required and name exactly one entry.
With ``--all``, any axis given is held fixed and the rest are swept, so
``--all --tier small`` builds every known entry of the cheap tier.

``--all`` builds every entry the *recipe* knows about, which today is fewer
than the 72 #67 specifies: the rest need axis values no ticket has decided,
and the recipe refuses them rather than guessing. Each ticket that decides an
axis lifts a refusal, and ``--all`` grows with it — #70 lifted two class
refusals and the sweep went from two entries to six on its own, and #71 lifted
the last one and took it to eight.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

# Run as a script rather than imported as a module, so the package that holds
# the three layers has to be findable from the file's own location.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulations import recipe as recipe_module  # noqa: E402
from simulations.recipe import EntryAxes, Recipe  # noqa: E402
from simulations.write import EntryState, entry_state, write_entry  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - the annotation only, never at runtime
    from simulations.render import RenderedEntry

#: The axis options that name one entry, in the order #67 lists them, paired
#: with the command-line flag each is spelled with.
AXIS_FLAGS = (
    ('tier', '--tier'),
    ('source_class', '--class'),
    ('regime', '--regime'),
    ('realisation', '--realisation'),
)


class StaleEntryError(RuntimeError):
    """An entry on disk was drawn from a recipe that no longer exists.

    Raised rather than silently rebuilt, because #67 asks for a rebuild to be
    possible and never accidental. The images beside the mismatched manifest
    are still readable, and a researcher may have numbers recorded against
    them; deciding to discard them belongs to a human with ``--force``.
    """


def _render_entry(recipe: Recipe) -> RenderedEntry:
    """Draw one entry with GalSim.

    Imported here rather than at module level so that ``--help``, the argument
    errors and the skip all work in any environment. Everything past this
    import needs ``dawis-sim``.
    """
    from simulations.render import render

    return render(recipe)


def build_entry(
    axes: EntryAxes,
    root: Path,
    force: bool = False,
    render: Callable[[Recipe], RenderedEntry] | None = None,
) -> tuple[Path, bool]:
    """Build one entry, or skip it if a matching one is already on disk.

    Returns the entry directory and whether it was actually built, so a caller
    reporting the outcome never has to re-derive the entry's name.

    ``render`` exists so the resume and rebuild decisions can be exercised
    without GalSim; leave it unset and the real renderer is imported lazily.
    """
    recipe = recipe_module.build_recipe(**asdict(axes))
    directory = recipe.entry_directory(root)

    if not force:
        state = entry_state(recipe, root)
        if state is EntryState.BUILT:
            return directory, False
        if state is EntryState.STALE:
            raise StaleEntryError(
                '%s holds a manifest that does not describe this entry: it is '
                'unreadable, or it was written by a recipe that has since '
                'changed. Rebuild it with --force, or move it aside to keep '
                'it.' % directory
            )

    rendered = (render or _render_entry)(recipe)
    return write_entry(recipe, rendered, root), True


def select_entries(
    build_all: bool,
    tier: str | None,
    source_class: str | None,
    regime: str | None,
    realisation: int | None,
) -> tuple[EntryAxes, ...]:
    """The entries one invocation asks for.

    With ``--all`` the given axes are filters over what the recipe can build.
    Without it the four axes name one entry, which is returned whether the
    recipe can build it or not — so that an undecided axis is refused by
    :func:`~simulations.recipe.build_recipe` with the message that says which
    ticket owns it, rather than silently selecting nothing.
    """
    if build_all:
        return recipe_module.built_entries(
            tier=tier,
            source_class=source_class,
            regime=regime,
            realisation=realisation,
        )
    return (EntryAxes(tier, source_class, regime, realisation),)


def _report_built(directory: Path, with_files: bool) -> None:
    print('built %s' % directory)
    if not with_files:
        return
    for name in recipe_module.ENTRY_FILE_NAMES:
        path = directory / name
        print('  %-20s %8.1f kB' % (name, path.stat().st_size / 1024))


def build_parser() -> argparse.ArgumentParser:
    """The command-line surface #67 specifies, in one place a test can read."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='build every entry the recipe knows about, narrowed by any axis given',
    )
    parser.add_argument(
        '--tier', choices=sorted(recipe_module.TIERS), help='image size and grid shape'
    )
    parser.add_argument(
        '--class',
        dest='source_class',
        choices=recipe_module.SOURCE_CLASSES,
        help='source class',
    )
    parser.add_argument(
        '--regime',
        choices=recipe_module.KNOWN_REGIME_NAMES,
        help='noise regime',
    )
    parser.add_argument(
        '--realisation',
        type=int,
        help='noise realisation index',
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if not arguments.all:
        missing = [
            flag for name, flag in AXIS_FLAGS if getattr(arguments, name) is None
        ]
        if missing:
            parser.error(
                'either --all, or all four axes: %s missing' % ', '.join(missing)
            )

    entries = select_entries(
        build_all=arguments.all,
        tier=arguments.tier,
        source_class=arguments.source_class,
        regime=arguments.regime,
        realisation=arguments.realisation,
    )
    if not entries:
        parser.exit(
            2,
            'generate.py: nothing to build for that selection. The recipe '
            'refuses the axis values no ticket has decided, and --all sweeps '
            'only what it can build; the ticket that decides them adds them.\n',
        )

    # #68's report, unchanged for the one-entry command it was written for.
    # A full sweep prints one line per entry and a tally instead.
    list_files = len(entries) == 1

    built = 0
    skipped = 0
    for axes in entries:
        try:
            directory, was_built = build_entry(
                axes, root=arguments.out, force=arguments.force
            )
        except (
            recipe_module.AxisNotBuiltError,
            recipe_module.GridTooTightError,
            recipe_module.PairsTooCloseError,
            recipe_module.SourceClippedError,
            recipe_module.TruncationTooTightError,
            StaleEntryError,
        ) as error:
            parser.exit(2, 'generate.py: %s\n' % error)

        if was_built:
            built += 1
            _report_built(directory, with_files=list_files)
        else:
            skipped += 1
            print('already built, skipping: %s' % directory)

    if len(entries) > 1:
        print('%d built, %d already present' % (built, skipped))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
