"""Turning one gallery entry into fixtures, and the sweep over the gallery.

Three things happen here, in this order, and they are separate because only
the first is decision-free.

**Match.** Every tree the mock iteration built is looked up against the
entry's injected sources. A tree that matches exactly one becomes a fixture;
a tree that matches none or two is dropped and counted. Detection performance
is not under test in #74, so a dropped tree is a fact about the dump rather
than a failure of it — but the counts are recorded, because they are the
number that says whether a fixture set is about the sources or about
detection.

**Cut.** The tubes come off the cubes at the tree's own bounding box, and the
truth is cut to the same box. Every source that falls inside it goes into the
cutout, not only the matched one: what is being scored is a patch of sky, and
the truth of that patch includes whatever else is in it.

**Reconstruct once, with the published default.** Every fixture carries what
:func:`~dawis.restoration.synthesis.restore_object` made of its tree under
the pipeline's own defaults. That is the "current default arm" #75 wants the
quicklook to draw, and it is the baseline #77 reads every other arm against.
It is recorded, not targeted: the whole point of #74 is that this arm may be
the wrong one.

The sweep
---------
``--all`` walks the gallery the way ``simulations/generate.py`` does: any axis
given is held fixed and the rest are swept. Blend entries are skipped, since
#75 scores single sources first. The sweep is resumable per entry — an entry
whose fixture directory already holds a manifest is left alone unless
``--force`` is given — because a dump of the whole gallery is minutes of work
that a killed run should not have to repeat.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from dawis.config import Config
from dawis.restoration.synthesis import restore_object
from restoration_tests.fixtures.entry import GalleryEntry
from restoration_tests.fixtures.matching import (
    AMBIGUOUS,
    MATCHED,
    UNMATCHED,
    Match,
    match_tree,
)
from restoration_tests.fixtures.mock import (
    FROZEN_SETTINGS,
    MockIteration,
    MockSettings,
    run_mock_iteration,
)
from restoration_tests.fixtures.store import (
    MANIFEST_FILE,
    Fixture,
    TreeGeometry,
    write_fixtures,
)

__all__ = ['DEFAULT_GALLERY_ROOT', 'DEFAULT_OUT_ROOT', 'already_built',
           'build_fixtures', 'build_entry', 'main', 'selects', 'sweep']

log = logging.getLogger(__name__)

#: Where ``simulations/generate.py`` writes the gallery by default.
DEFAULT_GALLERY_ROOT = Path.home() / 'dawis-data' / 'simulations'
#: Where the battery's outputs live. #74 puts everything under
#: ``~/dawis-tests/``.
DEFAULT_OUT_ROOT = Path.home() / 'dawis-tests' / 'restoration-fixtures'

#: The pipeline's own restoration defaults, read off :class:`dawis.config.Config`
#: rather than written down again, so the reference arm is the published one
#: even if the published one moves.
_DEFAULTS = Config()


def _reference_reconstruction(tree, iteration: MockIteration) -> tuple[np.ndarray, dict]:
    """What the published default arm makes of one tree."""

    image, _, filter_kw, flag_convergence, sum_wr, norm_wr, algorithm, \
        synthesis_operator = restore_object(
            tree, iteration.wavelet_cube, iteration.label_cube,
            extent_sep = _DEFAULTS.extent_sep,
            ecc_sep = _DEFAULTS.ecc_sep,
            lvl_sep_lin = _DEFAULTS.lvl_sep_lin,
            lvl_sep_big = _DEFAULTS.lvl_sep_big,
            lvl_sep_op = _DEFAULTS.lvl_sep_op,
            deconv = _DEFAULTS.deconv,
            cg_max_iter = _DEFAULTS.cg_max_iter,
        )

    return image, {
        'algorithm': str(algorithm),
        'filter': str(filter_kw),
        'synthesis_operator': str(synthesis_operator),
        'deconv': bool(_DEFAULTS.deconv),
        'cg_max_iter': int(_DEFAULTS.cg_max_iter),
        'flag_convergence': bool(flag_convergence),
        'sum_wr': float(sum_wr),
        'norm_wr': float(norm_wr),
    }


def build_fixtures(
    entry: GalleryEntry,
    iteration: MockIteration,
    settings: MockSettings = FROZEN_SETTINGS,
) -> tuple[list[Fixture], dict]:
    """Match, cut and reconstruct one entry's trees.

    Parameters
    ----------
    entry : GalleryEntry
        The gallery entry the iteration ran on.
    iteration : MockIteration
        What :func:`~restoration_tests.fixtures.mock.run_mock_iteration` made
        of ``entry.image``.
    settings : MockSettings, optional
        Carried only so the caller and this function cannot disagree about
        which settings produced the trees. Default :data:`FROZEN_SETTINGS`.

    Returns
    -------
    fixtures : list of Fixture
        One per matched tree, numbered in the order the trees came back.
    match_counts : dict
        How many trees ended in each of the three matching outcomes. Every
        outcome appears, zero included, so a reader never has to tell a count
        of zero from a key nobody wrote.
    """

    counts = {MATCHED: 0, UNMATCHED: 0, AMBIGUOUS: 0}
    fixtures = []

    for tree in iteration.trees:
        maximum = tree.interscale_maximum
        peak_row, peak_col = int(maximum.x_max), int(maximum.y_max)

        match: Match = match_tree(peak_row, peak_col, entry.sources,
                                  pixel_scale = entry.pixel_scale,
                                  psf_fwhm = entry.psf_fwhm)
        counts[match.outcome] += 1
        if match.outcome != MATCHED:
            continue

        wavelet_tube, support_tube, label_tube = tree.tubes(
            iteration.wavelet_cube, iteration.label_cube)
        row_min, col_min, row_max, col_max = (int(edge) for edge in tree.bbox)
        reconstruction, arm = _reference_reconstruction(tree, iteration)

        fixtures.append(Fixture(
            fixture_id = '%s/tree%03d' % (entry.name, len(fixtures)),
            entry = entry.name,
            source = match.source,
            geometry = TreeGeometry(
                peak_row = peak_row,
                peak_col = peak_col,
                level = int(maximum.level),
                bbox = (row_min, col_min, row_max, col_max),
                extent = float(tree.extent),
                eccentricity = float(tree.eccentricity),
            ),
            wavelet_tube = wavelet_tube,
            support_tube = support_tube,
            label_tube = label_tube,
            truth_cutout = np.ascontiguousarray(
                entry.truth_conv[row_min:row_max, col_min:col_max]),
            reference_reconstruction = np.ascontiguousarray(reconstruction),
            reference_arm = arm,
        ))

    return fixtures, counts


def build_entry(
    entry_path: str | Path,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    settings: MockSettings = FROZEN_SETTINGS,
    quicklook: bool = True,
) -> Path:
    """Dump one gallery entry's fixtures, and draw their quicklooks.

    Parameters
    ----------
    entry_path : str or pathlib.Path
        A gallery entry directory.
    out_root : str or pathlib.Path, optional
        Root of the fixture store. Default :data:`DEFAULT_OUT_ROOT`.
    settings : MockSettings, optional
        Default :data:`FROZEN_SETTINGS`.
    quicklook : bool, optional
        Draw the panels. Default True. Turned off when matplotlib is not
        installed, or when only the arrays are wanted.

    Returns
    -------
    pathlib.Path
        The directory written.
    """

    entry = GalleryEntry.read(entry_path)
    log.info('%s: running the mock iteration', entry.name)

    iteration = run_mock_iteration(entry.image, settings)
    fixtures, counts = build_fixtures(entry, iteration, settings)
    log.info('%s: %d trees --> %s', entry.name, len(iteration.trees), counts)

    directory = Path(out_root) / entry.name

    # The panels go first, and the manifest last. The manifest is what the
    # sweep's resume reads, so an entry that has one is an entry the sweep
    # will not come back to -- and matplotlib is an optional extra, so a
    # dump on an install without it would otherwise write the manifest,
    # raise from the drawing, and leave behind an entry marked built whose
    # panels do not exist and never will. Same order the gallery generator
    # uses, and for the same reason.
    if quicklook:
        from restoration_tests.fixtures.quicklook import draw_quicklooks

        # Cleared rather than overwritten: a rebuild that yields fewer
        # fixtures than the last one would otherwise leave panels behind
        # that no fixture corresponds to, and a rebuild that yields none
        # would leave the whole previous set standing.
        panels = directory / 'quicklook'
        if panels.exists():
            shutil.rmtree(panels)
        if fixtures:
            draw_quicklooks(fixtures, panels)

    write_fixtures(directory, fixtures, entry_name = entry.name,
                   settings = settings, entry_manifest = entry.manifest,
                   match_counts = counts)

    return directory


def already_built(directory: str | Path, settings: MockSettings) -> bool:
    """Whether this entry is already on disk **under these settings**.

    The resume test used to be no more than "a manifest exists", and the
    output path carries the entry's name and nothing about the settings. So a
    second sweep at another ``tau`` skipped every entry and reported success,
    leaving the first set on disk under a manifest naming the first ``tau`` —
    the quietest possible way to measure every arm of #74 on fixtures nobody
    meant to build. ``--force`` was no better: it overwrote the first set in
    place, so the two could never coexist.

    Comparing the settings instead makes the two cases distinct. A rerun at
    the same settings resumes; a rerun at different ones rebuilds, and the
    manifest it leaves describes what is actually in the file beside it.

    Parameters
    ----------
    directory : str or pathlib.Path
        The entry's fixture directory.
    settings : MockSettings
        The settings this sweep is running under.

    Returns
    -------
    bool
        False if there is no manifest, if it cannot be read, or if it names
        different settings.
    """

    manifest_path = Path(directory) / MANIFEST_FILE
    if not manifest_path.exists():
        return False

    try:
        recorded = json.loads(manifest_path.read_text()).get('settings')
    except (OSError, ValueError):
        return False

    return recorded == settings.as_dict()


def selects(
    entry_manifest: dict,
    tier: str | None = None,
    entry_class: str | None = None,
    regime: str | None = None,
    realisation: int | None = None,
) -> bool:
    """Whether a sweep with these axes should build this entry.

    Any axis left as None is swept; any axis given is held fixed. The
    ``blend`` class is refused whatever the axes say — #75 scores single
    sources first, and #74 leaves the blend class to a later pass.

    The axes are read off the entry's own ``manifest.json``, which names all
    four outright. Taking them from the directory name instead is what broke
    the first real sweep: an entry is ``<class>_<regime>_r<N>``, the
    ``read_noise`` regime has an underscore in it, and splitting on
    underscores turns ``blend_read_noise_r0`` into a ``blend_read`` of regime
    ``noise``. The blend skip then missed it and the run stopped on a blend
    entry. Nothing here parses a directory name.

    Parameters
    ----------
    entry_manifest : dict
        The entry's ``manifest.json``.
    tier, entry_class, regime : str, optional
        Axis values to hold fixed. Default None sweeps the axis.
    realisation : int, optional
        The same, for the realisation index.

    Returns
    -------
    bool
    """

    if entry_manifest['class'] == 'blend':
        return False

    return all(
        wanted is None or entry_manifest[axis] == wanted
        for axis, wanted in (('tier', tier), ('class', entry_class),
                             ('regime', regime), ('realisation', realisation))
    )


def sweep(
    gallery_root: str | Path = DEFAULT_GALLERY_ROOT,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    tier: str | None = None,
    entry_class: str | None = None,
    regime: str | None = None,
    realisation: int | None = None,
    force: bool = False,
    settings: MockSettings = FROZEN_SETTINGS,
    quicklook: bool = True,
) -> list[Path]:
    """Dump every gallery entry the given axes select.

    Any axis left as None is swept; any axis given is held fixed. Blend
    entries are skipped whatever the axes say — #75 scores single sources
    first, and #74 leaves the blend class to a later pass.

    An entry already built **under these settings** is skipped, so a killed
    sweep resumes where it stopped. An entry built under different ones is
    rebuilt: see :func:`already_built`.

    Returns
    -------
    list of pathlib.Path
        The directories written, in the order they were built. An entry
        skipped because it was already built is not in the list.
    """

    gallery_root, out_root = Path(gallery_root), Path(out_root)
    written = []

    for tier_dir in sorted(gallery_root.iterdir()):
        if not tier_dir.is_dir():
            continue

        for entry_dir in sorted(tier_dir.iterdir()):
            manifest_path = entry_dir / 'manifest.json'
            if not manifest_path.exists():
                continue

            if not selects(json.loads(manifest_path.read_text()), tier = tier,
                           entry_class = entry_class, regime = regime,
                           realisation = realisation):
                continue

            target = out_root / tier_dir.name / entry_dir.name
            if not force and already_built(target, settings):
                log.info('%s/%s: already built under these settings, skipping',
                         tier_dir.name, entry_dir.name)
                continue

            written.append(build_entry(entry_dir, out_root, settings,
                                       quicklook = quicklook))

    return written


def main(argv: list[str] | None = None) -> int:
    """Command line: dump one entry, or sweep the gallery."""

    parser = argparse.ArgumentParser(
        description = 'Dump restoration fixtures from the simulation gallery.')
    parser.add_argument('--gallery', type = Path, default = DEFAULT_GALLERY_ROOT,
                        help = 'gallery root (default %s)' % DEFAULT_GALLERY_ROOT)
    parser.add_argument('--out', type = Path, default = DEFAULT_OUT_ROOT,
                        help = 'fixture root (default %s)' % DEFAULT_OUT_ROOT)
    parser.add_argument('--entry', type = Path,
                        help = 'one gallery entry directory; ignores the axes')
    parser.add_argument('--tier')
    parser.add_argument('--class', dest = 'entry_class')
    parser.add_argument('--regime')
    parser.add_argument('--realisation', type = int)
    parser.add_argument('--tau', type = float,
                        help = 'override the frozen tau. A dump under a '
                               'different tau is a different fixture set; the '
                               'manifest records which.')
    parser.add_argument('--force', action = 'store_true',
                        help = 'rebuild an entry that is already built')
    parser.add_argument('--no-quicklook', action = 'store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(level = logging.INFO, format = '%(message)s')

    settings = FROZEN_SETTINGS
    if args.tau is not None:
        settings = replace(settings, tau = args.tau)

    if args.entry is not None:
        written = [build_entry(args.entry, args.out, settings,
                               quicklook = not args.no_quicklook)]
    else:
        written = sweep(args.gallery, args.out, tier = args.tier,
                        entry_class = args.entry_class, regime = args.regime,
                        realisation = args.realisation, force = args.force,
                        settings = settings,
                        quicklook = not args.no_quicklook)

    print('%d entries written under %s' % (len(written), args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
