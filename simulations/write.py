"""The write layer — one entry's arrays to disk (#68).

Thin by design, like the render layer. It puts the six files of #67's storage
contract into the entry directory and owns the idempotent skip.

One rule shapes every FITS header here: **the image DAWIS reads says nothing
about the noise**. #67 is explicit that DAWIS estimates the noise itself, as
it would on real data, and is told nothing. The true sky level, read noise and
gain go in ``manifest.json``, for a human reading a result. They do not go in
``image.fits``, where a run could pick them up.

``manifest.json`` is written last. It is the file the skip keys on, so an
interrupted generation leaves an entry that is visibly unfinished rather than
one that is silently half-built.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from astropy.io import fits

from simulations.recipe import MANIFEST_FILE_NAME, Recipe
from simulations.render import RenderedEntry

__all__ = ['SOURCE_COLUMNS', 'is_already_built', 'source_commit', 'write_entry']

#: The columns of ``sources.csv``, in the order #67 lists them. The blend
#: class adds companion id, separation and flux ratio; #68 does not build that
#: class, so those columns arrive with the ticket that does.
SOURCE_COLUMNS = (
    'source_id',
    'x',
    'y',
    'magnitude',
    'flux',
    'sersic_index',
    'half_light_radius',
    'e1',
    'e2',
)


def source_commit(repository_root: Path | None = None) -> str:
    """The commit the generator was run from, or a marker if that is unknown.

    #67 wants a regeneration months later to be reproducible rather than a
    guess. A dirty tree is recorded as such: an entry generated from uncommitted
    code is not reproducible from a commit, and saying so is more useful than
    recording the commit it was nearly built from.
    """
    root = repository_root or Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'

    return '%s-dirty' % commit if dirty else commit


def is_already_built(recipe: Recipe, root: Path | str) -> bool:
    """Whether this entry's manifest is on disk, so generation can resume."""
    return recipe.output_paths(root)[MANIFEST_FILE_NAME].exists()


def _write_fits(path: Path, array: np.ndarray, header: dict) -> None:
    """One array to one FITS file, as float32, with a plain header.

    float32 rather than float64: it is what the packaged gallery image is, it
    is well inside the precision the noise makes meaningful, and it halves
    720 MB of gallery.
    """
    hdu = fits.PrimaryHDU(data=np.asarray(array, dtype=np.float32))
    for key, value in header.items():
        hdu.header[key] = value
    hdu.writeto(path, overwrite=True)


def _entry_header(recipe: Recipe, content: str) -> dict:
    """What every FITS file in the entry says about itself.

    Deliberately excludes the noise parameters. See the module docstring.
    """
    return {
        'CONTENT': (content, 'which of the entry files this is'),
        'SIMTIER': (recipe.tier, 'gallery tier'),
        'SIMCLASS': (recipe.source_class, 'source class'),
        'SIMREGIM': (recipe.regime, 'noise regime'),
        'SIMREAL': (recipe.realisation, 'noise realisation'),
        'PIXSCALE': (recipe.pixel_scale, 'arcsec per pixel'),
        'ORIGIN': ('dawis-sim gallery', 'see manifest.json'),
    }


def _write_sources(path: Path, recipe: Recipe) -> None:
    """One row per source — the ground truth #25's metrics read."""
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(SOURCE_COLUMNS)
        for source in recipe.sources:
            writer.writerow([getattr(source, name) for name in SOURCE_COLUMNS])


def write_entry(
    recipe: Recipe,
    rendered: RenderedEntry,
    root: Path | str,
    commit: str | None = None,
) -> Path:
    """Write the six files of one entry, and return its directory.

    ``manifest.json`` goes last, because it is what :func:`is_already_built`
    reads.
    """
    directory = recipe.entry_directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    paths = recipe.output_paths(root)

    for name, array, content in (
        ('image.fits', rendered.image, 'observation'),
        ('truth_preconv.fits', rendered.truth_preconv, 'truth-preconv'),
        ('truth_conv.fits', rendered.truth_conv, 'truth-conv'),
        ('psf.fits', rendered.psf, 'psf'),
    ):
        _write_fits(paths[name], array, _entry_header(recipe, content))

    _write_sources(paths['sources.csv'], recipe)

    manifest = recipe.manifest(
        galsim_version=rendered.galsim_version,
        source_commit=commit if commit is not None else source_commit(),
    )
    paths[MANIFEST_FILE_NAME].write_text(json.dumps(manifest, indent=2) + '\n')

    return directory
