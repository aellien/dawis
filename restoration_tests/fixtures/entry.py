"""One simulation-gallery entry, read off disk.

The gallery (#67-#72) states its layout as a contract, and this is the one
place in the fixture layer that depends on it: six files under
``<data-root>/simulations/<tier>/<class>_<regime>_r<N>/``. The fixture layer
reads four of them — the image DAWIS runs on, the PSF-convolved truth it is
scored against, the source table and the manifest.

``truth_conv.fits`` and not ``truth_preconv.fits``, because #74 scopes this
map to the reconstruction of *observed* sources: PSF deconvolution as a
science goal stays out on #1. What a fixture asks is whether the solver
recovered the source as the telescope saw it, so the truth it carries is the
one with the PSF already in it and only the noise taken out.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from restoration_tests.fixtures.matching import GallerySource

__all__ = ['GalleryEntry']


@dataclass(frozen = True)
class GalleryEntry:
    """The four files of one gallery entry that the fixture layer reads.

    Attributes
    ----------
    name : str
        ``<tier>/<class>_<regime>_r<N>``, the entry's address within the
        gallery and the name a fixture records itself against.
    path : pathlib.Path
        The directory it was read from.
    image : numpy.ndarray
        ``image.fits`` — the simulated observation, sky left in, which is
        what the mock iteration runs on.
    truth_conv : numpy.ndarray
        ``truth_conv.fits`` — the sources convolved with the PSF and no
        noise. What a reconstruction is scored against.
    sources : list of GallerySource
        Every row of ``sources.csv``.
    pixel_scale, psf_fwhm : float
        Arcseconds per pixel, and the PSF width in arcseconds. The two
        numbers :func:`~restoration_tests.fixtures.matching.match_radius_pixels`
        needs.
    manifest : dict
        The whole of ``manifest.json``, carried so a fixture can record which
        gallery entry it came from without a second read.
    """

    name: str
    path: Path
    image: np.ndarray
    truth_conv: np.ndarray
    sources: list[GallerySource]
    pixel_scale: float
    psf_fwhm: float
    manifest: dict

    @classmethod
    def read(cls, path: str | Path) -> GalleryEntry:
        """Read one entry directory.

        Parameters
        ----------
        path : str or pathlib.Path
            The entry directory, ``<tier>/<class>_<regime>_r<N>``.

        Returns
        -------
        GalleryEntry

        Raises
        ------
        ValueError
            If ``image.fits`` and ``truth_conv.fits`` do not share a shape.

            If the entry is of the ``blend`` class. #75 scores single sources
            first and #74 leaves blends to a later pass, so a blend entry is
            refused here rather than filtered out of the sweep — pointing the
            builder straight at one then fails the same way it would inside a
            sweep, with a sentence saying why.
        """

        path = Path(path)
        manifest = json.loads((path / 'manifest.json').read_text())

        if manifest['class'] == 'blend':
            raise ValueError(
                '%s is a blend entry --> #75 scores single sources, and the '
                'blend class is left to a later pass of #74' % path)

        with open(path / 'sources.csv', newline = '') as table:
            sources = [
                GallerySource(
                    source_id = int(row['source_id']),
                    x = float(row['x']),
                    y = float(row['y']),
                    magnitude = float(row['magnitude']),
                    flux = float(row['flux']),
                    sersic_index = float(row['sersic_index']),
                    half_light_radius = float(row['half_light_radius']),
                    e1 = float(row['e1']),
                    e2 = float(row['e2']),
                )
                for row in csv.DictReader(table)
            ]

        image = np.asarray(fits.getdata(path / 'image.fits'), dtype = float)
        truth_conv = np.asarray(fits.getdata(path / 'truth_conv.fits'),
                                dtype = float)

        # A tree's bounding box is clamped to the cube built from the image,
        # and the truth is then cut at that box. A truth image of another
        # shape gives a cutout smaller than the tubes, and the mismatch
        # surfaces much later inside a metric -- or does not surface at all,
        # if it broadcasts. Refusing it here names the file that is wrong.
        if image.shape != truth_conv.shape:
            raise ValueError(
                '%s: image.fits is %s and truth_conv.fits is %s --> a '
                'fixture cuts the truth at the tree box the image fixed, so '
                'the two must share a shape'
                % (path, image.shape, truth_conv.shape))

        return cls(
            name = '%s/%s' % (manifest['tier'], path.name),
            path = path,
            image = image,
            truth_conv = truth_conv,
            sources = sources,
            pixel_scale = float(manifest['pixel_scale']),
            psf_fwhm = float(manifest['psf_fwhm']),
            manifest = manifest,
        )
