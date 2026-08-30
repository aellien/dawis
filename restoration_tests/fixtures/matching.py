"""Which injected source a tree is trying to reconstruct.

The rule of #75: a tree is matched to a source when its interscale maximum
falls inside that source's half-light radius. Detection performance is not
under test in this map, so this is a lookup against known positions rather
than a detection benchmark — an unmatched tree is dropped and counted, not
scored.

Two coordinate conventions meet here, and they are transposes of each other.

``sources.csv`` measures a source from the lower-left corner of the first
pixel, with ``x`` along the columns and ``y`` down the rows — GalSim's way of
seeing an image. A DAWIS region measures its peak as ``x_max``, ``y_max``,
and those are a numpy **row** and **column**: ``make_regions_full_props``
reads them straight out of ``skimage``'s ``coords``, which are ``(row,
col)``. So a source's ``x`` is a tree's ``y_max``, and a source's ``y`` is a
tree's ``x_max``.

Nothing raises if the two are swapped. The gallery lays its sources on a
grid, and a transposed grid is still a grid, so a swapped axis quietly
matches a tree to a different source of the same class and the same
brightness — which is exactly the fixture set a metric could not tell from a
correct one. :func:`match_tree` therefore takes its arguments named ``row``
and ``column`` rather than ``x`` and ``y``, so that neither name means two
things anywhere in this package.

They also differ by half a pixel. A source coordinate is measured from a
pixel *corner*, so index ``c`` holds coordinates ``[c, c + 1)`` and its
centre is at ``c + 0.5`` — verified against the gallery, where for every
point source ``floor(x)`` is the column of the one non-zero pixel of
``truth_preconv``. A peak's *index* compared against a source's *coordinate*
is therefore a fixed half-pixel bias on each axis, 0.71 px radially. Against
an extended source's ten-pixel radius that is noise; against the three pixels
a point source gets it is a quarter of the budget, spent systematically, and
it decides borderline point fixtures. So the comparison is made between two
centres.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    'AMBIGUOUS',
    'MATCHED',
    'UNMATCHED',
    'GallerySource',
    'Match',
    'match_radius_pixels',
    'match_tree',
]

#: A tree whose interscale maximum falls inside exactly one source.
MATCHED = 'matched'
#: A tree whose interscale maximum falls inside no source at all.
UNMATCHED = 'unmatched'
#: A tree whose interscale maximum falls inside two or more.
AMBIGUOUS = 'ambiguous'


@dataclass(frozen = True)
class GallerySource:
    """One row of a gallery entry's ``sources.csv`` — one injected source.

    The nine columns every class carries. ``blend`` carries three more, and
    is out of this pass: #75 scores single sources first.

    Attributes
    ----------
    source_id : int
        The row's identifier within its entry.
    x, y : float
        Position in pixels, measured from the lower-left corner of the first
        pixel. ``x`` runs along the **columns** and ``y`` down the **rows**.
    magnitude, flux : float
        Total brightness, against the gallery's zero point and in ADU.
    sersic_index, half_light_radius : float
        Profile shape. The radius is in **arcseconds**; both are zero for a
        point source, which has neither.
    e1, e2 : float
        True reduced-shear ellipticity components, post-PSF.
    """

    source_id: int
    x: float
    y: float
    magnitude: float
    flux: float
    sersic_index: float
    half_light_radius: float
    e1: float
    e2: float


@dataclass(frozen = True)
class Match:
    """The outcome of matching one tree, and the source if there was one.

    Attributes
    ----------
    outcome : str
        One of :data:`MATCHED`, :data:`UNMATCHED` or :data:`AMBIGUOUS`.
    source : GallerySource or None
        The matched source, and None for the other two outcomes.
    """

    outcome: str
    source: GallerySource | None


def match_radius_pixels(
    source: GallerySource,
    pixel_scale: float,
    psf_fwhm: float,
) -> float:
    """How close an interscale maximum has to be to count as on the source.

    #75 says the half-light radius, and for the two Sérsic classes that is
    what this returns. A point source is the case the rule does not cover: a
    delta function's half-light radius is zero — literally true, since it
    encloses all of its light at every radius — and no interscale maximum
    could ever fall inside a radius of zero, so every point fixture would be
    dropped as unmatched.

    The PSF is the floor, because the light a point source actually puts in
    the image is the PSF's, and its full width at half maximum is the width
    over which that light is spread. It is a floor rather than a term added
    to the radius: for the extended classes the half-light radius is already
    several times the PSF, and widening it further would start reaching for
    neighbours.

    Parameters
    ----------
    source : GallerySource
        The injected source.
    pixel_scale : float
        Arcseconds per pixel, from the entry's manifest.
    psf_fwhm : float
        PSF full width at half maximum in arcseconds, from the same manifest.

    Returns
    -------
    float
        The radius in pixels.
    """

    return max(source.half_light_radius, psf_fwhm) / pixel_scale


def match_tree(
    peak_row: int,
    peak_col: int,
    sources: list[GallerySource],
    pixel_scale: float,
    psf_fwhm: float,
) -> Match:
    """Match one tree's interscale maximum against the injected sources.

    Parameters
    ----------
    peak_row, peak_col : int
        The interscale maximum's pixel, as ``x_max`` and ``y_max`` of the
        root region — a numpy row and a numpy column, in that order. See the
        module docstring for why they are not called ``x`` and ``y`` here.
    sources : list of GallerySource
        Every source injected into the entry.
    pixel_scale, psf_fwhm : float
        Passed to :func:`match_radius_pixels`.

    Returns
    -------
    Match
        :data:`MATCHED` and the source when exactly one contains the maximum.
        :data:`AMBIGUOUS` and None when more than one does — two sources
        sharing a maximum is a blend under another name, and #75 scores
        single sources, so the tree is refused rather than assigned to the
        nearer of the two. :data:`UNMATCHED` and None when none does.
    """

    inside = [
        source for source in sources
        if (peak_col + 0.5 - source.x) ** 2 + (peak_row + 0.5 - source.y) ** 2
        <= match_radius_pixels(source, pixel_scale, psf_fwhm) ** 2
    ]

    if len(inside) == 1:
        return Match(outcome = MATCHED, source = inside[0])
    if len(inside) > 1:
        return Match(outcome = AMBIGUOUS, source = None)
    return Match(outcome = UNMATCHED, source = None)
