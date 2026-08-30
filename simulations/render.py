"""The render layer — a recipe handed to GalSim (#68, #71).

Thin by design. Every decision of #67 was made in :mod:`simulations.recipe`;
this module translates that description into GalSim objects, draws them, and
returns arrays. It decides nothing.

It imports GalSim, so it runs only inside the ``dawis-sim`` conda environment
and is never imported by the DAWIS test suite. That is why it is separate from
the recipe: the seam is what lets #67's decisions be tested without GalSim.

Three images come out of one recipe, all on the same pixel grid:

``truth_preconv``
    Sources only. No PSF, no noise.
``truth_conv``
    The same sources convolved with the PSF. Still no noise.
``image``
    ``truth_conv`` plus the sky level plus Poisson-Gaussian noise — the
    simulated observation, and the only one DAWIS ever reads. The sky is left
    in, so its median matches the gallery image the regime is anchored to.

The PSF kernel is returned beside them, so a deconvolution experiment has the
exact kernel that was applied rather than an approximation of it (#67).
"""

from __future__ import annotations

from dataclasses import dataclass

import galsim
import numpy as np

from simulations.recipe import Recipe, Source

__all__ = [
    'FLUX_TOLERANCE',
    'POINT_PLACEMENT_WINDOW',
    'FluxCheckError',
    'PointPlacementError',
    'RenderedEntry',
    'check_point_placement',
    'check_rendered_flux',
    'render',
]

#: Half-width of the stored PSF stamp, in units of the PSF FWHM. Wide enough
#: that a deconvolution has the wings it needs, narrow enough that the stamp
#: stays a small file.
PSF_STAMP_HALF_WIDTH_FWHM = 8.0

#: How far the drawn truth may sit from the recipe's analytic sum before the
#: generator refuses the entry.
#:
#: GalSim's own accuracy parameters, not the physics, set the floor here: the
#: default ``folding_threshold`` of 5e-3 lets a profile lose a fraction of a
#: per cent to the finite drawing bounds. One per cent leaves room for that
#: while still catching the mistakes this check exists for — a shear applied
#: to the wrong object, a flux normalisation dropped, a source drawn twice.
FLUX_TOLERANCE = 1e-2

#: Half-width, in pixels, of the window :func:`check_point_placement` compares
#: one point source's two pictures of itself in. Wide enough that a wrong
#: convention lands inside it and is reported rather than silently missed,
#: narrow enough to stay a small stamp.
#:
#: It does **not** have to clear a neighbouring source, and since #71 it could
#: not: a blend pair's tightest separation is well inside this window. The
#: check draws the source on its own instead, so nothing else can be in the
#: window to begin with.
POINT_PLACEMENT_WINDOW = 5


@dataclass(frozen=True)
class RenderedEntry:
    """The pixels of one entry, before anything has been written to disk."""

    image: np.ndarray
    truth_preconv: np.ndarray
    truth_conv: np.ndarray
    psf: np.ndarray
    galsim_version: str


class FluxCheckError(ValueError):
    """The drawn truth does not carry the flux the recipe intended.

    #67 puts this check on the generator rather than in a test on purpose: a
    rendering mistake should fail loudly at generation time, while a human is
    present and can act on it, rather than quietly producing a ground truth
    that is not the ground truth.
    """


def _psf(recipe: Recipe) -> galsim.GSObject:
    """The synthetic Moffat #67 specifies, at the recipe's fixed FWHM."""
    return galsim.Moffat(beta=recipe.psf_beta, fwhm=recipe.psf_fwhm)


def _profile(source: Source) -> galsim.GSObject:
    """One source as GalSim sees it: a truncated, sheared Sérsic.

    ``flux`` is the total flux of the *truncated* profile, which is what makes
    the recipe's analytic sum an exact statement rather than an approximate
    one. ``shear`` applies the true ``e1``/``e2`` the truth table records.

    A point source is a delta function instead, and is not sheared: it has no
    shape of its own, so the recipe records its true post-PSF ``e1``/``e2`` as
    zero and there is nothing here to apply.
    """
    if source.is_point:
        return galsim.DeltaFunction(flux=source.flux)

    profile = galsim.Sersic(
        n=source.sersic_index,
        half_light_radius=source.half_light_radius,
        trunc=source.truncation_radius,
        flux=source.flux,
    )
    return profile.shear(e1=source.e1, e2=source.e2)


def _blank(recipe: Recipe) -> galsim.Image:
    rows, columns = recipe.image_shape
    return galsim.ImageD(columns, rows, scale=recipe.pixel_scale)


def _source_centre(source: Source) -> galsim.PositionD:
    """The recipe's pixel coordinates in GalSim's one-based convention.

    The recipe measures from the lower-left *corner* of the first pixel, so
    the centre of that pixel is at 0.5. GalSim numbers pixel centres from 1.
    """
    return galsim.PositionD(source.x + 0.5, source.y + 0.5)


def _add_delta(canvas: galsim.Image, source: Source) -> None:
    """Put a point source's whole flux in the pixel that contains it.

    GalSim refuses to draw a bare delta function, and it is right to: a delta
    has no scale, so there are no drawing bounds to choose and no accuracy
    question to answer. The pre-convolution truth of a point source is not an
    image that has to be rendered — it is the statement that one pixel holds
    all the flux and every other pixel holds none, which is what this does.
    The *convolved* truth beside it is still GalSim's own work.

    Placed through :func:`_source_centre` and GalSim's own image coordinates
    rather than by indexing the array, so the recipe's pixel convention is
    converted in exactly the one place every other draw converts it. A GalSim
    pixel ``n`` spans ``[n - 0.5, n + 0.5)``, so rounding the source centre
    names the pixel that contains it.

    The recipe keeps every source clear of the border by at least the PSF
    margin, so that pixel is always inside the image.
    """
    centre = _source_centre(source)
    canvas.addValue(round(centre.x), round(centre.y), source.flux)


def _draw_sources(
    recipe: Recipe, sources: tuple[Source, ...], convolve_with_psf: bool
) -> np.ndarray:
    """The given sources added to one blank image, with or without the PSF.

    Takes the sources rather than reading them off the recipe so that one
    brightness bin can be drawn on its own and weighed — see
    :attr:`~simulations.recipe.Recipe.flux_groups` and :func:`render`.
    """
    canvas = _blank(recipe)
    psf = _psf(recipe)

    for source in sources:
        if source.is_point and not convolve_with_psf:
            _add_delta(canvas, source)
            continue

        profile = _profile(source)
        if convolve_with_psf:
            profile = galsim.Convolve([profile, psf])
        profile.drawImage(
            image=canvas,
            center=_source_centre(source),
            add_to_image=True,
        )

    return canvas.array.copy()


def _draw_psf_stamp(recipe: Recipe) -> np.ndarray:
    """The exact kernel that was applied, as a unit-flux stamp."""
    half_width = int(
        np.ceil(PSF_STAMP_HALF_WIDTH_FWHM * recipe.psf_fwhm / recipe.pixel_scale)
    )
    size = 2 * half_width + 1
    stamp = _psf(recipe).drawImage(nx=size, ny=size, scale=recipe.pixel_scale)
    return stamp.array.copy()


def _add_noise(recipe: Recipe, truth_conv: np.ndarray) -> np.ndarray:
    """Sky plus Poisson-Gaussian noise, drawn from the entry's derived seed.

    ``CCDNoise`` is Poisson on the signal *and* the sky, in electrons, plus
    Gaussian read noise — the camera model #67 asks for, rather than a pure
    Gaussian that would misrepresent the faint end.

    **The sky stays in the image.** #67 anchors the balanced regime to the
    packaged gallery image's statistics, and that anchor has two halves: a
    median of about 0.0108 as well as a standard deviation of about 0.0118.
    Subtracting the sky back out after drawing the noise would keep the second
    half and throw away the first, and it would leave the manifest's
    ``sky_level`` describing a level no longer present in the pixels — which is
    exactly the comparison #67 records it for.
    """
    canvas = _blank(recipe)
    canvas.array[:, :] = truth_conv
    canvas += recipe.noise.sky_level

    canvas.addNoise(
        galsim.CCDNoise(
            rng=galsim.BaseDeviate(recipe.seed),
            gain=recipe.noise.gain,
            read_noise=recipe.noise.read_noise,
        )
    )

    return canvas.array.copy()


def check_rendered_flux(
    drawn_image: np.ndarray,
    intended: float,
    what: str,
    tolerance: float = FLUX_TOLERANCE,
) -> None:
    """Compare a drawn image against the flux the recipe intended for it.

    ``what`` names the thing being weighed, because since #71 this is asked
    once of the whole convolved truth and once of each brightness bin inside
    it, and a refusal has to say which.

    Raises :class:`FluxCheckError` outside ``tolerance``, relative.
    """
    drawn = float(drawn_image.sum())
    deviation = abs(drawn - intended) / intended

    if deviation > tolerance:
        raise FluxCheckError(
            '%s carries %.6g ADU but the recipe specifies %.6g ADU, a relative '
            'deviation of %.3g which is outside the %.3g tolerance: the render '
            'does not reproduce the analytic sum, so this entry is not ground '
            'truth' % (what, drawn, intended, deviation, tolerance)
        )


def _peak_pixel(window: np.ndarray) -> tuple[int, int]:
    """The row and column of the brightest pixel, as plain ints.

    Plain rather than numpy scalars so that the refusal below reads as a pixel
    coordinate to whoever it stops.
    """
    row, column = np.unravel_index(np.argmax(window), window.shape)
    return int(row), int(column)


class PointPlacementError(ValueError):
    """A point source's truth pixel is not where its convolved image peaks.

    The pre-convolution truth of a point source is the only thing in the
    gallery that GalSim does not draw (see :func:`_add_delta`), so it is the
    only place where a coordinate-convention mistake would survive the flux
    check: moving all the flux one pixel sideways changes no total. Comparing
    the hand-placed pixel against the peak GalSim draws for the same source is
    what closes that gap, and it belongs at generation time for the same reason
    the flux check does — #67 wants a rendering mistake to stop the generator
    while a human is present.
    """


def _point_stamps(recipe: Recipe, source: Source) -> tuple[np.ndarray, np.ndarray]:
    """One point source's two pictures of itself, on one shared window.

    The first is where :func:`_add_delta` puts the source's whole flux; the
    second is where GalSim's own convolved draw of that same source peaks. Both
    are on the same small stamp of the entry's pixel grid, so their peak
    indices are directly comparable.

    Drawn **alone**, which is the whole point (#71). The blend class puts a
    faint point companion beside a source up to three thousand times brighter,
    at separations that start inside this window, so in the summed truth images
    neither peak in a window around the companion is the companion's. Isolating
    the source is the only way to keep the check live for the class that most
    needs it, and it costs nothing: the summed image is the sum of exactly
    these draws.
    """
    centre = _source_centre(source)
    half = POINT_PLACEMENT_WINDOW
    bounds = galsim.BoundsI(
        round(centre.x) - half,
        round(centre.x) + half,
        round(centre.y) - half,
        round(centre.y) + half,
    )

    placed = galsim.ImageD(bounds, scale=recipe.pixel_scale)
    _add_delta(placed, source)

    drawn = galsim.ImageD(bounds, scale=recipe.pixel_scale)
    galsim.Convolve([_profile(source), _psf(recipe)]).drawImage(
        image=drawn, center=centre, add_to_image=True
    )

    return placed.array, drawn.array


def check_point_placement(recipe: Recipe, truth_preconv: np.ndarray) -> None:
    """Check each point source against the peak GalSim draws for it.

    Does nothing for an entry of profiles, whose pre-convolution truth GalSim
    draws itself and which therefore cannot disagree with anything.

    Two questions, and they need different pictures. Whether the *convention*
    is right is asked of the source drawn alone, because in a blend field the
    summed images cannot answer it. Whether the truth image that reaches disk
    actually got that flux is asked of ``truth_preconv`` itself, so the check
    stays about the delivered array and not only about the function that
    fills it.

    The recipe's own position only picks the stamp to look in; what is compared
    for the convention is the two draws, so this does not re-derive the
    convention it exists to check.
    """
    for source in recipe.sources:
        if not source.is_point:
            continue

        placed_stamp, drawn_stamp = _point_stamps(recipe, source)
        placed = _peak_pixel(placed_stamp)
        drawn = _peak_pixel(drawn_stamp)

        centre = _source_centre(source)
        # GalSim numbers pixels from one; numpy indexes from zero. Every
        # contribution to the truth image is non-negative, so a neighbour can
        # only add to this pixel — never make up for a flux that is missing.
        delivered = float(
            truth_preconv[round(centre.y) - 1, round(centre.x) - 1]
        )
        if delivered < source.flux * (1.0 - 1e-9):
            raise PointPlacementError(
                'source %d carries %.6g ADU but the pre-convolution truth image '
                'holds only %.6g ADU in the pixel that contains it: the flux '
                'this class places by hand is not in the array that reaches '
                'disk, so this entry is not ground truth'
                % (source.source_id, source.flux, delivered)
            )

        if placed != drawn:
            raise PointPlacementError(
                'source %d has its whole flux in the truth pixel %s but the '
                'convolved truth peaks at %s, %d rows and %d columns away: the '
                'pre-convolution truth is not on the same pixel grid as the '
                'image DAWIS reads, so this entry is not ground truth'
                % (
                    source.source_id,
                    placed,
                    drawn,
                    drawn[0] - placed[0],
                    drawn[1] - placed[1],
                )
            )


def render(recipe: Recipe, tolerance: float = FLUX_TOLERANCE) -> RenderedEntry:
    """Draw one entry, and refuse to return pixels that fail its own checks.

    The convolved truth is drawn one brightness bin at a time so that each bin
    can be weighed on its own before they are added up. #67 asks for the whole
    image to be checked and that check is still made last; the per-bin ones are
    what stop it from being a statement about the brightest sources alone. See
    :attr:`~simulations.recipe.Recipe.flux_groups`.
    """
    truth_preconv = _draw_sources(recipe, recipe.sources, convolve_with_psf=False)

    truth_conv = np.zeros_like(truth_preconv)
    for group in recipe.flux_groups:
        drawn = _draw_sources(recipe, group, convolve_with_psf=True)
        check_rendered_flux(
            drawn,
            intended=sum(source.flux for source in group),
            what=(
                'the %d source(s) the recipe gives a flux of %.6g ADU'
                % (len(group), group[0].flux)
            ),
            tolerance=tolerance,
        )
        truth_conv += drawn

    check_rendered_flux(
        truth_conv,
        intended=recipe.total_flux,
        what='the convolved truth image',
        tolerance=tolerance,
    )
    check_point_placement(recipe, truth_preconv)

    return RenderedEntry(
        image=_add_noise(recipe, truth_conv),
        truth_preconv=truth_preconv,
        truth_conv=truth_conv,
        psf=_draw_psf_stamp(recipe),
        galsim_version=galsim.__version__,
    )
