"""The recipe layer — one entry of the simulation gallery, described (#68).

This module is the pure half of the generator. It takes the four axes of the
gallery — tier, class, regime, realisation — and returns a complete
description of one entry: where each source sits, how bright it is, what shape
it has, what the noise is, which files are written where, and what goes in the
manifest. It imports **numpy only**. No GalSim, no astropy, no DAWIS.

That restriction is the whole point of the layer. The generator runs inside
the ``dawis-sim`` conda environment, which the DAWIS test suite cannot enter;
keeping every decision of #67 on this side of the seam is what lets those
decisions be unit-tested in the ordinary test environment in seconds. See
``tests/unit/test_simulation_recipe.py``.

Scope of #68
------------
#68 is a tracer bullet: one narrow path through every layer, with the other
axes hard-coded for now. Two things follow.

The parts that are *general* are general here — seed derivation, entry and
file naming, tier geometry, the noise regime table. Those are decided outright
by #67 and cost nothing to state completely.

The parts #67 leaves to a later ticket are **refused**, not guessed. Asking
for a class or a regime whose numbers no ticket has decided raises
``AxisNotBuiltError``. A recipe that invented them would put an entry on disk
that looks exactly as authoritative as a decided one, and no later reader
could tell the two apart.

Units
-----
Grid geometry — positions, spacings, jitter, wing and edge radii — is in
**pixels**. Source *shape* radii are in **arcseconds**, because that is what
GalSim's profile constructors take and what ``sources.csv`` should carry for a
reader who does not know the pixel scale; :data:`PIXEL_SCALE` converts. The two
units meet in exactly one place, :func:`check_image_margin`, which takes the
pixel scale as an argument rather than reading it off the module.

Fluxes and sky levels are in **ADU**; read noise is in **electrons**; gain is
in **electrons per ADU**. This is the camera parameterisation #67 asks for, and
it is what GalSim's ``CCDNoise`` consumes directly.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    'ENTRY_FILE_NAMES',
    'KNOWN_REGIME_NAMES',
    'MANIFEST_FILE_NAME',
    'PIXEL_SCALE',
    'PROVENANCE_KEYS',
    'PSF_BETA',
    'PSF_FWHM',
    'PSF_TYPE',
    'REALISATIONS',
    'REGIMES',
    'SOURCE_CLASSES',
    'TIERS',
    'ZERO_POINT',
    'AxisNotBuiltError',
    'EntryAxes',
    'GridTooTightError',
    'NoiseParameters',
    'Recipe',
    'Source',
    'SourceClippedError',
    'TruncationTooTightError',
    'Tier',
    'all_entries',
    'build_recipe',
    'built_entries',
    'check_grid_spacing',
    'check_truncation',
    'check_image_margin',
    'convolved_unit_profile',
    'default_output_root',
    'derive_seed',
    'entry_name',
    'truncated_scale_radius',
]


# --------------------------------------------------------------------------
# The storage contract (#67)
# --------------------------------------------------------------------------

#: The six files that make up one entry, in the order #67 lists them. The
#: metric half of #25 is written against these names, so they are a published
#: contract and not an implementation detail of the write layer.
ENTRY_FILE_NAMES = (
    'image.fits',
    'truth_preconv.fits',
    'truth_conv.fits',
    'psf.fits',
    'sources.csv',
    'manifest.json',
)

#: The file whose presence means an entry is already built. The generator's
#: idempotent skip keys on it, so it is written last.
MANIFEST_FILE_NAME = 'manifest.json'

#: The manifest keys that record *what drew the pixels* rather than *what the
#: recipe decided*. The skip of #69 compares a manifest on disk against the
#: recipe and ignores these: a numpy upgrade alone changes no ground truth, and
#: treating it as a change would regenerate the whole gallery for nothing.
PROVENANCE_KEYS = ('galsim_version', 'numpy_version', 'source_commit')


def default_output_root() -> Path:
    """The user's data directory, where the real Euclid and HST data live.

    ``simulations/`` sits beside ``euclid/`` and ``misc/`` so synthetic and
    real data can never be confused for one another (#67).
    """
    return Path.home() / 'dawis-data' / 'simulations'


def entry_name(source_class: str, regime: str, realisation: int) -> str:
    """``<class>_<regime>_r<N>`` — the entry directory's name."""
    return '%s_%s_r%d' % (source_class, regime, realisation)


# --------------------------------------------------------------------------
# Seed derivation (#67)
# --------------------------------------------------------------------------

#: Mixed into the seed key so that a future generator with a different meaning
#: for the same three axes cannot collide with these images by accident.
_SEED_NAMESPACE = 'dawis-simulation-gallery-v1'


def derive_seed(source_class: str, regime: str, realisation: int) -> int:
    """The noise seed for an entry, from its class, regime and realisation.

    The tier is deliberately absent, and so, far more importantly, is the
    restoration variant under test. Every variant therefore reads literally
    the same image bytes, and a variant comparison is paired by construction
    rather than by the discipline of whoever runs the sweep (#67).

    ``blake2b`` rather than :func:`hash`, which is salted per process, or
    ``zlib.crc32``, which is short enough to collide across 72 entries plus
    whatever a later ticket adds. The digest is truncated to 63 bits because
    that is the widest value both numpy's ``SeedSequence`` and GalSim's
    ``BaseDeviate`` accept without complaint.
    """
    key = '%s|%s|%s|r%d' % (_SEED_NAMESPACE, source_class, regime, realisation)
    digest = hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'big') & (2**63 - 1)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


class AxisNotBuiltError(NotImplementedError):
    """An axis value whose numbers no ticket has decided yet.

    #68 builds ``full`` / ``sersic_n4`` / ``balanced`` / ``r0``. Everything
    else is refused rather than guessed.
    """


class GridTooTightError(ValueError):
    """The brightest source's wings reach its neighbour's cell.

    #67 makes this the generator's own check: without it, the isolated
    bright-galaxy class silently becomes a second blend class as soon as
    someone widens the magnitude range, and every number measured on it would
    be measuring the wrong thing.
    """


def check_grid_spacing(wing_radius: float, grid_spacing: float) -> None:
    """Refuse a grid whose cells are narrower than the brightest wings."""
    if wing_radius >= grid_spacing:
        raise GridTooTightError(
            'the brightest source has a wing radius of %.1f pixels, which does '
            'not fit inside the grid spacing of %.1f pixels: this field would '
            'be a blend field, not an isolated-source field'
            % (wing_radius, grid_spacing)
        )


class TruncationTooTightError(ValueError):
    """The profile is cut while it is still brighter than the background.

    The truncation exists to bound the source, not to shape it. If the cut
    lands above the noise it shows as a hard circular edge in the truth image —
    an artefact of the generator that a metric would read as structure.
    """


def check_truncation(wing_radius: float, truncation_radius: float) -> None:
    """Refuse a truncation that cuts the brightest source above the noise."""
    if truncation_radius <= wing_radius:
        raise TruncationTooTightError(
            'the brightest source is still above the background at %.1f pixels '
            'but is truncated at %.1f: the cut would show as a hard edge in the '
            'truth image' % (wing_radius, truncation_radius)
        )


class SourceClippedError(ValueError):
    """A source's truncated profile reaches past the edge of the image.

    The generator checks its rendered flux against the recipe's analytic sum
    (#68). Light that falls off the edge is light the render cannot account
    for, so a clipped source would make that check fail with no way to tell a
    rendering bug from a layout one. Catching it here says which it is.
    """


def check_image_margin(
    sources: 'tuple[Source, ...]',
    image_size: int,
    margin: float,
    pixel_scale: float,
) -> None:
    """Refuse a layout whose outermost sources are cut by the image edge.

    ``margin`` and ``image_size`` are in pixels; ``Source.truncation_radius``
    is in arcseconds, so ``pixel_scale`` is required rather than assumed.
    """
    for source in sources:
        reach = source.truncation_radius / pixel_scale + margin
        clearance = min(
            source.x, source.y, image_size - source.x, image_size - source.y
        )
        if reach > clearance:
            raise SourceClippedError(
                'source %d sits %.1f pixels from the nearest image edge but '
                'reaches %.1f pixels, so part of its flux would fall outside '
                'the %dx%d image and the analytic total could not be recovered'
                % (source.source_id, clearance, reach, image_size, image_size)
            )


# --------------------------------------------------------------------------
# The axes (#67)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier:
    """One image size and the number of grid steps laid out on it."""

    image_size: int
    grid_side: int

    def grid_spacing(self, edge_margin: float) -> float:
        """Pixels between adjacent grid positions, given an edge margin.

        The grid is **inset**, not a tiling of the whole image: the outermost
        positions sit ``edge_margin`` pixels in from the border, and the rest
        are spread evenly between them. A tiling would put its outer row half a
        cell from the edge, which for the classes this gallery draws is closer
        than a bright source's own truncation radius — the outer sources would
        be cut, and the generator's flux check could not pass.
        """
        return (self.image_size - 2.0 * edge_margin) / (self.grid_side - 1)


#: Two sizes: a cheap one for wide sweeps and a larger one for confirmation.
TIERS = {
    'small': Tier(image_size=512, grid_side=3),
    'full': Tier(image_size=1024, grid_side=5),
}

#: The four source classes of #67. Only ``sersic_n4`` is built by #68.
SOURCE_CLASSES = ('sersic_n4', 'point', 'sersic_n1_halo', 'blend')


@dataclass(frozen=True)
class NoiseParameters:
    """Poisson-Gaussian noise in camera terms (#67).

    ``sky_level`` is the mean background in ADU per pixel, ``read_noise`` the
    detector read noise in electrons, and ``gain`` the conversion in electrons
    per ADU. GalSim's ``CCDNoise`` takes exactly these three.

    These numbers are recorded in the manifest for a human reading a result.
    They are never passed into a DAWIS run: the algorithm estimates the noise
    itself, as it would on real data.
    """

    sky_level: float
    read_noise: float
    gain: float

    @property
    def photon_variance(self) -> float:
        """The shot-noise half of the background variance, in ADU squared."""
        return self.sky_level / self.gain

    @property
    def read_variance(self) -> float:
        """The read-noise half of the background variance, in ADU squared."""
        return (self.read_noise / self.gain) ** 2

    @property
    def sigma(self) -> float:
        """Background standard deviation in ADU — what an SNR is measured against."""
        return float(np.sqrt(self.photon_variance + self.read_variance))


#: The balanced regime, anchored to the packaged HST ACS F814W gallery image:
#: median 0.0108 ADU, standard deviation 0.0118 ADU (#67). Photon and read
#: noise contribute about half the variance each, which is what "balanced"
#: names. The gain is large because the gallery image is in calibrated units
#: rather than raw counts; it is a real conversion factor for that image, not
#: a round number chosen for looks.
_BALANCED_NOISE = NoiseParameters(sky_level=0.0108, read_noise=1.293, gain=155.0)

#: Only ``balanced`` is decided. The other two regimes are points in the same
#: three-parameter space, but which points is a decision #68 does not make.
REGIMES = {'balanced': _BALANCED_NOISE}

#: Named so that an unbuilt regime is refused with a message that says "not
#: yet" rather than "no such thing".
KNOWN_REGIME_NAMES = ('read_noise', 'balanced', 'photon')

#: The three noise realisations of #67. Raising the count later adds ``r3``
#: onward without disturbing these, which is the point of deriving the seed
#: from the realisation index rather than from a position in a list.
REALISATIONS = (0, 1, 2)


# --------------------------------------------------------------------------
# Optics and the brightness ladder (#67)
# --------------------------------------------------------------------------

#: Arcseconds per pixel, matching drizzled HST ACS, which is what the gallery
#: image the balanced regime is anchored to actually is.
PIXEL_SCALE = 0.03

#: A synthetic Moffat, per #67's "synthetic first" decision. The FWHM is fixed
#: and is not a swept axis; a real Euclid or HST PSF is deferred.
PSF_TYPE = 'moffat'
PSF_FWHM = 0.09
PSF_BETA = 3.5

#: The photometric zero point the magnitudes are quoted against — the AB zero
#: point of HST ACS F814W. It converts flux to a magnitude and back and has no
#: other effect: shifting it would relabel every source and change no pixel.
ZERO_POINT = 25.94

#: Peak SNR at the two ends of the ladder. #67 asks for roughly 3 — marginally
#: detectable — to roughly 1000. The steps between are geometric, so a column
#: index reads as a fixed factor in brightness.
PEAK_SNR_BRIGHTEST = 1000.0
PEAK_SNR_FAINTEST = 3.0

#: Sérsic half-light radius for the ``sersic_n4`` class, in arcseconds. Ten
#: pixels at the scale above — large enough that the profile is resolved well
#: past the PSF, small enough that twenty-five of them fit an image with room
#: for their wings.
SERSIC_N4_HALF_LIGHT_RADIUS = 0.30

#: The Sérsic profile is truncated at this multiple of the half-light radius.
#: An n=4 profile has infinite extent and GalSim would otherwise pick its own
#: cutoff from a folding threshold. Truncating explicitly does three things #68
#: needs: it makes the analytic total flux exact, so the generator's own flux
#: check has something firm to compare against; it bounds the wing radius the
#: grid-spacing refusal is measured on; and it lets the grid reserve a known
#: edge margin, without which the drawn flux could not match the analytic sum
#: however careful the renderer was.
#:
#: Eight is chosen so the cut lands *below the noise*: the brightest source on
#: the ladder is at about a quarter of the background sigma there, so the
#: truncation is invisible rather than a hard circular edge in the truth image.
#: Five was tried first and put the cut right at the noise floor.
SERSIC_TRUNCATION_FACTOR = 8.0

#: Position jitter, in pixels. The second grid axis buys independent samples
#: of one brightness rather than copies of it: sub-pixel phase relative to the
#: pixel grid is not the same for two sources, and neither is the noise under
#: them. Quoted in pixels rather than as a fraction of the spacing so that the
#: edge margin below is a fixed number and not a fixed point of its own.
JITTER_PIXELS = 12.0

#: How far past its own FWHM a Moffat is treated as finished, when reserving
#: room for it. A beta=3.5 Moffat is under one part in ``10**7`` of its peak by
#: six FWHM.
PSF_MARGIN_FWHM = 6.0

#: Slack added to the edge margin so that the margin and the clearance check
#: cannot come out exactly equal and turn on a floating-point rounding.
EDGE_PAD_PIXELS = 2.0

#: Ellipticity is drawn uniformly in magnitude below this, at a uniform angle.
#: Round sources would make the shape-bias metric of #25 a test of nothing.
MAX_ELLIPTICITY = 0.3


# --------------------------------------------------------------------------
# One source
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """One row of ``sources.csv`` — the ground truth for one object.

    ``x`` and ``y`` are pixel coordinates in the image, zero at the lower-left
    corner of the first pixel. ``flux`` is the total flux in ADU that the
    truth images integrate to, and ``magnitude`` is that flux against
    :data:`ZERO_POINT`. ``e1`` and ``e2`` are the true reduced-shear-style
    ellipticity components, recorded as numbers so that #25's shape-bias fit
    has a truth value carrying no measurement noise of its own.
    """

    source_id: int
    column: int
    row: int
    x: float
    y: float
    magnitude: float
    flux: float
    sersic_index: float
    half_light_radius: float
    e1: float
    e2: float
    truncation_radius: float
    peak_snr: float


# --------------------------------------------------------------------------
# One entry
# --------------------------------------------------------------------------


def _decided_fields(manifest: dict) -> dict:
    """A manifest with its provenance dropped — only what the recipe decided."""
    return {
        key: value for key, value in manifest.items() if key not in PROVENANCE_KEYS
    }


@dataclass(frozen=True)
class Recipe:
    """A complete description of one gallery entry, decided and not yet drawn.

    Everything the render and write layers need is here, and nothing they
    produce is. Handing this object to GalSim is the only thing the render
    layer does.
    """

    tier: str
    source_class: str
    regime: str
    realisation: int
    seed: int
    image_shape: tuple[int, int]
    pixel_scale: float
    grid_shape: tuple[int, int]
    grid_spacing: float
    jitter_amplitude: float
    psf_type: str
    psf_fwhm: float
    psf_beta: float
    zero_point: float
    noise: NoiseParameters
    sources: tuple[Source, ...]
    magnitude_ladder: tuple[float, ...]
    peak_snr_ladder: tuple[float, ...]
    wing_radius: float

    @property
    def total_flux(self) -> float:
        """The analytic sum the rendered truth image must reproduce.

        The generator compares the convolved truth image's total against this
        and fails loudly outside tolerance (#68). Because the Sérsic profiles
        are truncated and the PSF is normalised, this is an exact statement
        about the intended image and not an approximation of one.
        """
        return float(sum(source.flux for source in self.sources))

    def entry_directory(self, root: Path | str) -> Path:
        """``<root>/<tier>/<class>_<regime>_r<N>`` — one directory, one run."""
        return Path(root) / self.tier / entry_name(
            self.source_class, self.regime, self.realisation
        )

    def output_paths(self, root: Path | str) -> dict[str, Path]:
        """Every file of the entry, keyed by its contract name."""
        directory = self.entry_directory(root)
        return {name: directory / name for name in ENTRY_FILE_NAMES}

    def matches_manifest(self, manifest: dict) -> bool:
        """Whether a manifest on disk describes the entry this recipe describes.

        Compares every key except :data:`PROVENANCE_KEYS`. The comparison is
        exact rather than tolerant on purpose: the numbers here are the ground
        truth the restoration battery measures against, and a recipe that has
        drifted by a last digit has still drifted. #67 states outright that
        ground truth must not move without a signal, and this is the signal.
        """
        return _decided_fields(self.manifest()) == _decided_fields(manifest)

    def manifest(self, galsim_version: str = '', source_commit: str = '') -> dict:
        """Provenance and parameters, ready for :func:`json.dump`.

        The GalSim version and the source commit are filled in by the
        generator, which knows them; the recipe cannot, because it must not
        import GalSim. The keys are present either way, so a missing one is a
        failure in the test suite rather than a surprise on disk months later.
        """
        return {
            'tier': self.tier,
            'class': self.source_class,
            'regime': self.regime,
            'realisation': self.realisation,
            'seed': self.seed,
            'noise': {
                'sky_level': float(self.noise.sky_level),
                'read_noise': float(self.noise.read_noise),
                'gain': float(self.noise.gain),
            },
            'noise_sigma': float(self.noise.sigma),
            'image_shape': list(self.image_shape),
            'pixel_scale': float(self.pixel_scale),
            'grid_shape': list(self.grid_shape),
            'grid_spacing': float(self.grid_spacing),
            'jitter_amplitude': float(self.jitter_amplitude),
            'psf_type': self.psf_type,
            'psf_fwhm': float(self.psf_fwhm),
            'psf_beta': float(self.psf_beta),
            'zero_point': float(self.zero_point),
            'wing_radius': float(self.wing_radius),
            'total_flux': self.total_flux,
            'magnitude_ladder': [float(value) for value in self.magnitude_ladder],
            'peak_snr_ladder': [float(value) for value in self.peak_snr_ladder],
            'n_sources': len(self.sources),
            'galsim_version': galsim_version,
            'numpy_version': np.__version__,
            'source_commit': source_commit,
        }


# --------------------------------------------------------------------------
# The numerical profile — how a flux becomes a peak, and where the wings end
# --------------------------------------------------------------------------


def truncated_scale_radius(
    sersic_index: float,
    half_light_radius: float,
    truncation_radius: float,
    samples: int = 200_000,
) -> float:
    """The scale radius of a truncated Sérsic with the given half-light radius.

    ``I(r) = exp(-(r/r_s)**(1/n))`` up to ``truncation_radius`` and zero beyond
    it. For an untruncated profile ``r_s = r_e / b_n**n``; truncation removes
    outer light, so the *truncated* profile of that scale radius encloses half
    its remaining light well inside ``r_e``, and a larger, flatter ``r_s`` is
    needed to put the half-light radius back where it was asked for.

    This matters, and getting it wrong is not subtle. GalSim resolves the same
    constraint internally when ``trunc`` is given, so a recipe that used the
    untruncated relation would predict a profile roughly thirty per cent more
    concentrated than the one actually drawn, and every peak SNR on the ladder
    would be about a factor of two off its target.

    Bisection over a cumulative sum rather than a library root-finder, so that
    this module keeps to numpy. The enclosed-light fraction falls monotonically
    with ``r_s``, so the bracket is safe.
    """
    radius = np.linspace(0.0, truncation_radius, samples)
    weight = np.maximum(radius, 1e-12)

    def enclosed_fraction(scale_radius: float) -> float:
        intensity = np.exp(-((weight / scale_radius) ** (1.0 / sersic_index)))
        cumulative = np.cumsum(intensity * radius)
        return float(np.interp(half_light_radius, radius, cumulative / cumulative[-1]))

    low, high = truncation_radius * 1e-9, truncation_radius
    for _ in range(120):
        middle = 0.5 * (low + high)
        if enclosed_fraction(middle) > 0.5:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


#: The truncation used only to *measure* the wing radius, as a multiple of the
#: half-light radius. It is deliberately far larger than
#: :data:`SERSIC_TRUNCATION_FACTOR`. Anything past it reads as "wings wider
#: than the probe", which every refusal already rejects.
#:
#: Measuring on the rendered truncation would make the wing radius its own
#: prisoner: the cut bounds the profile, so the measured radius could never
#: exceed it, and :func:`check_grid_spacing` could never fire however bright
#: the ladder was made — which is the one scenario #67 names for that refusal.
#: Probing well past the cut gives the radius at which the source really falls
#: under the background, so both refusals are live.
WING_PROBE_TRUNCATION_FACTOR = 15.0

#: Sub-samples per pixel side used to model a profile before binning it back
#: to the pixel grid. An n=4 core is far narrower than a pixel, so a coarse
#: sampling overstates the peak badly: this converges from about forty per cent
#: high at three sub-samples to within two per cent of GalSim's own rendering
#: at fifteen, which is well inside the "roughly 3 to 1000" the SNR ladder is
#: quoted to. The result is cached because it costs a second or two.
PROFILE_OVERSAMPLE = 15

#: Sub-samples per pixel side for the wing-radius probe. The wing radius is a
#: crossing far out in the skirt, where a pixel is many times wider than any
#: feature, so it converges at a coarse sampling that the probe's much larger
#: grid can afford. Sampling the probe at :data:`PROFILE_OVERSAMPLE` would ask
#: for a grid of some hundreds of megabytes to answer a question accurate to
#: the pixel.
WING_PROBE_OVERSAMPLE = 3


@lru_cache(maxsize=8)
def convolved_unit_profile(
    sersic_index: float,
    half_light_radius_pixels: float,
    truncation_pixels: float,
    psf_fwhm_pixels: float,
    psf_beta: float,
    oversample: int = PROFILE_OVERSAMPLE,
) -> np.ndarray:
    """A unit-flux truncated Sérsic convolved with the Moffat PSF, per pixel.

    Source, PSF and convolution are all computed on a grid oversampled by
    :data:`PROFILE_OVERSAMPLE` and binned down at the end, so the pixel
    response is integrated rather than point-sampled — which is what GalSim
    does when it draws, and what makes this prediction comparable to the
    render.

    Used for two things and drawn once for both: the peak pixel fraction that
    turns a target SNR into a flux, and the radius at which the wings fall
    under the background, which the grid-spacing refusal is measured on. This
    is a *design* calculation, not the render — the pixels that reach disk are
    GalSim's.
    """
    psf_margin = int(np.ceil(6.0 * psf_fwhm_pixels))
    half_width = int(np.ceil(truncation_pixels)) + psf_margin
    side = 2 * half_width + 1

    fine_side = side * oversample
    axis = (np.arange(fine_side) - (fine_side - 1) / 2.0) / oversample
    radius = np.hypot(*np.meshgrid(axis, axis, indexing='ij'))

    scale_radius = truncated_scale_radius(
        sersic_index, half_light_radius_pixels, truncation_pixels
    )
    source = np.exp(
        -((np.maximum(radius, 1e-12) / scale_radius) ** (1.0 / sersic_index))
    )
    source[radius > truncation_pixels] = 0.0
    source /= source.sum()

    moffat_scale = psf_fwhm_pixels / (2.0 * np.sqrt(2.0 ** (1.0 / psf_beta) - 1.0))
    psf = (1.0 + (radius / moffat_scale) ** 2) ** (-psf_beta)
    psf /= psf.sum()

    # Both arrays are centred on the same grid, so the circular wrap of an FFT
    # convolution puts the result back at the centre after one ``fftshift``.
    convolved = np.fft.fftshift(
        np.fft.irfft2(np.fft.rfft2(source) * np.fft.rfft2(psf), s=source.shape)
    )

    binned = convolved.reshape(side, oversample, side, oversample).sum(axis=(1, 3))
    binned.flags.writeable = False
    return binned


def _peak_fraction(profile: np.ndarray) -> float:
    """The share of a source's total flux that lands in its brightest pixel."""
    return float(profile.max() / profile.sum())


def _wing_radius(profile: np.ndarray, brightness_floor: float) -> float:
    """The radius beyond which a source is fainter than the background.

    Measured on the convolved profile of the *brightest* source scaled to its
    real flux, against the background standard deviation. This is the radius
    #67's refusal compares to the grid spacing: past it the source contributes
    less than the noise, so a neighbour there is not blended with it.
    """
    half_width = (profile.shape[0] - 1) // 2
    axis = np.arange(-half_width, half_width + 1, dtype=float)
    radius = np.hypot(*np.meshgrid(axis, axis, indexing='ij'))

    above = profile >= brightness_floor
    if not above.any():
        return 0.0
    return float(radius[above].max())



# --------------------------------------------------------------------------
# Building the entry
# --------------------------------------------------------------------------


def _magnitude_from_flux(flux: float, zero_point: float) -> float:
    return float(zero_point - 2.5 * np.log10(flux))


def _draw_ellipticity(rng: np.random.Generator) -> tuple[float, float]:
    """A random ellipticity below :data:`MAX_ELLIPTICITY`, uniform in angle."""
    magnitude = rng.uniform(0.0, MAX_ELLIPTICITY)
    angle = rng.uniform(0.0, np.pi)
    return float(magnitude * np.cos(2.0 * angle)), float(magnitude * np.sin(2.0 * angle))


def _refuse_unbuilt_axes(
    tier: str, source_class: str, regime: str, realisation: int
) -> None:
    """Everything #68 did not decide, refused with a message that says why."""
    # The tier is checked but never deferred. #67 decides both tiers outright —
    # an image size and a grid side each, and nothing else — so unlike the other
    # three axes there is no undecided number here to guess at.
    if tier not in TIERS:
        raise AxisNotBuiltError(
            'unknown tier %r: the gallery has %s'
            % (tier, ' and '.join(sorted(TIERS)))
        )
    if source_class not in SOURCE_CLASSES:
        raise AxisNotBuiltError(
            'unknown class %r: the gallery has %s'
            % (source_class, ', '.join(SOURCE_CLASSES))
        )
    if source_class != 'sersic_n4':
        raise AxisNotBuiltError(
            'the %r class is specified in #67 but not built: #68 is the tracer '
            'bullet and builds sersic_n4 only. Its parameters belong to the '
            'ticket that decides them.' % source_class
        )
    if regime not in KNOWN_REGIME_NAMES:
        raise AxisNotBuiltError(
            'unknown regime %r: the gallery has %s'
            % (regime, ', '.join(KNOWN_REGIME_NAMES))
        )
    if regime not in REGIMES:
        raise AxisNotBuiltError(
            'the %r regime is specified in #67 but not built: #68 anchors the '
            'balanced regime to the gallery image and leaves the other two '
            'points in camera-parameter space to the ticket that decides them.'
            % regime
        )
    if realisation != 0:
        raise AxisNotBuiltError(
            'realisation r%d is specified in #67 but not built: #68 builds r0. '
            'Later realisations are additive and need no change here beyond '
            'lifting this refusal.' % realisation
        )


def build_recipe(
    tier: str, source_class: str, regime: str, realisation: int
) -> Recipe:
    """Describe one gallery entry completely, without drawing a single pixel.

    Raises :class:`AxisNotBuiltError` for an axis value #68 did not decide, and
    :class:`GridTooTightError` if the brightest source's wings would reach its
    neighbour's cell.
    """
    _refuse_unbuilt_axes(tier, source_class, regime, realisation)

    tier_geometry = TIERS[tier]
    noise = REGIMES[regime]
    seed = derive_seed(source_class, regime, realisation)
    rng = np.random.default_rng(seed)

    grid_side = tier_geometry.grid_side
    jitter_amplitude = JITTER_PIXELS

    half_light_radius_pixels = SERSIC_N4_HALF_LIGHT_RADIUS / PIXEL_SCALE
    truncation_pixels = SERSIC_TRUNCATION_FACTOR * half_light_radius_pixels
    psf_fwhm_pixels = PSF_FWHM / PIXEL_SCALE

    # Room for the whole of a jittered source, so that none of them is cut by
    # the border and the analytic flux sum stays recoverable.
    psf_margin = PSF_MARGIN_FWHM * psf_fwhm_pixels
    edge_margin = truncation_pixels + psf_margin + jitter_amplitude + EDGE_PAD_PIXELS
    spacing = tier_geometry.grid_spacing(edge_margin)

    unit_profile = convolved_unit_profile(
        sersic_index=4.0,
        half_light_radius_pixels=half_light_radius_pixels,
        truncation_pixels=truncation_pixels,
        psf_fwhm_pixels=psf_fwhm_pixels,
        psf_beta=PSF_BETA,
    )
    peak_fraction = _peak_fraction(unit_profile)

    # Measured on a far wider probe, so the render's own cut does not bound it.
    probe_profile = convolved_unit_profile(
        sersic_index=4.0,
        half_light_radius_pixels=half_light_radius_pixels,
        truncation_pixels=WING_PROBE_TRUNCATION_FACTOR * half_light_radius_pixels,
        psf_fwhm_pixels=psf_fwhm_pixels,
        psf_beta=PSF_BETA,
        oversample=WING_PROBE_OVERSAMPLE,
    )

    # Brightness varies along one grid axis, position jitter along the other,
    # so the full tier gives five brightness steps with five samples each —
    # the shape "bin residuals by SNR" wants (#67). The ladder is geometric in
    # peak SNR, so a column index is a fixed factor in brightness.
    peak_snr_ladder = np.geomspace(
        PEAK_SNR_BRIGHTEST, PEAK_SNR_FAINTEST, grid_side
    )
    flux_ladder = peak_snr_ladder * noise.sigma / peak_fraction
    magnitude_ladder = tuple(
        _magnitude_from_flux(flux, ZERO_POINT) for flux in flux_ladder
    )

    wing_radius = _wing_radius(
        probe_profile * float(flux_ladder[0]), brightness_floor=noise.sigma
    )
    check_truncation(wing_radius=wing_radius, truncation_radius=truncation_pixels)
    check_grid_spacing(wing_radius=wing_radius, grid_spacing=spacing)

    sources = []
    source_id = 0
    for row in range(grid_side):
        for column in range(grid_side):
            e1, e2 = _draw_ellipticity(rng)
            sources.append(
                Source(
                    source_id=source_id,
                    column=column,
                    row=row,
                    x=edge_margin
                    + column * spacing
                    + float(rng.uniform(-jitter_amplitude, jitter_amplitude)),
                    y=edge_margin
                    + row * spacing
                    + float(rng.uniform(-jitter_amplitude, jitter_amplitude)),
                    magnitude=magnitude_ladder[column],
                    flux=float(flux_ladder[column]),
                    sersic_index=4.0,
                    half_light_radius=SERSIC_N4_HALF_LIGHT_RADIUS,
                    e1=e1,
                    e2=e2,
                    truncation_radius=SERSIC_N4_HALF_LIGHT_RADIUS
                    * SERSIC_TRUNCATION_FACTOR,
                    peak_snr=float(peak_snr_ladder[column]),
                )
            )
            source_id += 1

    sources = tuple(sources)
    check_image_margin(
        sources,
        tier_geometry.image_size,
        margin=psf_margin,
        pixel_scale=PIXEL_SCALE,
    )

    return Recipe(
        tier=tier,
        source_class=source_class,
        regime=regime,
        realisation=realisation,
        seed=seed,
        image_shape=(tier_geometry.image_size, tier_geometry.image_size),
        pixel_scale=PIXEL_SCALE,
        grid_shape=(grid_side, grid_side),
        grid_spacing=spacing,
        jitter_amplitude=jitter_amplitude,
        psf_type=PSF_TYPE,
        psf_fwhm=PSF_FWHM,
        psf_beta=PSF_BETA,
        zero_point=ZERO_POINT,
        noise=noise,
        sources=sources,
        magnitude_ladder=magnitude_ladder,
        peak_snr_ladder=tuple(float(value) for value in peak_snr_ladder),
        wing_radius=wing_radius,
    )


# --------------------------------------------------------------------------
# The gallery as a whole (#69)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryAxes:
    """One point on the four axes — the address of a single gallery entry."""

    tier: str
    source_class: str
    regime: str
    realisation: int


def all_entries() -> tuple[EntryAxes, ...]:
    """Every point on the four axes of #67 — all 72 of them, built or not."""
    return tuple(
        EntryAxes(tier, source_class, regime, realisation)
        for tier, source_class, regime, realisation in itertools.product(
            TIERS, SOURCE_CLASSES, KNOWN_REGIME_NAMES, REALISATIONS
        )
    )


def built_entries(
    tier: str | None = None,
    source_class: str | None = None,
    regime: str | None = None,
    realisation: int | None = None,
) -> tuple[EntryAxes, ...]:
    """The entries this recipe can actually build, narrowed by any axis given.

    #67 specifies 72 entries and #68 decided the numbers for one path through
    them; the rest are refused rather than guessed. So "every entry" means every entry
    whose parameters a ticket has written down, and this asks the refusal
    itself rather than keeping a second list beside it. A ticket that decides
    the ``point`` class lifts one refusal and ``--all`` grows by itself, with
    no list here to forget to update.

    Only :class:`AxisNotBuiltError` is treated as "not built". A decided entry
    that fails a *geometry* check — :class:`GridTooTightError`,
    :class:`SourceClippedError` — is left in the sweep on purpose, so that
    ``--all`` stops on it. Dropping it here would hide a broken decision behind
    a shorter run, and #67 puts those checks in the generator precisely so a
    human sees them. Entries already written stay written, so the rerun after
    a fix still resumes.

    An axis passed as ``None`` is swept; an axis given a value is fixed.
    """
    wanted = {
        'tier': tier,
        'source_class': source_class,
        'regime': regime,
        'realisation': realisation,
    }
    selected = []
    for axes in all_entries():
        axis_values = asdict(axes)
        if any(
            value is not None and axis_values[name] != value
            for name, value in wanted.items()
        ):
            continue
        try:
            _refuse_unbuilt_axes(**axis_values)
        except AxisNotBuiltError:
            continue
        selected.append(axes)
    return tuple(selected)
