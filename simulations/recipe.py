"""The recipe layer — one entry of the simulation gallery, described (#68-#71).

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

What is decided, and what is refused
------------------------------------
#68 was a tracer bullet: one narrow path through every layer, with the other
axes hard-coded. #70 added the ``point`` and ``sersic_n1_halo`` classes and
#71 the ``blend`` class, which completes the class axis. Two things follow.

The parts that are *general* are general here — seed derivation, entry and
file naming, tier geometry, the noise regime table, and the grid the three
isolated classes share. Those are decided outright by #67 and cost nothing to
state completely.

The parts #67 leaves to a later ticket are **refused**, not guessed. Asking
for a regime or a realisation whose numbers no ticket has decided raises
``AxisNotBuiltError``. A recipe that invented them would put an entry on disk
that looks exactly as authoritative as a decided one, and no later reader
could tell the two apart.

The three isolated classes
--------------------------
``sersic_n4``, ``point`` and ``sersic_n1_halo`` share their whole layout —
magnitude across the grid, position jitter down it, the same peak-SNR range
on the same tiers — and differ only in :data:`SOURCE_PROFILES`. That is why
#70 added source construction and not a line of grid logic.

The blend class
---------------
``blend`` is the odd one out, and #71 built it as a *layout* rather than as a
profile. Every cell of its grid holds a **pair**: a bright ``sersic_n4``
component held at one fixed brightness, and a faint ``point`` companion offset
from it. Separation steps down the rows and the pair's flux ratio across the
columns, so the grid is the two-dimensional map #67 asks for — where the hole
around the bright source starts consuming its neighbour. Brightness binning is
already covered by the three isolated classes, which is exactly why the blend
grid spends its axes on something else.

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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    'BLEND_BRIGHT_PEAK_SNR',
    'BLEND_BRIGHT_PROFILE',
    'BLEND_CLASS',
    'BLEND_COMPANION_PEAK_SNR_BRIGHTEST',
    'BLEND_COMPANION_PROFILE',
    'BLEND_SEPARATION_MIN_FWHM',
    'BUILT_CLASSES',
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
    'SOURCE_PROFILES',
    'TIERS',
    'ZERO_POINT',
    'AxisNotBuiltError',
    'EntryAxes',
    'GridTooTightError',
    'NoiseParameters',
    'PairsTooCloseError',
    'Recipe',
    'Source',
    'SourceClippedError',
    'SourceProfile',
    'TruncationTooTightError',
    'Tier',
    'all_entries',
    'build_recipe',
    'built_entries',
    'check_grid_spacing',
    'check_pair_spacing',
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

    Both tiers build, and since #71 so does every class of #67 — the three
    isolated ones of :data:`SOURCE_PROFILES` and the ``blend`` layout beside
    them. The two regimes beside ``balanced`` and the realisations beyond
    ``r0`` are refused rather than guessed, each until the ticket that decides
    its numbers lands.
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


class PairsTooCloseError(ValueError):
    """One pair's companion reaches into the neighbouring pair's cell.

    Distinct from :class:`GridTooTightError`, and deliberately so. The
    within-pair separation is the thing the ``blend`` class exists to vary, so
    the refusal that has to stay live for it is the one about *neighbouring*
    pairs. A grid that let one pair's companion sit inside the next pair's
    bright source's wings would be measuring two blends at once, and no row of
    the map would mean what it says.
    """


def check_pair_spacing(
    wing_radius: float,
    separation: float,
    grid_spacing: float,
    jitter_amplitude: float,
) -> None:
    """Refuse a grid whose pairs reach each other once a pair is laid out.

    The reach that matters is the **bright** component's. The companion is
    fainter than it by construction — the flux-ratio axis never rises above
    :data:`BLEND_COMPANION_PEAK_SNR_BRIGHTEST`, an order of magnitude below
    the bright component's fixed peak SNR — so it is the bright source's wings
    that decide whether a neighbouring pair is blended with this one. A
    companion offset towards its neighbour closes that gap by the whole
    separation, which is why the two are added rather than compared one at a
    time.

    Unlike :func:`check_grid_spacing` this compares against the **closest**
    two cells can come, not the nominal spacing: the jitter moves two pair
    centres towards each other by up to twice its amplitude. #67 states the
    isolated refusal against the nominal spacing and that wording is left
    alone, but #71's is a refusal of its own, and here the difference is not a
    rounding — a pair already reaches half as far again as its bright
    component does, so the twenty-four pixels the jitter can take are a real
    part of the margin rather than slack around it.
    """
    reach = wing_radius + separation
    closest = grid_spacing - 2.0 * jitter_amplitude
    if reach >= closest:
        raise PairsTooCloseError(
            'a pair reaches %.1f pixels — %.1f of wings around the bright '
            'component plus a separation of %.1f — which does not fit inside '
            'the %.1f pixels two neighbouring cells can close to, from a grid '
            'spacing of %.1f and a jitter of %.1f either way: neighbouring '
            'pairs would blend into one another, and the separation axis would '
            'stop meaning the distance within a pair'
            % (reach, wing_radius, separation, closest, grid_spacing, jitter_amplitude)
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


@dataclass(frozen=True)
class EntryAxes:
    """One point on the four axes — the address of a single gallery entry.

    The four travel together everywhere: they name an entry to #69's sweep,
    they are what the refusals are stated about, and since #71 they are what
    each of the two layouts is handed. Passing them as one address rather than
    as four parallel arguments is what keeps those signatures short enough to
    read.
    """

    tier: str
    source_class: str
    regime: str
    realisation: int


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
SERSIC_N4_TRUNCATION_FACTOR = 8.0

#: Half-light radius for the ``sersic_n1_halo`` class, in arcseconds — about
#: seventeen pixels, well over the ``sersic_n4`` class's ten (#70).
#:
#: The class exists for the ICL-like *under*-restoration case, and what makes
#: that case is low surface brightness, not a low total flux. An n=1 profile
#: has no cusp, so at this radius it spreads its light about seven times more
#: thinly than the n=4 class does at the same peak SNR — the same ladder, a
#: seventh of the peak-pixel share, seven times the total flux. That contrast
#: is the whole point of running the two classes side by side.
#:
#: The size is bounded from above by the ``small`` tier, not by the ``full``
#: one: three grid steps on a 512-pixel image leave the least room, and the
#: brightest halo's wings have to stay inside one cell there too. 0.60" was
#: tried and passes :func:`check_grid_spacing` by only fifteen pixels once the
#: jitter has pushed two neighbours together; 0.75" is refused outright.
SERSIC_N1_HALO_HALF_LIGHT_RADIUS = 0.50

#: The ``sersic_n1_halo`` truncation, as a multiple of its half-light radius.
#: Smaller than the n=4 factor because an exponential falls off far faster
#: than a de Vaucouleurs profile: six half-light radii puts the cut at about a
#: twentieth of the background sigma for the brightest halo on the ladder,
#: comfortably invisible, where :data:`SERSIC_N4_TRUNCATION_FACTOR` needs
#: eight to reach a quarter of sigma. Five was tried and lands at a quarter of
#: sigma here too, but buys no isolation: the extra edge margin six costs is
#: repaid by the wing radius it does not have to cover.
SERSIC_N1_HALO_TRUNCATION_FACTOR = 6.0

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


def psf_margin_pixels() -> float:
    """Room to reserve for the PSF's own reach, in pixels.

    The same number for every class, because #67's PSF is: a single synthetic
    Moffat at a fixed FWHM, and not a swept axis. Stated as a function rather
    than carried on a per-class object, so that no caller can read a choice of
    class into it.
    """
    return PSF_MARGIN_FWHM * PSF_FWHM / PIXEL_SCALE

#: Ellipticity is drawn uniformly in magnitude below this, at a uniform angle.
#: Round sources would make the shape-bias metric of #25 a test of nothing.
MAX_ELLIPTICITY = 0.3

#: How far out the ``point`` class's wing radius is probed, in PSF FWHM.
#:
#: A point source has no truncation to probe past, so the probe needs a stated
#: reach of its own. Thirty FWHM is far enough that the measurement is never
#: the stamp's own edge in disguise: a beta=3.5 Moffat is ten orders of
#: magnitude below its peak there, so :func:`check_grid_spacing` stays a live
#: check on any ladder a ticket could sensibly ask for, which is the property
#: the Sérsic classes get from :data:`WING_PROBE_TRUNCATION_FACTOR`.
POINT_WING_PROBE_FWHM = 30.0


# --------------------------------------------------------------------------
# The source classes (#67, #68, #70)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceProfile:
    """The shape one class draws — everything about a class except its grid.

    The three isolated classes of #67 differ in exactly this object and in
    nothing else: they share the grid, the jitter, the ladder's SNR range, the
    PSF and the noise. Keeping the difference in one table is what lets #70
    add two classes without touching a line of grid logic.

    Radii are in **arcseconds**, matching :class:`Source` and ``sources.csv``.
    """

    #: Sérsic index, or ``0.0`` for a point source. A delta function has no
    #: index, and no half-light radius either; zero records both facts as
    #: numbers rather than leaving a cell of ``sources.csv`` empty. For the
    #: half-light radius zero is literally true — a delta encloses all of its
    #: light at every radius.
    sersic_index: float
    half_light_radius: float
    #: Multiples of the half-light radius at which the profile is cut. Zero
    #: for a point source, which has nothing to cut.
    truncation_factor: float
    #: The ellipticity ceiling the class draws from. Zero for the point class:
    #: a delta function has no shape of its own, so the shape a metric can
    #: measure is the PSF's, and #67's PSF is a circular Moffat. Recording
    #: ``e1 = e2 = 0`` is therefore the point source's true *post-PSF* value,
    #: not a placeholder (#70).
    max_ellipticity: float

    @property
    def is_point(self) -> bool:
        """Whether this class is a delta function rather than a profile."""
        return self.sersic_index == 0.0

    @property
    def truncation_radius(self) -> float:
        """Arcseconds past which the profile is zero; zero for a point source."""
        return self.half_light_radius * self.truncation_factor

    def wing_probe_radius(self, psf_fwhm: float) -> float:
        """How far out the wing radius is measured, in arcseconds.

        Deliberately well past whatever gets rendered, so that the measured
        radius is where the source really falls under the background and not
        wherever the render happened to stop. See
        :data:`WING_PROBE_TRUNCATION_FACTOR`, and
        :data:`POINT_WING_PROBE_FWHM` for the point class, which has no
        truncation to be measured past.
        """
        if self.is_point:
            return POINT_WING_PROBE_FWHM * psf_fwhm
        return WING_PROBE_TRUNCATION_FACTOR * self.half_light_radius


#: The three isolated classes of #67, keyed by class name. ``blend`` is absent
#: because it is not a profile: it draws two of these and varies separation and
#: flux ratio between them, so it is a layout, and it is built below.
SOURCE_PROFILES = {
    'sersic_n4': SourceProfile(
        sersic_index=4.0,
        half_light_radius=SERSIC_N4_HALF_LIGHT_RADIUS,
        truncation_factor=SERSIC_N4_TRUNCATION_FACTOR,
        max_ellipticity=MAX_ELLIPTICITY,
    ),
    'point': SourceProfile(
        sersic_index=0.0,
        half_light_radius=0.0,
        truncation_factor=0.0,
        max_ellipticity=0.0,
    ),
    'sersic_n1_halo': SourceProfile(
        sersic_index=1.0,
        half_light_radius=SERSIC_N1_HALO_HALF_LIGHT_RADIUS,
        truncation_factor=SERSIC_N1_HALO_TRUNCATION_FACTOR,
        max_ellipticity=MAX_ELLIPTICITY,
    ),
}


# --------------------------------------------------------------------------
# The blend class (#67, #71)
# --------------------------------------------------------------------------

#: The one class of #67 that is a layout rather than a profile, named here
#: because it is the only value :func:`build_recipe` branches on.
BLEND_CLASS = 'blend'

#: The classes the recipe can lay out. Since #71 that is every class #67
#: specifies, so no class is refused today; the tuple stays because a name
#: added to :data:`SOURCE_CLASSES` without a layout should be refused with the
#: message that says so rather than crash on a missing key.
BUILT_CLASSES = tuple(SOURCE_PROFILES) + (BLEND_CLASS,)

#: The profile the bright half of a pair draws — the class whose
#: over-subtraction bowl #25 names as the primary known failure mode. The hole
#: this class maps is then literally the same hole the isolated ``sersic_n4``
#: entry measures, so a blend result and an isolated one are read against each
#: other rather than against two different bright sources.
BLEND_BRIGHT_PROFILE = 'sersic_n4'

#: The profile the faint half draws. This is a *measurement* decision rather
#: than an astrophysical one: a point source's whole true flux sits inside one
#: PSF footprint, so "how much of the companion survived" stays a statement
#: about the hole even at the tightest separation, where the companion is
#: buried in the bright source's core. An extended companion's own wings would
#: overlap the bright source at every separation on the ladder, and no
#: measurement could then separate light lost to over-subtraction from light
#: that was never separable in the first place.
#:
#: It also costs the class no new shape numbers, and no new grid room: a point
#: source has no truncation to clear, so the pair fits the isolated classes'
#: own margins and the separation ladder can reach as far as the refusals
#: allow.
BLEND_COMPANION_PROFILE = 'point'

#: The bright component's peak SNR, held fixed over the whole grid per #67 —
#: the brightest rung of the ladder the isolated classes climb, and not a
#: number of its own. Brightness binning is the isolated classes' job, and
#: reusing their top rung is what lets a blend result at the widest separation
#: be read straight against the isolated ``sersic_n4`` entry's first column.
BLEND_BRIGHT_PEAK_SNR = PEAK_SNR_BRIGHTEST

#: The companion's peak SNR at the bright end of the flux-ratio axis: an order
#: of magnitude below the bright component, which is what makes the pair a
#: bright-plus-faint one rather than two comparable sources. The faint end of
#: the axis is :data:`PEAK_SNR_FAINTEST`, the same marginally-detectable rung
#: the isolated ladders end on.
#:
#: The axis is *recorded* as a flux ratio, which is what #67 names it, but it
#: is decided in peak SNR so that both of its ends are anchored to the
#: detectability the whole gallery is laid out by. The two are not the same
#: number: the companion is a point source and concentrates its light some
#: eight times more than the bright Sérsic, so a companion at a tenth of the
#: bright component's peak carries about a hundredth of its flux.
BLEND_COMPANION_PEAK_SNR_BRIGHTEST = 100.0

#: The tightest separation on the ladder, in PSF FWHM. One resolution element
#: is the case no deblender can win — the pair is a single blob and the
#: companion sits inside the bright source's core — and the map needs that
#: corner as much as it needs the clean one. The widest separation is not a
#: constant: it is the bright component's own wing radius, measured rather
#: than chosen, so the ladder ends exactly at the boundary the class exists to
#: locate.
BLEND_SEPARATION_MIN_FWHM = 1.0


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

    #: The three columns #67 adds for the ``blend`` class: the other member of
    #: this source's pair, and the two numbers that describe the pair. All
    #: three are ``None`` for a source that has no companion — an absence, not
    #: a measurement of zero, which is the same distinction that keeps the
    #: write layer from putting the columns in an isolated class's table at
    #: all.
    companion_id: int | None = None
    #: The distance between the two members of the pair, in **pixels** — the
    #: unit ``x`` and ``y`` are in, since this is the distance between two of
    #: them.
    separation: float | None = None
    #: The pair's faint-to-bright *total flux* ratio. Recorded identically on
    #: both rows because it is a property of the pair and not of one member:
    #: reading it off the bright row and off the faint row has to give the same
    #: number.
    flux_ratio: float | None = None

    @property
    def is_point(self) -> bool:
        """Whether this source is a delta function rather than a profile.

        A Sérsic index of zero is the recipe's sentinel for the ``point``
        class, and no real profile takes it. The render layer asks this rather
        than repeating the comparison, so the sentinel is written down in one
        place on the object that carries it (#70).
        """
        return self.sersic_index == 0.0


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

    ``magnitude_ladder`` and ``peak_snr_ladder`` hold one step per grid
    *column*, whatever the class. For the isolated classes that is the source's
    own brightness; for ``blend`` it is the **companion's**, because the
    column axis there is the flux ratio and the bright component does not move
    across the grid at all. The bright component is recorded once, in
    ``bright_magnitude`` and ``bright_peak_snr``.
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
    #: The ``blend`` class's two grid axes, and empty for every other class.
    #: ``separation_ladder`` is in pixels, one step per grid row;
    #: ``flux_ratio_ladder`` is faint-to-bright total flux, one step per
    #: column.
    separation_ladder: tuple[float, ...] = ()
    flux_ratio_ladder: tuple[float, ...] = ()
    #: The ``blend`` class's fixed bright component, and ``None`` elsewhere.
    #: Not a ladder, because the grid deliberately does not step it (#67).
    bright_magnitude: float | None = None
    bright_peak_snr: float | None = None

    @property
    def is_blend(self) -> bool:
        """Whether this entry lays out pairs rather than isolated sources."""
        return self.source_class == BLEND_CLASS

    @property
    def total_flux(self) -> float:
        """The analytic sum the rendered truth image must reproduce.

        The generator compares the convolved truth image's total against this
        and fails loudly outside tolerance (#68). Because the Sérsic profiles
        are truncated and the PSF is normalised, this is an exact statement
        about the intended image and not an approximation of one.
        """
        return float(sum(source.flux for source in self.sources))

    @property
    def flux_groups(self) -> tuple[tuple[Source, ...], ...]:
        """The entry's sources binned by the flux each is supposed to carry.

        The generator checks its drawn flux against the analytic sum **per
        bin** rather than once for the whole image (#71). One number for a
        whole entry is dominated by its brightest sources, and the gallery is
        deliberately built out of sources that differ by three orders of
        magnitude: the faintest column of an isolated class carries about three
        parts in a thousand of the total, and a blend entry's twenty-five
        companions about five parts in a thousand between them. Every one of
        them could fail to draw and a one per cent tolerance on the total would
        still be met, which would make the check pass without saying anything
        about the faint end — the end the gallery exists to measure.

        Every source is in exactly one bin, because the bins are built from
        the sources. Ordered brightest bin first; inside a bin, table order.
        """
        groups: dict[float, list[Source]] = {}
        for source in self.sources:
            groups.setdefault(source.flux, []).append(source)
        return tuple(tuple(groups[flux]) for flux in sorted(groups, reverse=True))

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

        The ``blend`` keys are added for that class only, rather than carried
        as empty values by every entry. They describe axes no other class has,
        and writing them everywhere would make every entry already on disk
        *stale* — a rebuild of six entries whose pixels #71 does not move,
        which is the same waste :data:`PROVENANCE_KEYS` exists to prevent for
        an upgrade.
        """
        manifest = {
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

        if self.is_blend:
            manifest.update(
                {
                    # ``magnitude_ladder`` and ``peak_snr_ladder`` are shared
                    # fields whose subject changes with the class, so the
                    # subject is written down rather than left to be inferred
                    # from the presence of the keys below it.
                    'ladders_describe': 'companion',
                    'n_pairs': len(self.sources) // 2,
                    'bright_magnitude': float(self.bright_magnitude),
                    'bright_peak_snr': float(self.bright_peak_snr),
                    'separation_ladder': [
                        float(value) for value in self.separation_ladder
                    ],
                    'flux_ratio_ladder': [
                        float(value) for value in self.flux_ratio_ladder
                    ],
                }
            )

        return manifest


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
#: :data:`SERSIC_N4_TRUNCATION_FACTOR`. Anything past it reads as "wings wider
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


@lru_cache(maxsize=16)
def convolved_unit_profile(
    sersic_index: float,
    half_light_radius_pixels: float,
    extent_pixels: float,
    psf_fwhm_pixels: float,
    psf_beta: float,
    oversample: int = PROFILE_OVERSAMPLE,
) -> np.ndarray:
    """A unit-flux source convolved with the Moffat PSF, per pixel.

    ``sersic_index`` of zero asks for a point source, and
    ``half_light_radius_pixels`` is then ignored: a delta function convolved
    with the PSF is the PSF, which is exactly the observed profile of the
    ``point`` class (#70).

    ``extent_pixels`` is how far the source itself reaches — the truncation
    radius for a Sérsic, and for a point source simply how much of the PSF's
    skirt to model. Either way the grid is that much wider than the PSF margin.

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
    half_width = int(np.ceil(extent_pixels)) + psf_margin
    side = 2 * half_width + 1

    fine_side = side * oversample
    axis = (np.arange(fine_side) - (fine_side - 1) / 2.0) / oversample
    radius = np.hypot(*np.meshgrid(axis, axis, indexing='ij'))

    if sersic_index == 0.0:
        # A delta function: one fine-grid cell holds all of it. The grid has
        # an odd side at every oversampling used here, so there is a cell
        # exactly on centre and the source needs no interpolation.
        source = np.zeros_like(radius)
        source[fine_side // 2, fine_side // 2] = 1.0
    else:
        scale_radius = truncated_scale_radius(
            sersic_index, half_light_radius_pixels, extent_pixels
        )
        source = np.exp(
            -((np.maximum(radius, 1e-12) / scale_radius) ** (1.0 / sersic_index))
        )
        source[radius > extent_pixels] = 0.0
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


def _draw_ellipticity(
    rng: np.random.Generator, max_ellipticity: float
) -> tuple[float, float]:
    """A random ellipticity below ``max_ellipticity``, uniform in angle.

    A ceiling of zero returns ``(0.0, 0.0)``, which is the point class's true
    post-PSF shape and not an absence of one: #67's PSF is a circular Moffat,
    so a delta function convolved with it is round.
    """
    magnitude = rng.uniform(0.0, max_ellipticity)
    angle = rng.uniform(0.0, np.pi)
    if max_ellipticity == 0.0:
        # Returned outright rather than multiplied through, which would put
        # signed zeros in ``sources.csv`` for half the sources. The draws above
        # still happen, so the jitter that follows them lands in the same place
        # whatever a class's ellipticity ceiling is.
        return 0.0, 0.0
    return float(magnitude * np.cos(2.0 * angle)), float(magnitude * np.sin(2.0 * angle))


def _refuse_unbuilt_axes(axes: EntryAxes) -> None:
    """Every axis value no ticket has decided, refused with the reason why."""
    tier, source_class, regime, realisation = (
        axes.tier,
        axes.source_class,
        axes.regime,
        axes.realisation,
    )
    # The tier is checked but never deferred. #67 decides both tiers outright —
    # an image size and a grid side each, and nothing else — so there is no
    # undecided number here to guess at. Since #71 the class axis is complete
    # too, and only the regime and the realisation are still narrowed.
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
    if source_class not in BUILT_CLASSES:
        raise AxisNotBuiltError(
            'the %r class is specified in #67 but not built: it has neither an '
            'entry in SOURCE_PROFILES nor a layout of its own, so its '
            'parameters belong to the ticket that decides them.' % source_class
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


@dataclass(frozen=True)
class _ProfileOptics:
    """What one class's profile decides before any flux is chosen.

    Split out of :func:`build_recipe` by #71, which needs it for *two*
    profiles at once: the bright component's, which the blend grid's geometry
    and the pair's wing radius come from, and the companion's, which turns a
    peak SNR into a flux.

    Every field is in pixels except ``peak_fraction``, which is a share of a
    source's total flux, and ``probe_profile``, which is a per-pixel image of a
    unit-flux source.
    """

    peak_fraction: float
    probe_profile: np.ndarray
    truncation_pixels: float
    edge_margin: float


def _profile_optics(profile: SourceProfile) -> _ProfileOptics:
    """Model one class's profile: its peak share, its wings and its margin.

    ``edge_margin`` is the room a jittered source of this class needs so that
    none of it is cut by the border and the analytic flux sum stays
    recoverable. A point source contributes nothing to the truncation term and
    lives entirely inside the PSF margin, which is why its grid is the
    roomiest.
    """
    half_light_radius_pixels = profile.half_light_radius / PIXEL_SCALE
    truncation_pixels = profile.truncation_radius / PIXEL_SCALE
    psf_fwhm_pixels = PSF_FWHM / PIXEL_SCALE

    unit_profile = convolved_unit_profile(
        sersic_index=profile.sersic_index,
        half_light_radius_pixels=half_light_radius_pixels,
        extent_pixels=truncation_pixels,
        psf_fwhm_pixels=psf_fwhm_pixels,
        psf_beta=PSF_BETA,
    )

    # Measured on a far wider probe, so the render's own cut does not bound it.
    probe_profile = convolved_unit_profile(
        sersic_index=profile.sersic_index,
        half_light_radius_pixels=half_light_radius_pixels,
        extent_pixels=profile.wing_probe_radius(PSF_FWHM) / PIXEL_SCALE,
        psf_fwhm_pixels=psf_fwhm_pixels,
        psf_beta=PSF_BETA,
        oversample=WING_PROBE_OVERSAMPLE,
    )

    return _ProfileOptics(
        peak_fraction=_peak_fraction(unit_profile),
        probe_profile=probe_profile,
        truncation_pixels=truncation_pixels,
        edge_margin=(
            truncation_pixels
            + psf_margin_pixels()
            + JITTER_PIXELS
            + EDGE_PAD_PIXELS
        ),
    )


def _flux_ladder(
    peak_snr_ladder: np.ndarray, peak_fraction: float, sigma: float
) -> np.ndarray:
    """The total fluxes that put the given peak SNRs in the brightest pixel.

    One ladder for an isolated class's sources, and one for the blend class's
    companions. Both are decided in peak SNR because that is the quantity #67
    lays the gallery out by, and both become a flux the same way.
    """
    return peak_snr_ladder * sigma / peak_fraction


def _jittered_position(
    edge_margin: float, index: int, spacing: float, rng: np.random.Generator
) -> float:
    """One grid coordinate, offset from its nominal cell position.

    The jitter buys independent samples rather than copies: sub-pixel phase
    relative to the pixel grid is not the same for two sources, and neither is
    the noise under them. Both layouts jitter — the isolated classes their
    sources, the blend class its pair centres.
    """
    return (
        edge_margin
        + index * spacing
        + float(rng.uniform(-JITTER_PIXELS, JITTER_PIXELS))
    )


def _entry_fields(
    axes: EntryAxes,
    seed: int,
    tier_geometry: Tier,
    noise: NoiseParameters,
    grid_spacing: float,
) -> dict:
    """The fields every entry carries, whatever its class lays out.

    Split out by #71 so that the two layouts — the isolated classes' brightness
    ladder and the blend class's pairs — state their differences and nothing
    else.
    """
    return {
        'tier': axes.tier,
        'source_class': axes.source_class,
        'regime': axes.regime,
        'realisation': axes.realisation,
        'seed': seed,
        'image_shape': (tier_geometry.image_size, tier_geometry.image_size),
        'pixel_scale': PIXEL_SCALE,
        'grid_shape': (tier_geometry.grid_side, tier_geometry.grid_side),
        'grid_spacing': grid_spacing,
        'jitter_amplitude': JITTER_PIXELS,
        'psf_type': PSF_TYPE,
        'psf_fwhm': PSF_FWHM,
        'psf_beta': PSF_BETA,
        'zero_point': ZERO_POINT,
        'noise': noise,
    }


def _build_isolated_entry(
    axes: EntryAxes, seed: int, tier_geometry: Tier, noise: NoiseParameters
) -> Recipe:
    """One entry of a class that lays isolated sources on a brightness ladder.

    Brightness varies along one grid axis, position jitter along the other, so
    the full tier gives five brightness steps with five samples each — the
    shape "bin residuals by SNR" wants (#67). The ladder is geometric in peak
    SNR, so a column index reads as a fixed factor in brightness.
    """
    rng = np.random.default_rng(seed)
    grid_side = tier_geometry.grid_side
    profile = SOURCE_PROFILES[axes.source_class]
    optics = _profile_optics(profile)
    edge_margin = optics.edge_margin
    spacing = tier_geometry.grid_spacing(edge_margin)

    peak_snr_ladder = np.geomspace(
        PEAK_SNR_BRIGHTEST, PEAK_SNR_FAINTEST, grid_side
    )
    flux_ladder = _flux_ladder(peak_snr_ladder, optics.peak_fraction, noise.sigma)
    magnitude_ladder = tuple(
        _magnitude_from_flux(flux, ZERO_POINT) for flux in flux_ladder
    )

    wing_radius = _wing_radius(
        optics.probe_profile * float(flux_ladder[0]), brightness_floor=noise.sigma
    )
    # A point source is not truncated, so there is no cut that could show as a
    # hard edge and nothing for this check to be about. Every profile class
    # answers to it.
    if not profile.is_point:
        check_truncation(
            wing_radius=wing_radius, truncation_radius=optics.truncation_pixels
        )
    check_grid_spacing(wing_radius=wing_radius, grid_spacing=spacing)

    sources = []
    source_id = 0
    for row in range(grid_side):
        for column in range(grid_side):
            e1, e2 = _draw_ellipticity(rng, profile.max_ellipticity)
            sources.append(
                Source(
                    source_id=source_id,
                    column=column,
                    row=row,
                    x=_jittered_position(edge_margin, column, spacing, rng),
                    y=_jittered_position(edge_margin, row, spacing, rng),
                    magnitude=magnitude_ladder[column],
                    flux=float(flux_ladder[column]),
                    sersic_index=profile.sersic_index,
                    half_light_radius=profile.half_light_radius,
                    e1=e1,
                    e2=e2,
                    truncation_radius=profile.truncation_radius,
                    peak_snr=float(peak_snr_ladder[column]),
                )
            )
            source_id += 1

    sources = tuple(sources)
    check_image_margin(
        sources,
        tier_geometry.image_size,
        margin=psf_margin_pixels(),
        pixel_scale=PIXEL_SCALE,
    )

    return Recipe(
        **_entry_fields(axes, seed, tier_geometry, noise, spacing),
        sources=sources,
        magnitude_ladder=magnitude_ladder,
        peak_snr_ladder=tuple(float(value) for value in peak_snr_ladder),
        wing_radius=wing_radius,
    )


def _build_blend_entry(
    axes: EntryAxes, seed: int, tier_geometry: Tier, noise: NoiseParameters
) -> Recipe:
    """One entry of the ``blend`` class — a pair of sources per grid cell (#71).

    Separation steps down the rows and the pair's flux ratio across the
    columns, with the bright component held at :data:`BLEND_BRIGHT_PEAK_SNR`
    over the whole grid. That is the two-dimensional map #67 asks for: where
    over-subtraction around the bright source starts consuming its neighbour.

    Neither axis is brightness, so neither can buy the independent samples the
    isolated classes get from their jitter row. The **pair centre** is jittered
    instead, and the companion is placed at a random position angle, so no two
    cells repeat a sub-pixel phase or an orientation while both grid axes stay
    spent on what the class is for.

    Both ends of the separation ladder are measured rather than chosen: one PSF
    FWHM at the tight end, and the bright component's own wing radius at the
    wide end — the radius at which it finally drops under the background, and
    so exactly the boundary this class exists to locate. The widest row
    therefore *lands on* that boundary rather than passing it: what lies
    outside it is the isolated ``sersic_n4`` entry, which is the same bright
    source with no companion at all.
    """
    rng = np.random.default_rng(seed)
    grid_side = tier_geometry.grid_side
    bright_profile = SOURCE_PROFILES[BLEND_BRIGHT_PROFILE]
    companion_profile = SOURCE_PROFILES[BLEND_COMPANION_PROFILE]
    bright_optics = _profile_optics(bright_profile)
    companion_optics = _profile_optics(companion_profile)

    # The bright component sits at the cell centre and is much the wider of
    # the two, so the grid is laid out on its margin — which is the isolated
    # ``sersic_n4`` entry's margin, and so its spacing. The companion is
    # displaced outward from there by up to the whole separation, and
    # ``check_image_margin`` below is what says that displacement still clears
    # the border.
    edge_margin = bright_optics.edge_margin
    spacing = tier_geometry.grid_spacing(edge_margin)

    bright_flux = float(
        _flux_ladder(
            np.array([BLEND_BRIGHT_PEAK_SNR]),
            bright_optics.peak_fraction,
            noise.sigma,
        )[0]
    )
    bright_magnitude = _magnitude_from_flux(bright_flux, ZERO_POINT)

    wing_radius = _wing_radius(
        bright_optics.probe_profile * bright_flux, brightness_floor=noise.sigma
    )
    check_truncation(
        wing_radius=wing_radius, truncation_radius=bright_optics.truncation_pixels
    )
    check_grid_spacing(wing_radius=wing_radius, grid_spacing=spacing)

    separation_ladder = np.geomspace(
        BLEND_SEPARATION_MIN_FWHM * PSF_FWHM / PIXEL_SCALE, wing_radius, grid_side
    )
    # The refusal the class needs of its own: the isolated one above asks
    # whether a *source* reaches its neighbour's cell, and this asks whether a
    # whole pair does. The widest separation is the binding one.
    check_pair_spacing(
        wing_radius=wing_radius,
        separation=float(separation_ladder[-1]),
        grid_spacing=spacing,
        jitter_amplitude=JITTER_PIXELS,
    )

    companion_peak_snr_ladder = np.geomspace(
        BLEND_COMPANION_PEAK_SNR_BRIGHTEST, PEAK_SNR_FAINTEST, grid_side
    )
    companion_flux_ladder = _flux_ladder(
        companion_peak_snr_ladder, companion_optics.peak_fraction, noise.sigma
    )
    flux_ratio_ladder = companion_flux_ladder / bright_flux
    companion_magnitude_ladder = tuple(
        _magnitude_from_flux(flux, ZERO_POINT) for flux in companion_flux_ladder
    )

    sources = []
    for row in range(grid_side):
        separation = float(separation_ladder[row])
        for column in range(grid_side):
            flux_ratio = float(flux_ratio_ladder[column])
            e1, e2 = _draw_ellipticity(rng, bright_profile.max_ellipticity)
            x = _jittered_position(edge_margin, column, spacing, rng)
            y = _jittered_position(edge_margin, row, spacing, rng)
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            companion_e1, companion_e2 = _draw_ellipticity(
                rng, companion_profile.max_ellipticity
            )

            bright_id = len(sources)
            companion_id = bright_id + 1
            sources.append(
                Source(
                    source_id=bright_id,
                    column=column,
                    row=row,
                    x=x,
                    y=y,
                    magnitude=bright_magnitude,
                    flux=bright_flux,
                    sersic_index=bright_profile.sersic_index,
                    half_light_radius=bright_profile.half_light_radius,
                    e1=e1,
                    e2=e2,
                    truncation_radius=bright_profile.truncation_radius,
                    peak_snr=float(BLEND_BRIGHT_PEAK_SNR),
                    companion_id=companion_id,
                    separation=separation,
                    flux_ratio=flux_ratio,
                )
            )
            sources.append(
                Source(
                    source_id=companion_id,
                    column=column,
                    row=row,
                    x=x + separation * float(np.cos(angle)),
                    y=y + separation * float(np.sin(angle)),
                    magnitude=companion_magnitude_ladder[column],
                    flux=float(companion_flux_ladder[column]),
                    sersic_index=companion_profile.sersic_index,
                    half_light_radius=companion_profile.half_light_radius,
                    e1=companion_e1,
                    e2=companion_e2,
                    truncation_radius=companion_profile.truncation_radius,
                    peak_snr=float(companion_peak_snr_ladder[column]),
                    companion_id=bright_id,
                    separation=separation,
                    flux_ratio=flux_ratio,
                )
            )

    sources = tuple(sources)
    check_image_margin(
        sources,
        tier_geometry.image_size,
        margin=psf_margin_pixels(),
        pixel_scale=PIXEL_SCALE,
    )

    return Recipe(
        **_entry_fields(axes, seed, tier_geometry, noise, spacing),
        sources=sources,
        magnitude_ladder=companion_magnitude_ladder,
        peak_snr_ladder=tuple(float(value) for value in companion_peak_snr_ladder),
        wing_radius=wing_radius,
        separation_ladder=tuple(float(value) for value in separation_ladder),
        flux_ratio_ladder=tuple(float(value) for value in flux_ratio_ladder),
        bright_magnitude=bright_magnitude,
        bright_peak_snr=float(BLEND_BRIGHT_PEAK_SNR),
    )


def build_recipe(
    tier: str, source_class: str, regime: str, realisation: int
) -> Recipe:
    """Describe one gallery entry completely, without drawing a single pixel.

    Two layouts, and the class picks between them: the isolated classes lay one
    source per cell on a brightness ladder, and ``blend`` lays a pair per cell
    over separation and flux ratio. Everything before the branch — the tier's
    geometry, the noise regime and the derived seed — is common to both.

    Raises :class:`AxisNotBuiltError` for an axis value no ticket has decided,
    and one of :class:`GridTooTightError`, :class:`PairsTooCloseError`,
    :class:`TruncationTooTightError` or :class:`SourceClippedError` if the
    layout the decided numbers produce is not the layout the class is for.
    """
    axes = EntryAxes(tier, source_class, regime, realisation)
    _refuse_unbuilt_axes(axes)

    build = _build_blend_entry if source_class == BLEND_CLASS else _build_isolated_entry
    return build(
        axes=axes,
        seed=derive_seed(source_class, regime, realisation),
        tier_geometry=TIERS[tier],
        noise=REGIMES[regime],
    )


# --------------------------------------------------------------------------
# The gallery as a whole (#69)
# --------------------------------------------------------------------------


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

    #67 specifies 72 entries and each ticket decides the numbers for some of
    them; the rest are refused rather than guessed. So "every entry" means
    every entry whose parameters a ticket has written down, and this asks the
    refusal itself rather than keeping a second list beside it. #70 lifted the
    refusals on ``point`` and ``sersic_n1_halo`` and the sweep grew from two
    entries to six by itself, with no list here to forget to update. The
    ticket that decides ``blend`` will do the same.

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
        if any(
            value is not None and getattr(axes, name) != value
            for name, value in wanted.items()
        ):
            continue
        try:
            _refuse_unbuilt_axes(axes)
        except AxisNotBuiltError:
            continue
        selected.append(axes)
    return tuple(selected)
