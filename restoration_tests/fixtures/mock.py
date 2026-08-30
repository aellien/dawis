"""The mock iteration — one pass from a gallery image to interscale trees.

Deliberately not a faithful reproduction of a DAWIS iteration, and #74 is
explicit about why. The unit under test is one box inside one iteration::

    image -> background -> Anscombe -> a trous -> detection -> trees

and there it stops. No reconstruction, no subtraction, no inpainting, no
greedy loop. The greedy loop exists precisely to compensate for a weak
reconstruction, so measuring the whole loop would hide the thing the map is
about.

What is copied from the pipeline, and what is not
-------------------------------------------------
Everything up to the trees is the pipeline's own sequence, in the pipeline's
own order, with the pipeline's own calls — the background subtraction, the
Anscombe stabilisation fitted once against the input image, the B3-spline à
trous, ``ms_detect_and_deblend`` on the stabilised planes and
``make_regions_full_props`` on the unstabilised ones. Reproducing that by
hand rather than calling the pipeline is the point: the pipeline is a greedy
loop with resume, dumps and convergence in it, and none of that is under
test.

One difference from the pipeline is worth naming. The pipeline decomposes to
``level`` — its outer loop variable, which climbs from ``starting_level`` to
``n_levels`` — so it sees a different depth at every stage. The mock
decomposes once, to :attr:`MockSettings.n_levels`, because a fixture set has
no stages.

``ceps`` is carried and never read
----------------------------------
#75 freezes ``ceps=1e-3`` alongside ``tau`` and ``n_levels``. It is the
greedy loop's convergence threshold — the pipeline compares it against the
convergence monitor's window average — and a single pass has no loop to stop.
It is recorded in :class:`MockSettings` and in every manifest because #75
names it as part of the frozen configuration, and it changes nothing here.
Saying so is cheaper than a later reader deciding it must have done
something.

Determinism
-----------
The Anscombe fit draws, and so does the transform: both fill NaNs from the
noise. #72 froze the gallery's seeds so that every arm sees the same bytes,
and the same argument applies one layer up — a re-dump that moved every tube
would break the pairing the frozen settings exist to protect. So the mock
takes its seed from the settings and threads one generator through both.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

from dawis.cubes import LabelCube, WaveletCube
from dawis.detection import make_regions_full_props, ms_detect_and_deblend
from dawis.restoration.inpainting import subtract_background
from dawis.trees import InterscaleTree, make_interscale_trees
from dawis.wavelets import AnscombeStabilisation, bspl_atrous

__all__ = ['FROZEN_SETTINGS', 'MockIteration', 'MockSettings',
           'run_mock_iteration']

log = logging.getLogger(__name__)


@dataclass(frozen = True)
class MockSettings:
    """Every knob the mock iteration reads, frozen across the whole battery.

    #75 freezes these because a comparison of arms is only paired if every
    arm sees the same tubes — the same argument #72 makes for the gallery's
    seeds, one layer up. They are a record rather than defaults to override:
    a dump under different settings is a different fixture set, and
    :meth:`as_dict` is what puts them in its manifest so the two can never be
    mistaken for each other.

    Attributes
    ----------
    n_levels : int
        Depth of the à trous decomposition. #75's 8.
    conditions : str
        Boundary condition of the transform. The pipeline's default.
    n_sigmas : float
        Detection threshold in sigmas of the noise, per scale.
    lvl_deblend : int
        Scale up to which detections are deblended.
    deblend_contrast : float
        Contrast the deblender splits on.
    tau : float
        Fraction of the brightest eligible intensity above which a region
        becomes a tree root. **0.01, not the 0.1 #75 writes** — see
        :data:`FROZEN_SETTINGS`.
    min_span, max_span : int
        Fewest scales a tree must span, and most a member may sit below its
        root. #75's 2 and 8.
    lvl_sep_big : int
        Scale at and above which both spans are forced to 1.
    ceps : float
        The greedy loop's convergence threshold. Recorded because #75 freezes
        it; read by nothing here, because a single pass has no loop.
    seed : int
        Seed of the generator the Anscombe fit and transform draw from.
    """

    n_levels: int = 8
    conditions: str = 'loop'
    n_sigmas: float = 5
    lvl_deblend: int = 3
    deblend_contrast: float = 0.1
    tau: float = 0.01
    min_span: int = 2
    max_span: int = 8
    lvl_sep_big: int = 6
    ceps: float = 1E-3
    seed: int = 0

    def as_dict(self) -> dict:
        """The settings as plain values, for a fixture manifest."""
        return asdict(self)


#: The settings every fixture in the battery is dumped under.
#:
#: ``tau`` is 0.01 and #75 writes 0.1. Measured this session: at 0.1 one mock
#: iteration on a small-tier entry returns three trees, and all three sit on
#: the brightest magnitude column — the faint columns produce no tree at all,
#: for every one of the three classes. A fixture set with one brightness bin
#: cannot carry #25's brightness-binned residuals, which #67 asks for. 0.01
#: adds the middle column, at six trees per entry. The faintest column, at
#: peak SNR 3, appears at neither value. Amael's call, this session; the
#: freezing is what #75 is protecting, not the number.
FROZEN_SETTINGS = MockSettings()


@dataclass(frozen = True)
class MockIteration:
    """What one pass produced.

    The cubes come back with the trees because a tree does not carry its
    tubes — :meth:`~dawis.trees.construction.InterscaleTree.tubes` cuts them
    out of the cubes on demand — so a caller holding trees and no cubes has
    nothing to dump and nothing to reconstruct.

    Attributes
    ----------
    trees : list of dawis.trees.construction.InterscaleTree
        Every tree the pass built.
    wavelet_cube : dawis.cubes.WaveletCube
        The à trous planes of the background-subtracted image — the
        coefficients, not the stabilised ones detection ran on.
    label_cube : dawis.cubes.LabelCube
        The region labels detection produced.
    det_err_array : numpy.ndarray
        Per-scale detection uncertainty.
    background_mean, background_sigma : float
        The sky level taken out of the image, and its scatter.
    """

    trees: list[InterscaleTree]
    wavelet_cube: WaveletCube
    label_cube: LabelCube
    det_err_array: np.ndarray
    background_mean: float
    background_sigma: float


def run_mock_iteration(
    image: np.ndarray,
    settings: MockSettings = FROZEN_SETTINGS,
) -> MockIteration:
    """Run one mock iteration over one image.

    Parameters
    ----------
    image : numpy.ndarray
        The simulated observation, sky left in — a gallery entry's
        ``image.fits``. Not modified.
    settings : MockSettings, optional
        Default :data:`FROZEN_SETTINGS`, which is what the battery dumps
        under. Overridden in tests, and nowhere else.

    Returns
    -------
    MockIteration
        The trees, and the cubes they were cut from. ``trees`` is empty when
        the pass detected nothing that spans enough scales — an image with no
        structure in it is not an error.
    """

    rng = np.random.RandomState(settings.seed)

    subtracted, mean, sigma = subtract_background(image, n_sigmas = 3,
                                                  bins = 200)

    stabilisation = AnscombeStabilisation.fit(subtracted, max_err = 1E-4,
                                              n_sigmas = 3, rng = rng)
    stabilised = stabilisation.transform(subtracted, rng = rng)

    stabilised_cube = bspl_atrous(stabilised, settings.n_levels, None,
                                  settings.conditions)
    label_cube, det_err_array = ms_detect_and_deblend(
        stabilised_cube,
        n_sigmas = settings.n_sigmas,
        lvl_deblend = settings.lvl_deblend,
        deblend_contrast = settings.deblend_contrast,
    )

    wavelet_cube = bspl_atrous(subtracted, settings.n_levels, None,
                               settings.conditions)

    region_list = make_regions_full_props(wavelet_cube, label_cube)
    trees = []
    if region_list:
        trees = make_interscale_trees(
            region_list, wavelet_cube, label_cube, det_err_array,
            tau = settings.tau,
            min_span = settings.min_span,
            max_span = settings.max_span,
            lvl_sep_big = settings.lvl_sep_big,
            n_workers = 1,
        ) or []

    return MockIteration(
        trees = trees,
        wavelet_cube = wavelet_cube,
        label_cube = label_cube,
        det_err_array = det_err_array,
        background_mean = float(mean),
        background_sigma = float(sigma),
    )
