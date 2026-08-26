"""The Tier 1 run definition (#36).

One seeded run of the published configuration on a 256x256 patch of the
gallery image. What is specific to Tier 1 lives here — the cutout bounds, the
pinned solver knob, the budget. How a golden run is driven and read back off
disk is shared with Tier 2 and lives in ``tests/goldenrun.py``.

Two things drive a Tier 1 run and they must agree exactly: the comparison test
in ``tests/regression/test_golden_cutout.py``, and the hand-run recording
script ``tests/goldens/record_cutout_golden.py``. Both import this module, so
a golden cannot be recorded under one configuration and compared under
another.

Nothing here imports pytest: the recording script loads this file directly,
outside a test session.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from goldenrun import GOLDEN_PARAMS, TEST_SEED, GoldenRun, run_golden

#: Centre of the Tier 1 cutout, as ``(x, y)``, supplied by Amaël on #3.
CUTOUT_CENTRE = (222, 242)

#: Side of the square Tier 1 cutout, in pixels.
CUTOUT_SIZE = 256

#: The resulting bounds, ``((y0, y1), (x0, x1))`` == ``((114, 370), (94, 350))``.
#:
#: Centre, size and bounds are all stated because the FITS array is indexed
#: ``(ny, nx)`` while the coordinates are ``(x, y)`` — so every misreading of
#: those two numbers still lands on a valid source-rich patch, and produces a
#: different golden *silently* rather than failing. The recorded
#: ``input_cutout.fits`` is compared for the same reason.
CUTOUT_SLICE = (
    (CUTOUT_CENTRE[1] - CUTOUT_SIZE // 2, CUTOUT_CENTRE[1] + CUTOUT_SIZE // 2),
    (CUTOUT_CENTRE[0] - CUTOUT_SIZE // 2, CUTOUT_CENTRE[0] + CUTOUT_SIZE // 2),
)

#: Iteration cap of the conjugate-gradient solver inside
#: ``InterscaleTree.CG_minimization``, for Tier 1 only.
#:
#: **A speed knob, not a science claim.** Tier 1's job is catching structural
#: breakage during the carve, which it does equally well at any iteration
#: count; the shipping value of 200 costs about six times more per atom (#33)
#: and would push this tier past its three-minute budget. Tier 2 (#37) runs
#: the shipping value and is the science check.
CG_MAX_ITER = 3

#: Name of the FITS file the cutout is written to before every Tier 1 run.
TIER1_INFILE = "tier1_cutout.fits"

#: Tier 1 runs the shared published configuration unchanged. Named separately
#: because ``conftest``'s ``seeded_config`` fixture is written against it.
TIER1_PARAMS = dict(GOLDEN_PARAMS)


def cutout_of(image: np.ndarray) -> np.ndarray:
    """Slice the agreed Tier 1 cutout out of the full gallery image."""
    (y0, y1), (x0, x1) = CUTOUT_SLICE
    return image[y0:y1, x0:x1]


def run_tier1(dawis, image: np.ndarray, work_dir: Path, seed: int = TEST_SEED,
              cg_max_iter: int = CG_MAX_ITER,
              n_threads: int | None = None) -> GoldenRun:
    """Run the Tier 1 configuration on ``image`` and return what it wrote.

    ``n_threads`` replaces the maths-library thread count; ``None`` keeps
    :data:`goldenrun.GOLDEN_N_THREADS`, which is what the golden was recorded
    at and what the comparison runs at.
    """
    return run_golden(dawis, image, work_dir, TIER1_INFILE, cg_max_iter, seed,
                      n_threads=n_threads)
