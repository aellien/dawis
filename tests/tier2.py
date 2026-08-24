"""The Tier 2 run definition (#37).

One seeded run of the published configuration on the *whole* 900x900 gallery
image, at the shipping conjugate-gradient setting. This is the science check:
Tier 1 asks whether the carve broke the structure of the pipeline, Tier 2 asks
whether the pipeline still does the published science.

What is specific to Tier 2 lives here — the full image rather than a patch,
and ``CG_MAX_ITER`` at its shipped value. How a golden run is driven and read
back off disk is shared with Tier 1 and lives in ``tests/goldenrun.py``, so
the two tiers differ only where they mean to.

Two things drive a Tier 2 run and they must agree exactly: the comparison test
in ``tests/regression/test_golden_gallery.py``, and the hand-run recording
script ``tests/goldens/record_gallery_golden.py``.

Budget: **a full half hour** on a 2017 laptop. Both the recording and the
first comparison run measured 1641 s; #33's 1167.6 s for this configuration was
measured at seed 0, which restores 394 atoms against this seed's 463. That cost
is why the tier is marked ``slow`` and runs at named carve checkpoints and
nightly in CI (#19), not on every commit.

Nothing here imports pytest: the recording script loads this file directly,
outside a test session.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from goldenrun import (
    GOLDEN_PARAMS,
    SHIPPING_CG_MAX_ITER,
    TEST_SEED,
    GoldenRun,
    run_golden,
)

#: Iteration cap of the conjugate-gradient solver inside
#: ``interscale_tree.CG_minimization``, for Tier 2.
#:
#: Unlike Tier 1's, this is a science claim and not a speed knob: 200 is the
#: value Ellien+2021 ran, `7c5ba8e` cut it to 3 in 2024, and #32 reverted that
#: drift on ``legacy-seeded``. At 200 the ``sigma_flux=1e-3`` convergence test
#: inside the solver is live rather than vestigial, and atom fluxes sit 25-29%
#: higher than at 3 (#29, measured on #33). Tier 2 exists to pin *this* value's
#: behaviour, so it is never lowered to fit a budget.
#:
#: It is still forced rather than inherited: the default is 200 on
#: ``legacy-seeded`` but 3 on the drifted tree the refactor branched from, so
#: inheriting it would compare a golden recorded at 200 against a run at 3.
CG_MAX_ITER = SHIPPING_CG_MAX_ITER

#: Name of the FITS file the gallery image is written to before every Tier 2
#: run. The image is copied into the run directory rather than read in place,
#: so the pipeline never writes its outputs next to the packaged data.
TIER2_INFILE = "tier2_gallery.fits"

#: Tier 2 runs the shared published configuration unchanged — the same
#: parameters as Tier 1. Only the input and ``CG_MAX_ITER`` differ.
TIER2_PARAMS = dict(GOLDEN_PARAMS)


def run_tier2(dawis, image: np.ndarray, work_dir: Path, seed: int = TEST_SEED,
              cg_max_iter: int = CG_MAX_ITER) -> GoldenRun:
    """Run the Tier 2 configuration on ``image`` and return what it wrote."""
    return run_golden(dawis, image, work_dir, TIER2_INFILE, cg_max_iter, seed)
