"""The Tier 2 run definition (#37).

One seeded run of the published configuration on the *whole* 900x900 gallery
image, at the shipping conjugate-gradient setting. This is the science check:
Tier 1 asks whether the carve broke the structure of the pipeline, Tier 2 asks
whether the pipeline still does the published science.

What is specific to Tier 2 lives here — the full image rather than a patch,
and ``CG_MAX_ITER`` at its shipped value. How a golden run is driven and read
back off disk is shared with Tier 1 and lives in ``tests/goldenrun.py``, so
the two tiers differ only where they mean to.

Three things drive a Tier 2 run and they must agree exactly: the comparison
test in ``tests/regression/test_golden_gallery.py``, the parallel-consistency
check in ``tests/regression/test_parallel_consistency.py``, and the hand-run
recording script ``tests/goldens/record_gallery_golden.py``.

Budget: **about fifteen minutes** on a 2017 laptop — 643 s for the #105
recording and 625 s for the whole ``-m slow`` selection. Before #105 ratcheted
the goldens to the shipping arm it was a full half hour, 1641 s either way.

Do not read that drop as what the new arm costs: it confounds the arm with a
greedy loop that runs 16 iterations rather than 27, and with #101's removal of
``deconv`` — the old recording came from ``legacy-seeded``, where the
``ADJOINT`` branch ran ten Richardson-Lucy steps per atom for 141 of its 463
atoms. #106 is the cost measurement. #33's 1167.6 s is not a clean baseline
either: seed 0, 394 atoms, and no pickle read-back.

Even at a quarter hour this is far too much for a per-commit gate, which is
why the tier is marked ``slow`` and runs at named carve checkpoints and
nightly in CI (#19).

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
#: ``InterscaleTree.CG_minimization``, for Tier 2.
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

#: Workers the parallel half of the consistency check of #57 asks for.
#:
#: Four is the figure the ticket names. It is a fixed number rather than the
#: host's core count because what is asserted is bit-identity: a worker count
#: that moved with the machine would make a failure something the next machine
#: could not reproduce.
#:
#: The backend is left to the worker count to choose, which above one worker is
#: :data:`dawis.parallel.DEFAULT_PARALLEL_BACKEND`. Naming ``'serial'`` instead
#: would still exercise the patching but would never cross a process, and
#: crossing one is half of what could break bit-identity.
PARALLEL_N_WORKERS = 4

#: Maths-library threads the *thread* half of the consistency check of #61
#: asks for, against :data:`goldenrun.GOLDEN_N_THREADS` on the other side.
#:
#: Two, not the host's core count, for the reason
#: :data:`PARALLEL_N_WORKERS` is four: what is asserted is a bound on how far
#: an atom moved, and a thread count that moved with the machine would make a
#: failure something the next machine could not reproduce. Two is also the
#: smallest count that reassociates a sum at all, which is the whole effect
#: under test — a larger one would demonstrate nothing more.
#:
#: The worker count stays at the published 1 on both sides of that comparison,
#: so the thread count is the only thing that differs. Threads apply per
#: process and are not divided among workers, so the two knobs are independent
#: and pairing them would only make a failure ambiguous.
PARALLEL_N_THREADS = 2


def run_tier2(dawis, image: np.ndarray, work_dir: Path, seed: int = TEST_SEED,
              cg_max_iter: int = CG_MAX_ITER,
              n_workers: int | None = None,
              n_threads: int | None = None) -> GoldenRun:
    """Run the Tier 2 configuration on ``image`` and return what it wrote.

    ``n_workers`` replaces the published single worker, which makes the run
    incomparable against the recorded golden — see :func:`goldenrun.run_golden`.
    ``None`` keeps the published value, which is what the golden comparison and
    the recording script both take; #57's parallel half passes
    :data:`PARALLEL_N_WORKERS`.

    ``n_threads`` replaces the maths-library thread count the same way, and
    ``None`` keeps :data:`goldenrun.GOLDEN_N_THREADS`. #61's thread half
    passes :data:`PARALLEL_N_THREADS`; that run is also incomparable against
    the recording, but for a different reason — the worker count cannot move
    an atom at all, while the thread count moves one within tolerance.
    """
    return run_golden(dawis, image, work_dir, TIER2_INFILE, cg_max_iter, seed,
                      n_workers, n_threads)
