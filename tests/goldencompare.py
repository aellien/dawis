"""How a recorded golden is compared, shared by both regression tiers.

Tier 1 and Tier 2 record the same four things — a convergence trace, an atom
catalogue, a restored image and a residual image — and #3 lays down one rule
for comparing them at either tier: float tolerance on images and per-atom
quantities, exact on atom count, iteration count, positions, levels, the
chosen synthesis operator and the convergence flag.

The rule lives here once so the two tiers cannot enforce it differently. The
tolerances themselves are *not* here: each tier reads its own from its
``provenance.toml``, so a golden can never be recorded under one set of numbers
and compared under another.
"""

from __future__ import annotations

import numpy as np

from goldenrun import CONVERGENCE_APPROXIMATE

#: Per-atom columns compared exactly. Positions, levels, the chosen synthesis
#: operator and the convergence flag are decisions, not measurements: a
#: tolerance on them would hide the very drift the tier is looking for.
ATOM_EXACT = (
    "iteration", "index", "level", "bbox_0", "bbox_1", "bbox_2", "bbox_3",
    "flag_convergence", "filter",
)

#: Per-atom columns compared by tolerance — everything measured off the atom.
ATOM_APPROXIMATE = (
    "eccentricity", "sum_wr", "norm_wr", "flux", "peak", "det_err_flux",
)

#: Convergence columns compared exactly.
CONVERGENCE_EXACT = ("level", "iteration", "n_atoms")


def assert_image_matches(actual, recorded, tolerances, what):
    """Compare two images under a golden's recorded image tolerance."""
    np.testing.assert_allclose(
        actual,
        recorded,
        rtol=tolerances["image_rtol"],
        atol=tolerances["image_atol"],
        err_msg=f"{what} differs from the recorded golden",
    )


def assert_convergence_matches(actual_rows, recorded_rows, tolerances):
    """Every iteration ran at the same level and produced the same atoms.

    The convergence numbers are what the greedy loop steers on, so a drift
    here is the earliest visible symptom of a behaviour change.
    """
    assert len(actual_rows) == len(recorded_rows)

    for actual, recorded in zip(actual_rows, recorded_rows):
        where = f"iteration {recorded['iteration']}"
        for name in CONVERGENCE_EXACT:
            assert actual[name] == recorded[name], f"{where}: {name}"
        for name in CONVERGENCE_APPROXIMATE:
            np.testing.assert_allclose(
                actual[name],
                recorded[name],
                rtol=tolerances["convergence_rtol"],
                atol=tolerances["convergence_atol"],
                err_msg=f"{where}: {name}",
            )


def assert_atoms_match(actual_rows, recorded_rows, tolerances):
    """Every atom is restored at the same place, at the same scale, with the
    same flux, in the same order."""
    assert len(actual_rows) == len(recorded_rows)

    for actual, recorded in zip(actual_rows, recorded_rows):
        where = f"atom {recorded['index']} of iteration {recorded['iteration']}"
        for name in ATOM_EXACT:
            assert actual[name] == recorded[name], f"{where}: {name}"
        for name in ATOM_APPROXIMATE:
            np.testing.assert_allclose(
                actual[name],
                recorded[name],
                rtol=tolerances["atom_rtol"],
                atol=tolerances["atom_atol"],
                err_msg=f"{where}: {name}",
            )
