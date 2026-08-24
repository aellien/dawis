"""Tier 1 — the cutout golden regression (#36).

One seeded run of the published configuration on a 256x256 patch of the
gallery image, compared against the output recorded from ``legacy-seeded``.
This is the gate that runs before every commit of the 1.0 carve: it asserts
what the pipeline *produces* from a given input, never how it produces it, so
functions can move between modules freely and only a behaviour change turns it
red.

Tolerances come from ``tests/goldens/tier1_cutout/provenance.toml``, so the
recording and the comparison cannot drift apart. Per #3 they are float
tolerances on images and per-atom quantities, and exact on atom and iteration
counts.

Budget: about 40 s on a 2017 laptop, held there by ``tier1.CG_MAX_ITER``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from tier1 import Tier1Run


def assert_close(actual, desired, tolerances, what):
    """Compare arrays under the goldens' recorded image tolerance."""
    np.testing.assert_allclose(
        actual,
        desired,
        rtol=tolerances["image_rtol"],
        atol=tolerances["image_atol"],
        err_msg=f"{what} differs from the recorded Tier 1 golden",
    )


def test_cutout_matches_the_recorded_input(cutout, tier1_goldens: Path):
    """The run under test is fed exactly the patch the golden was recorded on.

    See ``tier1.CUTOUT_SLICE`` for why a wrong slice would otherwise pass
    silently.
    """
    recorded = fits.getdata(tier1_goldens / "input_cutout.fits")
    np.testing.assert_array_equal(np.asarray(cutout), recorded)


def test_iteration_count_is_exact(tier1_convergence, golden_convergence):
    """The greedy loop takes the same number of iterations. Exact, per #3."""
    assert len(tier1_convergence) == len(golden_convergence)


def test_atom_count_is_exact(tier1_atoms, golden_atoms):
    """The run produces the same number of atoms. Exact, per #3."""
    assert len(tier1_atoms) == len(golden_atoms)


def test_convergence_trace_matches(tier1_convergence, golden_convergence,
                                   tier1_provenance):
    """Every iteration ran at the same level and produced the same atoms.

    The convergence numbers are what the loop steers on, so a drift here is
    the earliest visible symptom of a behaviour change.
    """
    tolerances = tier1_provenance["tolerances"]
    assert len(tier1_convergence) == len(golden_convergence)

    for actual, expected in zip(tier1_convergence, golden_convergence):
        where = f"iteration {expected['iteration']}"
        for name in ("level", "iteration", "n_atoms"):
            assert actual[name] == expected[name], f"{where}: {name}"
        for name in ("window_normeps", "normeps", "eps", "flux_rec", "flux_res"):
            np.testing.assert_allclose(
                actual[name],
                expected[name],
                rtol=tolerances["convergence_rtol"],
                atol=tolerances["convergence_atol"],
                err_msg=f"{where}: {name}",
            )


def test_atom_catalogue_matches(tier1_atoms, golden_atoms, tier1_provenance):
    """Every atom is restored at the same place, at the same scale, with the
    same flux, in the same order.

    Positions, levels, the chosen synthesis operator and the convergence flag
    are exact; the measured quantities carry the recorded float tolerance.
    """
    tolerances = tier1_provenance["tolerances"]
    assert len(tier1_atoms) == len(golden_atoms)

    exact = ("iteration", "index", "level", "bbox_0", "bbox_1", "bbox_2",
             "bbox_3", "flag_convergence", "filter")
    approximate = ("eccentricity", "sum_wr", "norm_wr", "flux", "peak",
                   "det_err_flux")

    for actual, expected in zip(tier1_atoms, golden_atoms):
        where = f"atom {expected['index']} of iteration {expected['iteration']}"
        for name in exact:
            assert actual[name] == expected[name], f"{where}: {name}"
        for name in approximate:
            np.testing.assert_allclose(
                actual[name],
                expected[name],
                rtol=tolerances["atom_rtol"],
                atol=tolerances["atom_atol"],
                err_msg=f"{where}: {name}",
            )


def test_restored_image_matches(tier1_run: Tier1Run, tier1_goldens: Path,
                                tier1_provenance):
    """The sum of every atom, pixel for pixel within tolerance."""
    assert_close(
        fits.getdata(tier1_run.restored),
        fits.getdata(tier1_goldens / "restored.fits"),
        tier1_provenance["tolerances"],
        "restored image",
    )


def test_residual_image_matches(tier1_run: Tier1Run, tier1_goldens: Path,
                                tier1_provenance):
    """What the loop left behind, pixel for pixel within tolerance."""
    assert_close(
        fits.getdata(tier1_run.residuals),
        fits.getdata(tier1_goldens / "residuals.fits"),
        tier1_provenance["tolerances"],
        "residual image",
    )
