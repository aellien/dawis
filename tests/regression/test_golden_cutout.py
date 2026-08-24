"""Tier 1 — the cutout golden regression (#36).

One seeded run of the published configuration on a 256x256 patch of the
gallery image, compared against the output recorded from ``legacy-seeded``.
This is the gate that runs before every commit of the 1.0 carve: it asserts
what the pipeline *produces* from a given input, never how it produces it, so
functions can move between modules freely and only a behaviour change turns it
red.

Tolerances come from ``tests/goldens/tier1_cutout/provenance.toml``, so the
recording and the comparison cannot drift apart. The comparison rule itself is
shared with Tier 2 and lives in ``tests/goldencompare.py``.

Budget: about 40 s on a 2017 laptop, held there by ``tier1.CG_MAX_ITER``. The
same comparison at the shipping solver setting is Tier 2 (#37), which takes
half an hour and is marked ``slow``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from goldencompare import (
    assert_atoms_match,
    assert_convergence_matches,
    assert_image_matches,
)
from goldenrun import GoldenRun


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
    """Every iteration ran at the same level and produced the same atoms."""
    assert_convergence_matches(
        tier1_convergence, golden_convergence, tier1_provenance["tolerances"]
    )


def test_atom_catalogue_matches(tier1_atoms, golden_atoms, tier1_provenance):
    """Every atom is restored at the same place, scale and flux, in order."""
    assert_atoms_match(tier1_atoms, golden_atoms, tier1_provenance["tolerances"])


def test_restored_image_matches(tier1_run: GoldenRun, tier1_goldens: Path,
                                tier1_provenance):
    """The sum of every atom, pixel for pixel within tolerance."""
    assert_image_matches(
        fits.getdata(tier1_run.restored),
        fits.getdata(tier1_goldens / "restored.fits"),
        tier1_provenance["tolerances"],
        "Tier 1 restored image",
    )


def test_residual_image_matches(tier1_run: GoldenRun, tier1_goldens: Path,
                                tier1_provenance):
    """What the loop left behind, pixel for pixel within tolerance."""
    assert_image_matches(
        fits.getdata(tier1_run.residuals),
        fits.getdata(tier1_goldens / "residuals.fits"),
        tier1_provenance["tolerances"],
        "Tier 1 residual image",
    )
