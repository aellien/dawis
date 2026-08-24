"""Tier 2 — the full-image golden regression (#37).

One seeded run of the published configuration on the whole 900x900 gallery
image at the shipping conjugate-gradient ``max_iter=200``, compared against
the output recorded from ``legacy-seeded``.

This is the science check. Tier 1 (#36) runs the same comparison on a 256x256
patch at a cheap solver setting and answers "did the carve break the
structure"; this tier answers "does the pipeline still do the published
science", on the whole image, at the value 1.0 actually ships.

Every test here carries the ``slow`` marker and is excluded from the default
run by ``-m "not slow"`` in ``pyproject.toml``. It runs at named carve
checkpoints — after each subpackage is fully carved out — and nightly in CI
(#19). Budget a full half hour — 1641 s measured on a 2017 laptop; the
figure is in the goldens' ``provenance.toml`` under ``[budget]``.

    pytest -m slow tests/regression/test_golden_gallery.py

Tolerances come from ``tests/goldens/tier2_gallery/provenance.toml``, and the
comparison rule is shared with Tier 1 in ``tests/goldencompare.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from astropy.io import fits

from goldencompare import (
    assert_atoms_match,
    assert_convergence_matches,
    assert_image_matches,
)
from goldenrun import GoldenRun, sha256_of

# Twenty minutes of pipeline. Marked at module scope so no test in this file
# can be added without the marker, and so the session-scoped Tier 2 run
# fixture is never built under the default `-m "not slow"` selection.
pytestmark = pytest.mark.slow


def test_gallery_image_matches_the_recorded_input(gallery_image_path: Path,
                                                  tier2_provenance):
    """The run under test is fed exactly the image the golden was recorded on.

    Tier 2 takes the whole packaged image rather than a slice of it, so there
    is no cutout convention to misread — but the packaged file can still be
    replaced. The digest turns that into a loud failure here instead of a
    silently different golden everywhere else.
    """
    recorded = tier2_provenance["input"]
    assert sha256_of(gallery_image_path) == recorded["image_sha256"]


def test_iteration_count_is_exact(tier2_convergence, golden_gallery_convergence):
    """The greedy loop takes the same number of iterations. Exact, per #3."""
    assert len(tier2_convergence) == len(golden_gallery_convergence)


def test_atom_count_is_exact(tier2_atoms, golden_gallery_atoms):
    """The run produces the same number of atoms. Exact, per #3."""
    assert len(tier2_atoms) == len(golden_gallery_atoms)


def test_convergence_trace_matches(tier2_convergence, golden_gallery_convergence,
                                   tier2_provenance):
    """Every iteration ran at the same level and produced the same atoms."""
    assert_convergence_matches(
        tier2_convergence,
        golden_gallery_convergence,
        tier2_provenance["tolerances"],
    )


def test_atom_catalogue_matches(tier2_atoms, golden_gallery_atoms,
                                tier2_provenance):
    """Every atom is restored at the same place, scale and flux, in order.

    At the shipping ``max_iter`` the solver's ``sigma_flux`` convergence test
    is live, so ``flag_convergence`` and the measured fluxes here carry real
    information that Tier 1's cheap setting cannot report.
    """
    assert_atoms_match(
        tier2_atoms, golden_gallery_atoms, tier2_provenance["tolerances"]
    )


def test_restored_image_matches(tier2_run: GoldenRun, tier2_goldens: Path,
                                tier2_provenance):
    """The sum of every atom, pixel for pixel within tolerance."""
    assert_image_matches(
        fits.getdata(tier2_run.restored),
        fits.getdata(tier2_goldens / "restored.fits"),
        tier2_provenance["tolerances"],
        "Tier 2 restored image",
    )


def test_residual_image_matches(tier2_run: GoldenRun, tier2_goldens: Path,
                                tier2_provenance):
    """What the loop left behind, pixel for pixel within tolerance."""
    assert_image_matches(
        fits.getdata(tier2_run.residuals),
        fits.getdata(tier2_goldens / "residuals.fits"),
        tier2_provenance["tolerances"],
        "Tier 2 residual image",
    )
