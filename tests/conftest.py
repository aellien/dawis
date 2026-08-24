"""Shared fixtures for the DAWIS test suite.

The suite is organised in the three tiers decided on #3:

* ``tests/unit`` — Tier 0. Mathematical properties and validation, seconds,
  no goldens.
* ``tests/regression`` — Tier 1 (256x256 cutout, every commit) and Tier 2
  (full 900x900 gallery run, ``slow``).
* ``tests/goldens`` — the recorded outputs plus ``provenance.toml``.

Fixtures only. What a golden run *is* lives outside pytest, because the
hand-run recording scripts share it: the machinery common to both tiers in
``tests/goldenrun.py``, and what each tier changes — the input, the pinned
solver knob — in ``tests/tier1.py`` and ``tests/tier2.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from goldenrun import (
    ATOM_COLUMNS,
    CONVERGENCE_COLUMNS,
    GALLERY_IMAGE,
    TEST_SEED,
    GoldenRun,
    atom_catalogue,
    convergence_table,
    read_table,
)
from tier1 import TIER1_PARAMS, cutout_of, run_tier1
from tier2 import run_tier2


@pytest.fixture(scope="session")
def gallery_image_path() -> Path:
    """Path to the gallery FITS image inside the installed package."""
    import dawis

    path = Path(dawis.__file__).parent / "gallery" / GALLERY_IMAGE
    if not path.is_file():
        pytest.skip(f"gallery image not found: {path}")
    return path


@pytest.fixture(scope="session")
def gallery_image(gallery_image_path: Path) -> np.ndarray:
    """The full 900x900 gallery image, read once per session.

    The array is returned read-only so one test cannot corrupt another's
    input. Copy it if you need to modify it.
    """
    from astropy.io import fits

    data = fits.getdata(gallery_image_path)
    data.setflags(write=False)
    return data


@pytest.fixture(scope="session")
def cutout(gallery_image: np.ndarray) -> np.ndarray:
    """The agreed 256x256 Tier 1 cutout of the gallery image.

    Read-only, like :func:`gallery_image`. Copy it if you need to modify it —
    the pipeline subtracts the background in place.
    """
    patch = np.array(cutout_of(gallery_image))
    patch.setflags(write=False)
    return patch


@pytest.fixture
def seeded_config() -> dict:
    """Pipeline knobs for a deterministic test run.

    Returned as a plain mapping of keyword arguments because ``Config`` does
    not exist yet; it lands in #53 and this fixture becomes a ``Config``
    instance then.

    The ``max_iter`` here is the *outer* greedy loop's. The conjugate-gradient
    solver has an unrelated knob of the same name, pinned separately — see
    ``tier1.CG_MAX_ITER``.
    """
    return dict(TIER1_PARAMS, seed=TEST_SEED)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """An empty directory for one run's outputs, removed after the test."""
    path = tmp_path / "run"
    path.mkdir()
    return path


# ---------------------------------------------------------------------------
# Tier 1 — the recorded cutout goldens (#36)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tier1_goldens() -> Path:
    """Directory holding the recorded Tier 1 goldens."""
    path = Path(__file__).parent / "goldens" / "tier1_cutout"
    if not (path / "provenance.toml").is_file():
        pytest.skip(
            f"Tier 1 goldens not recorded: {path}. "
            "Record them with tests/goldens/record_cutout_golden.py."
        )
    return path


@pytest.fixture(scope="session")
def tier1_provenance(tier1_goldens: Path) -> dict:
    """The goldens' ``provenance.toml``, which owns the tolerances."""
    import tomllib

    with open(tier1_goldens / "provenance.toml", "rb") as handle:
        return tomllib.load(handle)


@pytest.fixture(scope="session")
def tier1_run(cutout: np.ndarray, tier1_goldens: Path, tmp_path_factory) -> GoldenRun:
    """Run Tier 1 against the tree under test, once per session.

    Depends on the goldens so a missing recording skips before the run rather
    than after it.
    """
    import dawis

    return run_tier1(dawis, cutout, tmp_path_factory.mktemp("tier1"))


@pytest.fixture(scope="session")
def tier1_atoms(tier1_run: GoldenRun) -> list[dict]:
    """Every atom the tree under test produced, in the order it produced them."""
    import dawis

    return atom_catalogue(dawis, tier1_run)


@pytest.fixture(scope="session")
def tier1_convergence(tier1_run: GoldenRun) -> list[dict]:
    """The convergence trace of the run under test, one row per iteration."""
    return convergence_table(tier1_run)


@pytest.fixture(scope="session")
def golden_atoms(tier1_goldens: Path) -> list[dict]:
    """The recorded atom catalogue."""
    return read_table(tier1_goldens / "atoms.csv", ATOM_COLUMNS)


@pytest.fixture(scope="session")
def golden_convergence(tier1_goldens: Path) -> list[dict]:
    """The recorded convergence trace."""
    return read_table(tier1_goldens / "convergence.csv", CONVERGENCE_COLUMNS)


# ---------------------------------------------------------------------------
# Tier 2 — the recorded full-image goldens (#37)
# ---------------------------------------------------------------------------
#
# Every Tier 2 test carries the ``slow`` marker, so under the default
# ``-m "not slow"`` these fixtures are never built and the twenty-minute run
# never starts.


@pytest.fixture(scope="session")
def tier2_goldens() -> Path:
    """Directory holding the recorded Tier 2 goldens."""
    path = Path(__file__).parent / "goldens" / "tier2_gallery"
    if not (path / "provenance.toml").is_file():
        pytest.skip(
            f"Tier 2 goldens not recorded: {path}. "
            "Record them with tests/goldens/record_gallery_golden.py."
        )
    return path


@pytest.fixture(scope="session")
def tier2_provenance(tier2_goldens: Path) -> dict:
    """The goldens' ``provenance.toml``, which owns the tolerances."""
    import tomllib

    with open(tier2_goldens / "provenance.toml", "rb") as handle:
        return tomllib.load(handle)


@pytest.fixture(scope="session")
def tier2_run(gallery_image: np.ndarray, tier2_goldens: Path,
              tmp_path_factory) -> GoldenRun:
    """Run Tier 2 against the tree under test, once per session.

    About half an hour — 1641 s measured. Depends on the goldens so a missing
    recording skips before the run rather than after it.
    """
    import dawis

    return run_tier2(dawis, gallery_image, tmp_path_factory.mktemp("tier2"))


@pytest.fixture(scope="session")
def tier2_atoms(tier2_run: GoldenRun) -> list[dict]:
    """Every atom the tree under test produced, in the order it produced them."""
    import dawis

    return atom_catalogue(dawis, tier2_run)


@pytest.fixture(scope="session")
def tier2_convergence(tier2_run: GoldenRun) -> list[dict]:
    """The convergence trace of the run under test, one row per iteration."""
    return convergence_table(tier2_run)


@pytest.fixture(scope="session")
def golden_gallery_atoms(tier2_goldens: Path) -> list[dict]:
    """The recorded Tier 2 atom catalogue."""
    return read_table(tier2_goldens / "atoms.csv", ATOM_COLUMNS)


@pytest.fixture(scope="session")
def golden_gallery_convergence(tier2_goldens: Path) -> list[dict]:
    """The recorded Tier 2 convergence trace."""
    return read_table(tier2_goldens / "convergence.csv", CONVERGENCE_COLUMNS)
