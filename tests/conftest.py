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

One thing here is not a fixture: the OpenMP thread cap below, which has to be
set before numpy is imported and so cannot wait for one.
"""

from __future__ import annotations

import os

# Cap the maths libraries at one thread, for the whole session, before numpy
# loads — OpenBLAS and OpenMP read this at library load and never again, which
# is why it sits above the import rather than in a fixture.
#
# This is not about speed; it costs nothing measurable, because DAWIS spends
# its time in convolutions rather than in the linear algebra that threads well.
# It is about the run depending on the input and nothing else. numpy splits a
# single array operation over threads on its own, invisibly, whatever
# ``n_workers`` says — ``n_workers`` sends whole trees to other *processes* and is a
# different knob entirely — and a sum split over four threads is added in a
# different order from one added on a single thread. On the Tier 1 cutout that
# moves an atom's flux in its last digit: 30.25294122005867 unpinned against
# 30.252941220058663 at one thread, from the same serial run.
#
# The parallel-consistency check of #57 cannot live with that, because ray
# caps its own workers at one thread while a pytest process is capped at
# nothing, so an unpinned session would compare four threads against one and
# call the last-digit difference a defect of the worker count. Pinning both
# sides is what leaves the worker count as the only thing that differs.
#
# All four names, because they are not synonyms and the wrong one alone leaves
# a hole: OpenBLAS reads OPENBLAS_NUM_THREADS in preference to OMP_NUM_THREADS
# and MKL reads MKL_NUM_THREADS, so either of those set in a contributor's
# environment would silently defeat a pin that named only OMP_NUM_THREADS —
# and the symptom would be exactly the spurious last-digit failure the pin
# exists to prevent.
#
# ``setdefault``, so a caller that has chosen a cap of its own keeps it.
for _thread_knob in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_knob, "1")

from pathlib import Path  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from goldenrun import (  # noqa: E402
    ATOM_COLUMNS,
    CONVERGENCE_COLUMNS,
    GALLERY_IMAGE,
    TEST_SEED,
    GoldenRun,
    atom_catalogue,
    atom_catalogue_with_digests,
    convergence_table,
    read_table,
)
from tier1 import TIER1_PARAMS, cutout_of, run_tier1  # noqa: E402
from tier2 import PARALLEL_N_WORKERS, run_tier2  # noqa: E402

if TYPE_CHECKING:
    # Imported for the annotation only. Every fixture here reaches for `dawis`
    # inside its own body rather than at module scope, so that collecting the
    # suite does not require the package to be importable.
    from dawis.config import Config


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
def seeded_config() -> "Config":
    """A ``Config`` for a deterministic test run.

    The shared golden parameters with the test seed pinned. Keys ``Config``
    does not have are dropped — ``gif`` is one, a parameter of the
    ``legacy-seeded`` entry point that #46 removed — because the golden set is
    filtered against whichever entry point is being driven, and this fixture
    describes the 1.0 tree only.

    The ``max_iter`` here is the *outer* greedy loop's. The conjugate-gradient
    solver has an unrelated knob of the same name, now ``Config.cg_max_iter``,
    pinned separately for Tier 1 — see ``tier1.CG_MAX_ITER``.
    """
    from dawis.config import Config

    knobs = dict(TIER1_PARAMS, seed=TEST_SEED)
    return Config(**{
        name: value
        for name, value in knobs.items()
        if name in Config.__dataclass_fields__
    })


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
def tier2_run(gallery_image: np.ndarray, tmp_path_factory) -> GoldenRun:
    """Run Tier 2 against the tree under test, once per session.

    About half an hour — 1641 s measured.

    It does *not* depend on the goldens, though the comparison that reads it
    usually does. Wiring the skip in here would have made every consumer of
    this run skip on a missing recording, including #57's consistency check,
    which compares this run against a second one and reads no recording at
    all — a silent skip in the file whose whole job is to run. The guard lives
    where the recording is actually wanted instead, as the session-scoped
    autouse fixture at the top of ``test_golden_gallery.py``, so a missing
    recording still skips before the half hour rather than after it.
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


# ---------------------------------------------------------------------------
# Tier 2 — the parallel-consistency run (#57)
# ---------------------------------------------------------------------------
#
# The same Tier 2 configuration a second time, on PARALLEL_N_WORKERS workers
# instead of one. Its atoms are compared against the serial run's rather than
# against a recording, so this half of the tier reads no golden and cannot be
# skipped by a missing one — see `tier2_run` for the wiring that keeps that
# true.
#
# It does add a second half-hour run to the tier. The serial half is the
# existing session fixture rather than a run of its own, which is what holds
# the cost at two runs rather than three.


@pytest.fixture(scope="session")
def parallel_backend_name() -> str:
    """The backend a four-worker run picks, proven to start on this machine.

    Asked of ``dawis.parallel`` rather than named here, so the probe follows
    ``DEFAULT_PARALLEL_BACKEND`` if that ever changes. It starts the backend
    and stops it again rather than testing whether the extra imports: the
    adapters import their extra lazily, so importability is all that could be
    established without starting one, and importability is not the same claim
    as *runs here*.

    The name is returned, and asserted not to be ``'serial'``, because a
    worker count that quietly fell back to the serial backend would compare a
    serial run against a serial run and report bit-identity — truthfully, and
    uselessly.
    """
    from dawis import parallel

    backend = parallel.make_backend(n_workers=PARALLEL_N_WORKERS)
    try:
        backend.start()
    except ImportError as error:
        pytest.skip(
            f"the {backend.name!r} parallel backend is unavailable: {error}"
        )
    finally:
        # Runs on the skip and on any other failure too, so a probe that got
        # part way does not leave a cluster up for the rest of the session.
        backend.stop()

    assert backend.name != "serial", (
        f"{PARALLEL_N_WORKERS} workers selected the serial backend, so there is "
        "no parallel path here to compare against"
    )
    return backend.name


@pytest.fixture(scope="session")
def tier2_parallel_run(gallery_image: np.ndarray, parallel_backend_name: str,
                       tmp_path_factory) -> GoldenRun:
    """Run Tier 2 on :data:`tier2.PARALLEL_N_WORKERS` workers, once per session.

    Every knob but the worker count is the one ``tier2_run`` uses, so the two
    runs differ in scheduling and in nothing else. About half an hour, like the
    run it is compared against. Depends on the backend probe so that a machine
    without the extra skips before the run rather than half an hour into it.
    """
    import dawis

    return run_tier2(dawis, gallery_image,
                     tmp_path_factory.mktemp("tier2_parallel"),
                     n_workers=PARALLEL_N_WORKERS)


@pytest.fixture(scope="session")
def tier2_serial_atoms(tier2_run: GoldenRun) -> list[dict]:
    """The serial run's atoms, described down to their image bytes.

    ``tier2_atoms`` describes an atom by measurements taken off its image,
    which is what a golden comparison wants. #57 claims two runs produce the
    same *atom*, so it needs the images themselves — see
    :func:`goldenrun.atom_catalogue_with_digests`.
    """
    import dawis

    return atom_catalogue_with_digests(dawis, tier2_run)


@pytest.fixture(scope="session")
def tier2_parallel_atoms(tier2_parallel_run: GoldenRun) -> list[dict]:
    """The parallel run's atoms, described the same way as the serial run's."""
    import dawis

    return atom_catalogue_with_digests(dawis, tier2_parallel_run)
