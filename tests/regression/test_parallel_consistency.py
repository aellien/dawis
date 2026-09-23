"""Tier 2 — the parallel-consistency check (#57, #61).

DAWIS has two kinds of parallelism, and this file holds one case for each.
Both run the published configuration on the whole 900x900 gallery image
against the same configuration with one knob changed, and both compare the
atoms the two runs produced. What they claim is deliberately different,
because the two knobs promise different things.

**The worker count changes nothing at all.** A worker is a whole process this
run hands a patch of trees to; the arithmetic inside a tree is untouched, so
the atoms of a run on :data:`tier2.PARALLEL_N_WORKERS` workers must agree with
the single-worker run **to the bit**, down to the bytes of the restored
images.

**The thread count changes only the last digits.** A thread is what a maths
library splits a single array operation across inside one process. The partial
sums are then added in a different association order, so an atom's flux moves
in its last digits — floating-point addition is not associative, and no
argument makes that a defect. A run at :data:`tier2.PARALLEL_N_THREADS`
threads must therefore agree with the published one-thread run *within the
golden tolerances* — and exactly on every decision: the atom count, each
atom's level, its bounding box, its chosen filter, its convergence flag.

That asymmetry is the whole of #61. Before it, DAWIS set no thread count at
all: ray forced every worker it started to one thread while the driver took
every core, so raising the worker count silently raised the thread count too
and the last-digit movement looked like a defect of the worker count.

No recording is read by the worker-count case, so nothing there can be skipped
by a missing golden. The thread case reads a golden's ``provenance.toml`` for
its tolerances and nothing else — never its recorded atoms — so it too
compares two runs of the tree under test against each other.

The rules being applied, and the argument for why a tolerance would defeat the
purpose of the first one, live with the comparisons in
:func:`goldencompare.assert_atoms_are_identical` and
:func:`goldencompare.assert_atoms_are_equivalent`; so does the reason atom
order is normalised away rather than compared.

Scope of the bit-identity claim
-------------------------------

**One machine, one BLAS build, one thread count, and the worker count as the
only variable.** That is what has been measured: a four-core 2017 laptop, the
BLAS numpy happened to load there, ray 2.57, both runs pinned to
:data:`goldenrun.GOLDEN_N_THREADS`. Bit-identity across heterogeneous nodes —
different CPUs, a different BLAS build on the worker than on the driver — is
untested and is not claimed here.

Why the full image
------------------

The scatter has two halves and the Tier 1 cutout only reaches one of them.
Restoration patches one tree at a time, so it scatters on any input — the
cutout's 76 atoms go out as 76 patches. Tree building sends nothing anywhere
until it has more than ``size_patch`` roots, which is 100, and the cutout
finds twelve; it logs *Attaching regions in process* and works the whole batch
here whatever ``n_workers`` says. Only the full image crosses that threshold, so
only the full image compares both halves.

That puts this file at a second half-hour run, and therefore behind the
``slow`` marker with the rest of Tier 2.

    pytest -m slow tests/regression/test_parallel_consistency.py

What is *not* a reason to sit at Tier 2 is the solver knob. The harness forces
``cg_max_iter`` by patching the solver in this process, and a worker is not
this process — but the ``Config`` carries the same number to the worker by the
other route, so both halves solve at the same setting whatever the tier. See
:func:`goldenrun.run_golden`.

Threads, which are not workers
------------------------------

Every run in the worker-count case is pinned to
:data:`goldenrun.GOLDEN_N_THREADS`, through ``Config.n_threads`` and DAWIS's
own ``threadpoolctl`` call — the same code path a user runs. Without that pin
this file would be comparing thread counts as well as worker counts, and
reading the last-digit difference that follows as a defect of the latter,
which is exactly what #61 found it doing.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from goldencompare import (
    assert_atoms_are_equivalent,
    assert_atoms_are_identical,
    assert_convergence_matches,
    assert_image_matches,
)
from goldenrun import GOLDEN_N_THREADS, golden_config
from tier2 import CG_MAX_ITER, PARALLEL_N_THREADS, PARALLEL_N_WORKERS

# A second half-hour run. Marked at module scope so no test in this file can be
# added without the marker, and so neither session-scoped run fixture is ever
# built under the default `-m "not slow"` selection.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="session", autouse=True)
def _require_a_parallel_backend(parallel_backend_name: str):
    """Skip this file, before either run starts, if nothing can spread work.

    Both halves are pointless without a parallel backend, and the serial half
    is a session fixture shared with the golden file, so without this guard a
    machine lacking the ``[ray]`` extra would run a whole Tier 2 pass and only
    then discover it had nothing to compare against.

    Session-scoped and autouse for the same reason as the goldens guard in
    ``test_golden_gallery.py``: pytest builds higher-scoped fixtures first, so
    a function-scoped guard would be set up after the runs it guards.
    """


# ---------------------------------------------------------------------------
# The worker count: nothing may move at all
# ---------------------------------------------------------------------------


def test_the_two_runs_differ_in_the_worker_count_and_nothing_else():
    """The premise of the comparison, asserted rather than assumed.

    Both runs are built from one parameter set with one argument changed, so
    they cannot drift apart by editing — but ``n_workers`` is not the only knob
    that touches scheduling, and a ``backend`` or ``size_patch`` override
    creeping into either half later would turn a bit-identity claim into a
    comparison of two configurations that still passes.
    """
    import dawis

    serial = asdict(golden_config(dawis, CG_MAX_ITER))
    parallel = asdict(golden_config(dawis, CG_MAX_ITER,
                                    n_workers=PARALLEL_N_WORKERS))

    assert {name for name in serial
            if serial[name] != parallel[name]} == {"n_workers"}
    assert serial["n_workers"] == 1
    assert parallel["n_workers"] == PARALLEL_N_WORKERS


def test_atom_count_is_exact(tier2_serial_atoms, tier2_parallel_atoms):
    """Both runs restore the same number of atoms.

    Separated from the comparison below because the two failures mean
    different things: a differing count says a tree was lost or restored
    twice — the defects #56 found in the old parallel branch's patching —
    while a differing atom says one tree was restored differently.
    """
    assert len(tier2_parallel_atoms) == len(tier2_serial_atoms)


def test_atoms_are_bit_identical(tier2_serial_atoms, tier2_parallel_atoms):
    """Every atom of the parallel run matches the serial run exactly.

    Position, scale, bounding box, chosen filter, convergence flag, every
    measured quantity, and a digest of the restored image and detection-error
    image themselves — with atom order normalised away first and no tolerance
    on anything.
    """
    assert_atoms_are_identical(
        tier2_parallel_atoms,
        tier2_serial_atoms,
        what_actual="the %d-worker run" % PARALLEL_N_WORKERS,
        what_reference="the serial run",
    )


# ---------------------------------------------------------------------------
# The thread count: only the last digits may move
# ---------------------------------------------------------------------------


def test_the_two_runs_differ_in_the_thread_count_and_nothing_else():
    """The premise of the second comparison, asserted rather than assumed.

    The worker count stays at the published 1 on both sides. Threads apply per
    process and are not divided among workers, so the two knobs are
    independent — and a worker-count override creeping into either side later
    would make a failure ambiguous between two causes that promise different
    things.
    """
    import dawis

    published = asdict(golden_config(dawis, CG_MAX_ITER))
    threaded = asdict(golden_config(dawis, CG_MAX_ITER,
                                    n_threads=PARALLEL_N_THREADS))

    assert {name for name in published
            if published[name] != threaded[name]} == {"n_threads"}
    assert published["n_threads"] == GOLDEN_N_THREADS
    assert threaded["n_threads"] == PARALLEL_N_THREADS
    assert published["n_workers"] == threaded["n_workers"] == 1


def test_the_thread_count_reaches_the_maths_library(tier2_threaded_run):
    """The run under test actually threaded at what it asked for.

    Without this the tolerance comparison below would pass just as happily on
    a knob that did nothing at all — which is the exact failure #61 exists to
    fix, and it passed unnoticed for as long as it did because no test ever
    asked what the library did.

    The *effective* count is observed, not the environment variable: the
    variable is the mechanism and the effective count is the behaviour. It is
    observed inside a fresh block rather than read off the finished run,
    because the run put the caller's limits back on its way out — which is
    itself the behaviour being relied on.

    Depends on the threaded run so the assertion follows a real run of it
    rather than standing on its own.
    """
    from dawis.parallel import observed_threads, parallel_backend

    with parallel_backend(n_workers=1, n_threads=PARALLEL_N_THREADS):
        effective, vendor = observed_threads()

    if effective is None:
        pytest.skip("no threaded BLAS is loaded, so there is no thread count "
                    "to vary and nothing this case could compare")

    assert effective == PARALLEL_N_THREADS, (
        f"asked for {PARALLEL_N_THREADS} threads and the loaded {vendor} "
        f"reports {effective}, so this case is comparing two runs at the same "
        "thread count and proves nothing"
    )


def test_the_thread_count_restores_the_same_atoms(
        tier2_threaded_atoms, tier2_measured_atoms, tier2_provenance):
    """Changing the thread count moves only the last digits.

    Every decision exactly — the atom count, and each atom's level, bounding
    box, chosen filter and convergence flag — and every measured quantity
    within the golden's own recorded tolerance. Those tolerances are read from
    ``provenance.toml`` and not invented here, so this case can never be more
    forgiving of a thread count than the golden comparison is of a change of
    machine.
    """
    assert_atoms_are_equivalent(
        tier2_threaded_atoms,
        tier2_measured_atoms,
        tier2_provenance["tolerances"],
        what_actual="the %d-thread run" % PARALLEL_N_THREADS,
        what_reference="the %d-thread run" % GOLDEN_N_THREADS,
    )


def test_the_thread_count_leaves_the_convergence_trace_where_it_was(
        tier2_threaded_convergence, tier2_convergence, tier2_provenance):
    """The greedy loop took the same trajectory, iteration for iteration.

    The trace is what the loop steers on, so it is the earliest visible
    symptom of a thread count that moved more than the last digits: the level
    each iteration ran at and the atoms it produced exactly, the epsilons and
    fluxes within the golden's recorded convergence tolerance.
    """
    assert_convergence_matches(
        tier2_threaded_convergence,
        tier2_convergence,
        tier2_provenance["tolerances"],
    )


@pytest.mark.parametrize("image", ["restored", "residuals"])
def test_the_thread_count_leaves_the_output_images_where_they_were(
        image, tier2_threaded_run, tier2_run, tier2_provenance):
    """The sum of every atom, and what the loop left behind.

    The atom catalogue describes an atom by measurements taken off it; these
    are the images themselves, which is the other half of what a run
    *produces*. Compared under the golden's own recorded image tolerance, so a
    thread count can never be allowed to move a pixel further than a change of
    machine is.
    """
    from astropy.io import fits

    assert_image_matches(
        fits.getdata(getattr(tier2_threaded_run, image)),
        fits.getdata(getattr(tier2_run, image)),
        tier2_provenance["tolerances"],
        "the %d-thread run's %s image" % (PARALLEL_N_THREADS, image),
    )
