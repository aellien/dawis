"""Tier 2 — the parallel-consistency check (#57).

Two runs of the published configuration on the whole 900x900 gallery image,
identical in every knob but the worker count: one on a single worker, one on
:data:`tier2.PARALLEL_N_WORKERS`. Their atoms must agree **to the bit**, down to
the bytes of the restored images.

No recording is read, so nothing here can be skipped by a missing golden.
The rule being applied, and the argument for why a tolerance would defeat the
purpose, live with the comparison in
:func:`goldencompare.assert_atoms_are_identical`; so does the reason atom
order is normalised away rather than compared.

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

Both halves run at one maths-library thread, pinned session-wide at the top of
``conftest``, which is where the argument for it lives. Without that pin this
file would be comparing thread counts as well as worker counts, and reading
the last-digit difference that follows as a defect of the latter.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from goldencompare import assert_atoms_are_identical
from goldenrun import golden_config
from tier2 import CG_MAX_ITER, PARALLEL_N_WORKERS

# A second half-hour run. Marked at module scope so no test in this file can be
# added without the marker, and so neither session-scoped run fixture is ever
# built under the default `-m "not slow"` selection.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="session", autouse=True)
def _require_a_parallel_backend(parallel_backend_name: str):
    """Skip this file, before either run starts, if nothing can spread work.

    Both halves are pointless without a parallel backend, and the serial half
    is a session fixture shared with the golden file, so without this guard a
    machine lacking the ``[ray]`` extra would run for half an hour and only
    then discover it had nothing to compare against.

    Session-scoped and autouse for the same reason as the goldens guard in
    ``test_golden_gallery.py``: pytest builds higher-scoped fixtures first, so
    a function-scoped guard would be set up after the runs it guards.
    """


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
