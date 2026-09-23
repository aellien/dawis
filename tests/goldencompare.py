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

#: Per-atom columns compared exactly. Positions, levels, the arm the tree was
#: solved with and the convergence flag are decisions, not measurements: a
#: tolerance on them would hide the very drift the tier is looking for.
#:
#: ``algorithm`` and ``synthesis_operator`` arrive with #105. This list and
#: the module docstring above both promised the chosen synthesis operator was
#: compared exactly from the day they were written, and until #105 recorded
#: the column there was nothing to compare — the promise was carried by the
#: routing rule, which derived the operator from ``filter`` and ``level``, and
#: it outlived that rule by one ticket when #103 switched the routing off.
ATOM_EXACT = (
    "iteration", "index", "level", "bbox_0", "bbox_1", "bbox_2", "bbox_3",
    "flag_convergence", "filter", "algorithm", "synthesis_operator",
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


# ---------------------------------------------------------------------------
# Comparing two runs of the tree under test against each other (#57)
# ---------------------------------------------------------------------------
#
# Not a golden comparison, and so not governed by #3's tolerance rule above.
# What is compared here is two runs of the *same* tree that differ only in how
# the work was scheduled, and #56 made that a difference the science cannot
# see: every atom is produced by the same arithmetic in the same order on
# either path, not by equivalent arithmetic. There is nothing left for a
# tolerance to absorb except a defect, and #21 is the defect in question — a
# rare atom returning a flux orders of magnitude wrong in a run whose every
# other atom looks fine. A tolerance wide enough to be comfortable would hide
# it.
#
# Ordering is normalised away first. #57 expects the two orders to differ,
# because scheduling decides which patch of trees is submitted first; measured
# since #56 they agree, because the atoms are laid back out in tree order once
# the patches come home. Normalising regardless is what keeps this a statement
# about content, so that a later change to the patching rule moves nothing
# here.

#: Per-atom columns that say *which* atom a row is rather than measure it, and
#: so the columns ordering is normalised on. Distinct from ``ATOM_EXACT``
#: above, which answers a different question — which columns a *golden*
#: comparison may not put a tolerance on.
ATOM_IDENTITY = (
    "iteration", "level", "bbox_0", "bbox_1", "bbox_2", "bbox_3", "filter",
)

#: The one column never compared: ``index`` *is* the ordering being normalised
#: away, so it cannot also stand as evidence. Everything else a row carries is
#: compared, whatever the caller put there — see
#: :func:`assert_atoms_are_identical`.
ATOM_NOT_COMPARED = frozenset({"index"})


def _atom_key(row):
    """The identity of one atom, as the tuple ordering is normalised on."""
    return tuple(row[name] for name in ATOM_IDENTITY)


def _equal_or_both_nan(left, right) -> bool:
    """Exact equality, reading two NaNs as equal.

    ``float('nan') != float('nan')``, so a plain ``==`` would report a
    difference between two runs that agree perfectly on a diverged atom.
    """
    if left == right:
        return True
    return (isinstance(left, float) and isinstance(right, float)
            and np.isnan(left) and np.isnan(right))


def _assert_keys_are_unique(rows, what):
    """The identity columns pair rows one-to-one, or the comparison is void.

    Two atoms of one iteration sharing a scale, a bounding box and a filter
    would make the sort ambiguous, and the comparison would then pass or fail
    on which of the two the sort happened to put first. That has never been
    observed; it is asserted rather than assumed because a silent version of
    it looks exactly like a passing test.
    """
    keys = [_atom_key(row) for row in rows]
    if len(set(keys)) == len(keys):
        return
    duplicated = sorted({key for key in keys if keys.count(key) > 1})
    raise AssertionError(
        "%s: %d atoms share an identity with another, so atom order cannot be "
        "normalised. First: %r"
        % (what, len(duplicated), duplicated[0])
    )


def _paired_atoms(actual_rows, reference_rows, what_actual, what_reference):
    """Pair two runs' atoms by identity, or say why they cannot be paired.

    The preamble both two-run comparisons share, and it is shared because
    getting it wrong is the one way either of them could pass without meaning
    anything: a differing count, an ambiguous sort, or an atom one run has and
    the other does not.

    Yields
    ------
    tuple of (dict, dict, str)
        Each pair of rows, and how to name the atom in a failure.
    """
    assert len(actual_rows) == len(reference_rows), (
        "%s restored %d atoms, %s restored %d"
        % (what_actual, len(actual_rows), what_reference, len(reference_rows))
    )

    _assert_keys_are_unique(actual_rows, what_actual)
    _assert_keys_are_unique(reference_rows, what_reference)

    actual_sorted = sorted(actual_rows, key=_atom_key)
    reference_sorted = sorted(reference_rows, key=_atom_key)

    for actual, reference in zip(actual_sorted, reference_sorted, strict=True):
        assert _atom_key(actual) == _atom_key(reference), (
            "%s has an atom %r that %s does not"
            % (what_actual, _atom_key(actual), what_reference)
        )
        yield actual, reference, "atom %r" % (_atom_key(reference),)


def assert_atoms_are_identical(actual_rows, reference_rows, what_actual,
                               what_reference):
    """Two runs restored the same atoms, bit for bit, in any order.

    Every column the rows carry is compared except ``index``, rather than a
    fixed list of them. That is what lets the caller decide how much of an
    atom is under test: :func:`goldenrun.atom_catalogue` describes an atom by
    its measurements, and :func:`goldenrun.atom_catalogue_with_digests` adds a
    digest of the image bytes themselves, which is the difference between
    "the same flux" and "the same atom".
    """
    for actual, reference, where in _paired_atoms(
            actual_rows, reference_rows, what_actual, what_reference):
        assert actual.keys() == reference.keys(), (
            "%s and %s describe an atom by different columns: %r against %r"
            % (what_actual, what_reference,
               sorted(actual), sorted(reference))
        )
        for name in sorted(actual.keys() - ATOM_NOT_COMPARED):
            assert _equal_or_both_nan(actual[name], reference[name]), (
                "%s: %s is %r in %s and %r in %s"
                % (where, name, actual[name], what_actual,
                   reference[name], what_reference)
            )


# ---------------------------------------------------------------------------
# Comparing two runs that differ in the thread count (#61)
# ---------------------------------------------------------------------------
#
# The mirror image of :func:`assert_atoms_are_identical` above, and the reason
# the two exist side by side is that DAWIS has two kinds of parallelism with
# two different promises.
#
# The worker count is scheduling: it hands whole trees to other processes, the
# arithmetic inside a tree is untouched, and nothing may move at all. The
# thread count is not scheduling: it splits one array operation across threads
# inside a process, the partial sums are added in a different association
# order, and an atom's flux moves in its last digits. That is not a defect and
# no tolerance can be argued away — it is floating-point addition not being
# associative.
#
# What must *not* move either way is a decision. An atom count, a level, a
# bounding box, a chosen filter and a convergence flag are choices the pipeline
# made, not quantities it measured; a change in one of them is real science
# drift and has to turn this red. So the split is the same one #3 lays down for
# a golden — ``ATOM_EXACT`` against ``ATOM_APPROXIMATE`` — applied to two runs
# of the tree under test rather than to a recording.
#
# The tolerance is the golden's own, read out of ``provenance.toml`` by the
# caller. Inventing a second, wider number here would be claiming the thread
# count is allowed to move an atom further than a whole change of machine is,
# which is the opposite of what #61 measured.


def assert_atoms_are_equivalent(actual_rows, reference_rows, tolerances,
                                what_actual, what_reference):
    """Two runs restored the same atoms, to within a tolerance, in any order.

    Every *decision* is compared exactly and every *measurement* under
    ``tolerances``. Atom order is normalised away first, by the same identity
    columns and for the same reason as
    :func:`assert_atoms_are_identical` — see the note above that function.

    Parameters
    ----------
    actual_rows, reference_rows : list of dict
        Atom catalogues, as :func:`goldenrun.atom_catalogue` builds them.
        *Not* the digested form: a digest is an exact comparison in disguise,
        and the whole claim here is that the last digits may move.
    tolerances : dict
        The golden's recorded tolerances. ``atom_rtol`` and ``atom_atol`` are
        read.
    what_actual, what_reference : str
        How each side is named in a failure.
    """
    for actual, reference, where in _paired_atoms(
            actual_rows, reference_rows, what_actual, what_reference):
        for name in ATOM_EXACT:
            if name in ATOM_NOT_COMPARED:
                continue
            assert actual[name] == reference[name], (
                "%s: %s is %r in %s and %r in %s — a decision, which the "
                "thread count may not move"
                % (where, name, actual[name], what_actual,
                   reference[name], what_reference)
            )
        for name in ATOM_APPROXIMATE:
            np.testing.assert_allclose(
                actual[name],
                reference[name],
                rtol=tolerances["atom_rtol"],
                atol=tolerances["atom_atol"],
                err_msg="%s: %s in %s against %s"
                % (where, name, what_actual, what_reference),
            )
