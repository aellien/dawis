"""Tier 0 checks that one arm restores every tree (#103).

``restore_object`` used to route. A tree that was small or elongated —
extent below ``extent_sep``, or eccentricity at or above ``ecc_sep`` — and sat
below ``lvl_sep_lin`` was synthesised with the Haar filter; everything else
took the B3-spline, from the plain sum of planes at or below ``lvl_sep_op``
and from the adjoint operator above it. #82 decided DAWIS 1.0 ships one arm
instead, and #103 is where the dispatch goes.

The four thresholds are **kept** — they stay on :class:`~dawis.config.Config`
and route nothing. Whether a routing rule should ever come back is a question
#96 rules outside its destination, and the evidence that would answer it does
not exist yet: interscale-maximum level is perfectly confounded with source
class on all 222 of #74's fixtures, so "best on halos" and "best at level 5"
are the same sentence. Deleting the knobs would make that question more
expensive to ask and no easier to answer.

What is under test here is therefore a *negative*: that four numbers a run
still accepts reach no decision. The checks drive
``restore_objects_default`` over fake trees spread across the space the old
rule split on, and assert that every one of them reaches the solver with the
same filter and the same operator — at the published thresholds, at
thresholds inverted so the old rule would have sent almost every tree down
the other branch, and at each threshold inverted on its own, so a failure
names the knob that came back to life.

The trees are fakes, as in ``test_restoration_unification.py`` and for the
same reason: what is under test is which arguments reach the solver, which a
fake records exactly and a real tree would spend a minute earning. What the
fakes cannot show is that a *run* is unmoved by the thresholds; that is the
seam comparison, which needs ``restoration_tests/`` and lives in
``.research/103-routing-off/``.
"""

from __future__ import annotations

import numpy as np
import pytest

from dawis.exceptions import DawisValueError
from dawis.restoration.synthesis import FILTERS, restore_objects_default

#: The published thresholds — :class:`~dawis.config.Config`'s own defaults.
ROUTED = dict(extent_sep = 0.1, ecc_sep = 0.95, lvl_sep_lin = 2,
              lvl_sep_op = 3)

#: Thresholds chosen to disagree with :data:`ROUTED` about every tree there
#: is. ``extent_sep`` above every extent and ``ecc_sep`` below every
#: eccentricity make the Haar test true for all of them; ``lvl_sep_lin`` above
#: every level lets it through; ``lvl_sep_op`` below every level would have
#: sent whatever was left to the adjoint. Under the old rule these two
#: dictionaries share no tree. Under #103 they share every one.
UNROUTED = dict(extent_sep = 1.0, ecc_sep = 0.0, lvl_sep_lin = 99,
                lvl_sep_op = 0)


class _FakeMaximum:
    """The descriptors of an interscale maximum restoration reads."""

    def __init__(self, level: int, eccentricity: float):
        self.level = level
        self.area = 100
        self.eccentricity = eccentricity


class _FakeTree:
    """Records the arm it was solved with instead of solving."""

    def __init__(self, level: int, extent: float, eccentricity: float):
        self.interscale_maximum = _FakeMaximum(level, eccentricity)
        self.extent = extent
        self.eccentricity = eccentricity
        self.x_size = self.y_size = 8
        self.bbox = (0, 0, 8, 8)
        self.filters: list[tuple[float, ...]] = []
        self.operators: list[str] = []

    def CG_minimization(self, wavelet_cube, label_cube, filter,
                        synthesis_operator = 'ADJOINT', **kwargs):
        self.filters.append(tuple(filter))
        self.operators.append(synthesis_operator)
        return np.ones((8, 8)), 1.0, 2.0

    def det_err_tube(self, wavelet_cube, label_cube):
        return np.zeros((8, 8, 2))


def _trees() -> list[_FakeTree]:
    """Trees spread across the space the old rule split on.

    Below and above ``lvl_sep_lin``, below and above ``lvl_sep_op``, round and
    elongated, small and large in extent. Under routing no two of these five
    took the same branch as each other throughout.
    """

    return [
        _FakeTree(level = 1, extent = 0.05, eccentricity = 0.99),
        _FakeTree(level = 1, extent = 0.80, eccentricity = 0.10),
        _FakeTree(level = 3, extent = 0.05, eccentricity = 0.99),
        _FakeTree(level = 5, extent = 0.80, eccentricity = 0.10),
        _FakeTree(level = 7, extent = 0.05, eccentricity = 0.99),
    ]


def _restore(trees, **overrides):
    """Restore `trees` at the published thresholds, or at `overrides`."""

    return restore_objects_default(
        trees,
        oimage = None, cg_gamma = 0.01, niter = 15,
        wavelet_cube = None, label_cube = None,
        lvl_sep_big = 6, rm_gamma_for_big = False, gamma = 1.0,
        size_patch_small = 1, size_patch_big = 1, size_big_objects = 512,
        n_workers = 1, cg_max_iter = 3,
        **dict(ROUTED, **overrides),
    )


@pytest.mark.parametrize('thresholds', [ROUTED, UNROUTED],
                         ids = ['published', 'inverted'])
def test_every_tree_reaches_the_solver_with_the_same_arm(thresholds):
    """No threshold selects a filter or an operator any more."""

    trees = _trees()
    _restore(trees, filter_kw = 'BSPL', synthesis_operator = 'TRANSPOSE',
             **thresholds)

    assert {tuple(tree.filters) for tree in trees} == {
        (tuple(FILTERS['BSPL']),)}
    assert {tuple(tree.operators) for tree in trees} == {('TRANSPOSE',)}


def test_the_arm_the_caller_names_is_the_arm_the_atom_records():
    """An atom's provenance is the arm asked for, not one re-derived.

    Haar on every tree is a combination the routed path could not produce:
    it reserved Haar for small or elongated trees below ``lvl_sep_lin``, and
    always paired it with ``SUM``.
    """

    trees = _trees()
    atoms = _restore(trees, filter_kw = 'HAAR', synthesis_operator = 'SUM')

    assert {atom.filter for atom in atoms} == {'HAAR'}
    assert {atom.synthesis_operator for atom in atoms} == {'SUM'}
    assert {tuple(tree.filters) for tree in trees} == {
        (tuple(FILTERS['HAAR']),)}


@pytest.mark.parametrize('arm', [dict(filter_kw = 'B3SPLINE'),
                                 dict(synthesis_operator = 'TRANSPOSED')])
def test_an_arm_that_is_not_one_is_refused_at_the_door(arm):
    """Both axes are checked before any tree is scheduled (#108's rule).

    ``CG_minimization`` checks the operator too, and that check stays — but it
    fires from inside the first solve, on a worker process under any backend
    but the serial one, where the traceback lands in the backend's log and not
    the run's. The filter has no second checker at all: ``restore_object`` is
    where the name becomes an array.
    """

    with pytest.raises(DawisValueError):
        _restore(_trees(), **arm)


@pytest.mark.parametrize('threshold', sorted(ROUTED))
def test_no_single_threshold_moves_an_atom(threshold):
    """Inverted one at a time, so a failure says which knob came back."""

    published = _trees()
    inverted = _trees()
    _restore(published)
    _restore(inverted, **{threshold: UNROUTED[threshold]})

    assert ([tree.operators for tree in published]
            == [tree.operators for tree in inverted])
    assert ([tree.filters for tree in published]
            == [tree.filters for tree in inverted])
