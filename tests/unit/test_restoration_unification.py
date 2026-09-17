"""Tier 0 checks that the worker count never selects an algorithm (#56).

Restoration used to have two branches. ``n_workers == 1`` looped over the trees
in process; anything else sorted the list by area, cut it into patches and
sent them to :func:`dawis.parallel.pmap`. #5 measured a 308% flux difference
between the two, and #20 decided the published path survives and the
divergence ends.

There is one path now, and these checks are what "one path" means: the same
call per tree whatever the worker count, the same atom for that tree, and the
same list back — same atoms, same order. The worker count reaches
:func:`~dawis.restoration.synthesis.restore_objects_default` and decides only
how the patches are spread.

The trees are fakes. What is under test is the dispatch — which tree is
restored, with which arguments, in which order — and a fake tree records
exactly that while a real one would spend a minute solving. The solver itself
is Tier 1 and Tier 2's business.

The multi-worker runs name the serial backend explicitly rather than letting
:func:`~dawis.parallel.make_backend` choose. Asking for four workers and no
backend asks for ray, which is the optional extra and is not installed here;
naming the backend exercises the same ``n_workers > 1`` dispatch without it.
Substituting one backend for another is exactly what the seam promises.
"""

from __future__ import annotations

import numpy as np
import pytest

from dawis import parallel
from dawis.restoration.synthesis import restore_objects_default

#: Restoration parameters shared by every run here. Chosen so the branch
#: taken is the plain B3-spline one for every fake tree: no tree is small or
#: eccentric enough for Haar, and none sits above ``lvl_sep_op``.
REST_PARAMS = dict(
    lvl_sep_big=6,
    rm_gamma_for_big=False,
    extent_sep=0.1,
    ecc_sep=0.95,
    lvl_sep_lin=2,
    lvl_sep_op=3,
    gamma=0.2,
    size_patch_small=1,
    size_patch_big=1,
    size_big_objects=512,
    cg_max_iter=3,
)


class _FakeMaximum:
    """The descriptors of an interscale maximum restoration reads."""

    def __init__(self, level: int, area: int, eccentricity: float = 0.5):
        self.level = level
        self.area = area
        self.eccentricity = eccentricity


#: Tags of the trees solved so far, in the order the solver reached them.
#: Class-level rather than per-tree because the order under test is a
#: property of the whole run, not of any one tree. ``_restore`` clears it.
SOLVE_ORDER: list[float] = []


class _FakeTree:
    """A tree that records how it was restored instead of solving.

    Carries the attributes :func:`restore_object` and :func:`restore_patch`
    read, and returns an image whose every pixel is ``value`` — distinct per
    tree, so an atom can be traced back to the tree it came from.
    """

    def __init__(self, value: float, level: int = 4, area: int = 100,
                 x_size: int = 8, y_size: int = 8, extent: float = 0.5,
                 eccentricity: float = 0.5):
        self.value = value
        self.interscale_maximum = _FakeMaximum(level, area, eccentricity)
        self.extent = extent
        self.eccentricity = eccentricity
        self.x_size = x_size
        self.y_size = y_size
        self.bbox = (0, 0, x_size, y_size)
        self.calls: list[dict] = []

    def CG_minimization(self, wavelet_cube, label_cube, filter, **kwargs):
        """Record the call, and hand back a flat image tagged with `value`."""
        self.calls.append(dict(filter=tuple(filter), **kwargs))
        SOLVE_ORDER.append(self.value)
        return np.full((self.x_size, self.y_size), self.value), 1.0, 2.0

    def det_err_tube(self, wavelet_cube, label_cube):
        return np.zeros((self.x_size, self.y_size, 2))


def _restore(trees, n_workers, **overrides):
    """Run restoration over `trees`, on `n_workers` workers, serially either way."""
    params = dict(REST_PARAMS, **overrides)
    SOLVE_ORDER.clear()
    with parallel.parallel_backend('serial', n_workers=n_workers):
        return restore_objects_default(
            trees,
            oimage=None, cg_gamma=0.01, niter=15,   # accepted and never read
            wavelet_cube=None, label_cube=None,     # only the fake trees see these
            n_workers=n_workers, **params
        )


def _values(atoms):
    """The tag of the tree each atom came from, in the order returned.

    Rounded because the tag is read back through the ``gamma`` attenuation
    the restorer applies, and the round trip is not exact. The tags are whole
    numbers, so rounding recovers them and nothing else.
    """
    return [round(float(np.max(atom.image)) / REST_PARAMS['gamma'])
            for atom in atoms]


def _trees(count, first=1, **kwargs):
    """`count` fake trees tagged consecutively from `first`."""
    return [_FakeTree(value=first + index, **kwargs) for index in range(count)]


# ---------------------------------------------------------------------------
# One path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('n_workers', [1, 2, 4, 8])
def test_every_tree_is_restored_exactly_once_whatever_the_worker_count(n_workers):
    trees = _trees(7)
    atoms = _restore(trees, n_workers)
    assert len(atoms) == 7
    assert [len(tree.calls) for tree in trees] == [1] * 7


@pytest.mark.parametrize('n_workers', [2, 4, 8])
def test_the_atoms_are_the_serial_atoms_in_the_serial_order(n_workers):
    serial = _values(_restore(_trees(7), 1))
    assert _values(_restore(_trees(7), n_workers)) == serial


@pytest.mark.parametrize('n_workers', [2, 4, 8])
def test_the_solver_is_called_with_the_same_arguments_on_every_worker_count(n_workers):
    serial_trees = _trees(5)
    _restore(serial_trees, 1)
    parallel_trees = _trees(5)
    _restore(parallel_trees, n_workers)
    assert [tree.calls for tree in parallel_trees] == \
           [tree.calls for tree in serial_trees]


def test_the_atoms_come_back_in_the_order_the_trees_were_given():
    atoms = _restore(_trees(5), 4)
    assert _values(atoms) == [1, 2, 3, 4, 5]


def test_the_tree_list_the_caller_passed_is_not_reordered():
    trees = _trees(5, area=100)
    trees[0].interscale_maximum.area = 1
    trees[-1].interscale_maximum.area = 10000
    before = list(trees)
    _restore(trees, 4)
    assert trees == before


# ---------------------------------------------------------------------------
# Patching: the three defects of the old parallel branch
# ---------------------------------------------------------------------------


def test_a_big_tree_at_the_end_of_the_list_is_not_dropped():
    """The old branch never flushed a big patch left over at the end."""
    trees = _trees(3) + [_FakeTree(value=4, x_size=600, y_size=600)]
    atoms = _restore(trees, 4, size_patch_big=8)
    assert _values(atoms) == [1, 2, 3, 4]


def test_a_big_tree_is_collected_once_and_not_once_per_following_small_tree():
    """The old branch re-collected the big accumulator it never cleared."""
    trees = [_FakeTree(value=1, x_size=600, y_size=600)] + _trees(4, first=2)
    atoms = _restore(trees, 4, size_patch_big=8)
    assert _values(atoms) == [1, 2, 3, 4, 5]


def test_a_tree_big_on_one_axis_only_is_not_a_big_object():
    """`x > s & y > s` chained as `x > (s & y) > s`, so wide trees counted big."""
    wide = _FakeTree(value=1, x_size=600, y_size=8)
    atoms = _restore([wide, _FakeTree(value=2)], 4, size_patch_big=8)
    assert _values(atoms) == [1, 2]
    assert len(wide.calls) == 1


@pytest.mark.parametrize('size_patch_small', [0, 1, 3, 100])
def test_the_patch_size_changes_nothing_but_the_size_of_a_patch(size_patch_small):
    """Zero included: a patch of no trees is not a unit of work, and asking
    for one must not lose a tree or ask ``range`` for a step of zero."""
    expected = _values(_restore(_trees(7), 1))
    atoms = _restore(_trees(7), 4, size_patch_small=size_patch_small)
    assert _values(atoms) == expected


@pytest.mark.parametrize('size_big_objects', [4, 512])
def test_where_the_big_small_split_falls_changes_nothing_but_the_schedule(
        size_big_objects):
    """The one thing the goldens cannot check, so it is checked here.

    Big trees are worked before small ones, so ``size_big_objects`` decides
    the order the solver is called in. Tier 1 can never exercise that: its
    cutout is 256x256, no tree can exceed the 512-pixel default on both axes,
    and the big list is always empty. Only Tier 2 reaches it in a real run.

    At 4 every tree here is big and at 512 none is, so the two runs schedule
    the list in opposite groupings — and must still return the same atoms in
    the same places, because restoring one tree reads nothing another wrote.
    """
    trees = _trees(6, x_size=8, y_size=8)
    assert _values(_restore(trees, 4, size_big_objects=size_big_objects)) == \
           _values(_restore(_trees(6), 1))


def test_big_trees_are_solved_first_and_still_come_back_in_tree_order():
    """The two halves of the scheduling claim, in one check.

    The long pole is submitted first so no worker is handed it last, and the
    reordering afterwards means the caller cannot tell that it was.
    """
    small_first = [_FakeTree(value=1),
                   _FakeTree(value=2, x_size=600, y_size=600)]
    atoms = _restore(small_first, 4)

    assert SOLVE_ORDER == [2, 1]
    assert _values(atoms) == [1, 2]


def test_no_trees_no_atoms():
    assert _restore([], 4) == []
