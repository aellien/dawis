"""Tier 0 — the criterion protocol a selection is written against (#125).

A criterion is a vectorised predicate over the atom index: index rows in, one
boolean per atom out, and no pixel read (#17, points 1 and 2). These pin the
four promises of that protocol — a mask of the right shape or an error, an
algebra that composes criteria, a :func:`split` that gives back both sides so
no atom is lost, and a selection that runs with the run's pixels gone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from test_dataclasses import a_tree  # noqa: E402
from test_store import SHAPE, an_atom  # noqa: E402

from dawis.io.store import COMPUTED_COLUMNS, INDEX_DTYPE, AtomStore  # noqa: E402
from dawis.selection import Criterion, split  # noqa: E402


def an_index(levels=(1, 2, 3, 4), fluxes=(10., 20., 30., 40.)) -> np.ndarray:
    rows = np.zeros(len(levels), dtype=INDEX_DTYPE)
    rows['level'] = levels
    rows['flux'] = fluxes
    rows['atom_id'] = np.arange(len(levels))
    return rows


@Criterion
def big(index):
    """Every atom above level 2."""
    return index['level'] > 2


@Criterion
def bright(index):
    return index['flux'] > 25.


def test_a_criterion_returns_one_boolean_per_atom():
    index = an_index()

    np.testing.assert_array_equal(big(index), [False, False, True, True])


def test_the_algebra_composes_criteria():
    index = an_index(levels=(1, 3, 3), fluxes=(10., 10., 40.))

    np.testing.assert_array_equal((big & bright)(index), [False, False, True])
    np.testing.assert_array_equal((big | bright)(index), [False, True, True])
    np.testing.assert_array_equal((~big)(index), [True, False, False])
    np.testing.assert_array_equal((~(big & bright))(index), [True, True, False])


def test_the_algebra_takes_a_plain_callable_on_the_right():
    """A user writes criteria with ``def``; they need nothing from us (#17)."""
    index = an_index()

    np.testing.assert_array_equal((big & (lambda i: i['level'] < 4))(index),
                                  [False, False, True, False])


def test_split_gives_back_both_sides():
    index = an_index()

    kept, rest = split(index, big)

    assert list(kept['level']) == [3, 4]
    assert list(rest['level']) == [1, 2]
    assert len(kept) + len(rest) == len(index)


def test_split_chains_without_losing_an_atom():
    """The cut-then-cut-the-rest shape #17 asks for: a partition, always."""
    index = an_index()

    first, rest = split(index, big)
    second, rest = split(rest, Criterion(lambda i: i['level'] == 2))

    assert sorted(np.concatenate([first, second, rest])['atom_id'].tolist()) == [0, 1, 2, 3]


def test_split_takes_a_bare_callable():
    kept, rest = split(an_index(), lambda i: i['flux'] > 25.)

    assert list(kept['flux']) == [30., 40.]
    assert list(rest['flux']) == [10., 20.]


def test_an_empty_index_splits_into_two_empty_halves():
    kept, rest = split(an_index(levels=(), fluxes=()), big)

    assert len(kept) == len(rest) == 0


@pytest.mark.parametrize('bad, why', [
    (lambda index: np.array([True, False]), 'a mask of the wrong length'),
    (lambda index: index['level'], 'not a boolean mask at all'),
])
def test_a_criterion_that_does_not_return_a_mask_raises(bad, why):
    """The failure worth catching: a wrong mask selects wrong atoms quietly."""
    with pytest.raises(ValueError, match='one boolean per atom'):
        Criterion(bad)(an_index())


def test_a_criterion_on_a_missing_column_raises_and_names_the_store():
    """Reading a column an old store lacks raises — only when it is read (#17).

    The store warns at read time; the raise arrives here, and only for a
    criterion that actually touches one of the four columns.
    """
    old = np.zeros(3, dtype=[(name, INDEX_DTYPE[name]) for name in INDEX_DTYPE.names
                             if name not in COMPUTED_COLUMNS])
    old['level'] = (1, 3, 4)

    np.testing.assert_array_equal(big(old), [False, True, True])
    with pytest.raises(ValueError, match='this criterion reads flux'):
        bright(old)


def test_a_criterion_with_a_bug_of_its_own_is_not_blamed_on_the_store():
    """An old index does not make every ValueError the store's fault.

    The rewrite only fires when the failure names a column the index lacks;
    a broadcast that does not fit is the criterion's own bug, and telling the
    user to re-run the pipeline for it would be the worse error.
    """
    old = np.zeros(3, dtype=[(name, INDEX_DTYPE[name]) for name in INDEX_DTYPE.names
                             if name not in COMPUTED_COLUMNS])
    broken = Criterion(lambda i: i['level'] > np.zeros(5))

    with pytest.raises(ValueError) as raised:
        broken(old)
    assert 'atom store' not in str(raised.value)


def test_a_selection_reads_zero_pixels(tmp_path):
    """The whole point of the index: the run's pixels need not be there.

    The store is deleted after the index is read, so a selection that touched
    a single pixel could not finish. That is the strongest form of the check
    the ticket asks for — zero pixel reads — and it needs no instrumentation.
    """
    import dawis

    outpath = str(tmp_path / 'run')
    with AtomStore(outpath + '.atoms.hdf5', SHAPE) as store:
        store.append(1, [an_atom(0.), an_atom(100., (0, 0, 1, 5))],
                     [a_tree(), a_tree()])

    index = dawis.read_atom_index(outpath)
    Path(outpath + '.atoms.hdf5').unlink()

    icl, rest = split(index, bright & Criterion(lambda i: ~np.isnan(i['centroid_x'])))

    assert len(icl) + len(rest) == 2
    assert not Path(outpath + '.atoms.hdf5').exists()
