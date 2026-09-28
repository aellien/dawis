"""The criterion protocol, and the algebra that composes it (#125).

A **selection criterion** is a vectorised predicate over a run's atom index:
it takes the structured array :func:`dawis.read_atom_index` returns and gives
back one boolean per atom. It reads columns, never pixels, so a whole
selection runs on the index alone — which is what #118's store was for.

A criterion is *just a callable*, so a user writes one with ``def`` and needs
nothing from this module. :class:`Criterion` is the wrapper that adds the
algebra — ``&``, ``|`` and ``~`` — and checks that what comes back is a mask
of the right length. :func:`split` is the one composer this module ships: it
returns **both** sides, so an atom is never silently lost.

The criteria themselves — level, size, distance, mask, kurtosis — and the
ordered cascade are #126's; this is the protocol they are written against.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from dawis.io.store import COMPUTED_COLUMNS

#: What a criterion is: index rows in, one boolean per row out.
Test = Callable[[np.ndarray], np.ndarray]


class Criterion:
    """A criterion, with the algebra to combine it with another.

    Wraps any callable of the index rows. Used as a decorator on a ``def``,
    or around a lambda::

        @Criterion
        def bright(index):
            return index['flux'] > 1e3

        faint_and_small = ~bright & Criterion(lambda i: i['level'] < 3)

    Parameters
    ----------
    test : callable
        Takes the index rows, returns a boolean array as long as them.

    Raises
    ------
    ValueError
        When called, if ``test`` does not return a boolean mask of the index's
        length — a mask of another length selects the wrong atoms quietly —
        or if it reads an index column the store does not carry.
    """

    def __init__(self, test: Test):
        # Unwrapped, so that wrapping a criterion again — which ``&``, ``|``
        # and :func:`split` all do to whatever they are handed — adds no
        # second layer of checking, and no second copy of its error message.
        self._test = test._test if isinstance(test, Criterion) else test

    def __call__(self, index: np.ndarray) -> np.ndarray:
        try:
            mask = self._test(index)
        except ValueError as failed:
            # Only when the failure names a column the index does not carry.
            # A criterion with a bug of its own — a bad reshape, a broadcast
            # that does not — raises ValueError too, and sending the user off
            # to re-run the pipeline for it would be a worse error than none.
            missing = [name for name in COMPUTED_COLUMNS
                       if name not in (index.dtype.names or ()) and name in str(failed)]
            if missing:
                raise ValueError('this criterion reads %s, which this atom store does not '
                                 'carry: it was written before the index gained the '
                                 'columns measured off the pixels, so re-run to select on '
                                 'them (%s)' % (', '.join(missing), failed)) from failed
            raise
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (len(index),):
            raise ValueError('a criterion must return one boolean per atom: %d rows in, '
                             '%s of %s out' % (len(index), mask.shape, mask.dtype))
        return mask

    def __and__(self, other: Test) -> Criterion:
        other = Criterion(other)
        return Criterion(lambda index: self(index) & other(index))

    def __or__(self, other: Test) -> Criterion:
        other = Criterion(other)
        return Criterion(lambda index: self(index) | other(index))

    def __invert__(self) -> Criterion:
        return Criterion(lambda index: ~self(index))


def split(index: np.ndarray, criterion: Test) -> tuple[np.ndarray, np.ndarray]:
    """The atoms a criterion keeps, and every atom it does not.

    Both sides, always, so cuts chain without losing an atom::

        icl, rest = split(index, big)
        satellites, rest = split(rest, small)

    Parameters
    ----------
    index : numpy.ndarray
        Index rows, from :func:`dawis.read_atom_index`.
    criterion : callable
        A :class:`Criterion`, or any callable of the same shape.

    Returns
    -------
    kept : numpy.ndarray
        The rows the criterion is True on, in the order they came in.
    rest : numpy.ndarray
        The rows it is False on. ``len(kept) + len(rest) == len(index)``.
    """

    mask = Criterion(criterion)(index)
    return index[mask], index[~mask]
