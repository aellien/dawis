"""The criterion protocol, the algebra that composes it, and the six criteria.

The protocol and the algebra are #125's; the criteria are #126's.

A **selection criterion** is a vectorised predicate over a run's atom index:
it takes the structured array :func:`dawis.read_atom_index` returns and gives
back one boolean per atom. It reads columns, never pixels, so a whole
selection runs on the index alone — which is what #118's store was for.

A criterion is *just a callable*, so a user writes one with ``def`` and needs
nothing from this module. :class:`Criterion` is the wrapper that adds the
algebra — ``&``, ``|`` and ``~`` — and checks that what comes back is a mask
of the right length. :func:`split` is the one composer this module ships: it
returns **both** sides, so an atom is never silently lost.

The six criteria that ship are here too — :class:`Level`, :class:`Size`,
:class:`Distance`, :class:`InMask`, :class:`Kurtosis`,
:class:`Eccentricity` (#17, point 4). Each is a :class:`Criterion`, so each
composes with the algebra and with a user's own ``def``. The ordering
construct that turns them into a partition is :mod:`dawis.selection.cascade`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from dawis.io.store import COMPUTED_COLUMNS

#: What a criterion is: index rows in, one boolean per row out.
Test = Callable[[np.ndarray], np.ndarray]


class MissingColumn(ValueError):
    """A criterion read an index column this atom store was written without.

    A :exc:`ValueError`, because that is what numpy raises for the missing
    field and what callers already catch. It has its own name only so that a
    composed criterion can tell its own message apart from a fresh failure and
    re-raise it unchanged.
    """


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
        # second layer of checking.
        self._test = test._test if isinstance(test, Criterion) else test

    def __call__(self, index: np.ndarray) -> np.ndarray:
        try:
            mask = self._test(index)
        except MissingColumn:
            # Already ours, from a criterion this one composes. ``&``, ``|``
            # and ``~`` wrap a lambda rather than a :class:`Criterion`, so the
            # unwrap above cannot reach it and the sentence would otherwise be
            # rewritten once per level of composition.
            raise
        except ValueError as failed:
            # Only when the failure names a column the index does not carry.
            # A criterion with a bug of its own — a bad reshape, a broadcast
            # that does not — raises ValueError too, and sending the user off
            # to re-run the pipeline for it would be a worse error than none.
            missing = [name for name in COMPUTED_COLUMNS
                       if name not in (index.dtype.names or ()) and name in str(failed)]
            if missing:
                raise MissingColumn(
                    'this criterion reads %s, which this atom store does not '
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


def _in_range(values: np.ndarray, minimum, maximum) -> np.ndarray:
    """``minimum <= values <= maximum``, both bounds inclusive and optional.

    A NaN never passes. It is what a float index column holds where the
    quantity is not defined — an atom of no flux has no centroid, an atom
    whose pixels are all one value has no kurtosis — and "not defined" is not
    "inside the range".

    So a NaN lands on the **negated** side of the cut, and that is not always
    the harmless place for it: see :class:`Kurtosis`, whose negation is the
    side a user reaches for.
    """

    keep = np.ones(values.shape, dtype = bool)
    if values.dtype.kind == 'f':
        keep &= ~np.isnan(values)
    if minimum is not None:
        keep &= values >= minimum
    if maximum is not None:
        keep &= values <= maximum
    return keep


def _bounds(minimum, maximum, what: str) -> None:
    """Refuse the two bound pairs that are not a cut.

    No bound at all keeps every atom, and an inverted band keeps none. Both
    are cuts that look like cuts and are not: inside a :class:`Cascade` the
    named group comes back empty, its atoms go to ``unclassified``, and
    nothing says why.
    """

    if minimum is None and maximum is None:
        raise ValueError('%s needs a minimum, a maximum, or both: with neither it keeps '
                         'every atom, which is a cut that looks like a cut and is not'
                         % what)
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError('%s has its bounds the wrong way round — minimum %r above '
                         'maximum %r — so it keeps no atom at all'
                         % (what, minimum, maximum))


def _as_pixels(units, minimum, maximum):
    """The two bounds in pixels: unchanged, or converted from kpc by ``units``."""

    if units is None:
        return minimum, maximum
    return (None if minimum is None else units.kpc_to_pix(minimum),
            None if maximum is None else units.kpc_to_pix(maximum))


def atom_position(index: np.ndarray,
                  centroid: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Where each atom *is*, for the criteria that test a position.

    The flux-weighted centroid by default, the centre of the bounding box on
    request or wherever the centroid could not be computed — which is
    ``CONTEXT.md``'s *Atom position*, and the charting ruling behind it. The
    centroid is read off the index, never recomputed: #125 has the pipeline
    write it at the point where it already holds the pixels.

    Both are in frame coordinates, along the axes of ``bbox``, so ``x`` runs
    down the frame's first axis. A bounding box ``(x0, y0, x1, y1)`` is
    half-open, so its centre is at ``(x0 + x1 - 1) / 2``: the middle of the
    pixels it covers, on the same scale as the centroid.

    Parameters
    ----------
    index : numpy.ndarray
        Index rows, from :func:`dawis.read_atom_index`.
    centroid : bool, optional
        False to use the bounding-box centre for every atom, which is the
        only position a store written before #125 carries.

    Returns
    -------
    x, y : numpy.ndarray
        One position per atom, as floats.
    """

    bbox = index['bbox']
    centre_x = (bbox[:, 0] + bbox[:, 2] - 1) / 2.
    centre_y = (bbox[:, 1] + bbox[:, 3] - 1) / 2.
    if not centroid:
        return centre_x, centre_y
    x, y = index['centroid_x'], index['centroid_y']
    failed = np.isnan(x) | np.isnan(y)
    return np.where(failed, centre_x, x), np.where(failed, centre_y, y)


class Level(Criterion):
    """Atoms whose wavelet level is in a range, both bounds inclusive.

    ``Level(5)`` is CFIS's ``lvlo >= lvl_sep`` (``:536``), and ``~Level(5)``
    is the ``lvlo < lvl_sep`` its satellite rule tested against the same
    threshold (``:549``).

    CFIS also had an upper level cut, and it **never fired**: ``lvl_sep_max``
    was 1000 against ``n_levels = 10`` (``:866-867``), so the published run
    used none of it (#17). ``maximum`` is here because a bounded level band is
    an ordinary thing to want, not because that rule is being reproduced.

    Parameters
    ----------
    minimum, maximum : int, optional
        The level range to keep. At least one is required.
    """

    def __init__(self, minimum = None, maximum = None):
        _bounds(minimum, maximum, 'a level criterion')
        super().__init__(lambda index: _in_range(index['level'], minimum, maximum))


class Size(Criterion):
    """Atoms whose bounding box is in a size range, in pixels or in kpc.

    The size of an atom is the two sides of its bounding box, and the two
    CFIS papers disagreed about how to combine them:
    ``make_synthesis_sdssrm_xcs.py:536`` needs **both** sides past the
    threshold, ``make_bkg_maps_for_FGs.py:109`` needs **either**. Both are
    here, both sides is the default, and the choice is the user's (#17,
    point 4).

    Parameters
    ----------
    minimum, maximum : float, optional
        The size range to keep, in pixels — or in kpc when ``units`` is
        given. At least one is required.
    either : bool, optional
        True to keep an atom whose *either* side is in range, instead of the
        default, which needs both.
    units : dawis.selection.Units, optional
        A units frame. With it, ``minimum`` and ``maximum`` are read as kpc
        and converted to pixels once, here.
    """

    def __init__(self, minimum = None, maximum = None, *, either: bool = False,
                 units = None):
        _bounds(minimum, maximum, 'a size criterion')
        minimum, maximum = _as_pixels(units, minimum, maximum)

        def test(index):
            bbox = index['bbox']
            in_x = _in_range(bbox[:, 2] - bbox[:, 0], minimum, maximum)
            in_y = _in_range(bbox[:, 3] - bbox[:, 1], minimum, maximum)
            return in_x | in_y if either else in_x & in_y

        super().__init__(test)


class Distance(Criterion):
    """Atoms whose position is within a distance of a point, in pixels or kpc.

    CFIS's analysis radius (``:521-523``, where ``dR > R`` drops the atom) and
    its BCG-halo rescue (``:569``, inside a radius) are both this criterion.
    The ``rc`` cut of ``:934`` is not: it was computed and never read, so it
    does not ship (#17).

    Parameters
    ----------
    centre : tuple of float
        The point to measure from, ``(x, y)`` in frame pixels, on the axes of
        ``bbox`` — so ``x`` runs down the frame's first axis.
    minimum, maximum : float, optional
        The distance range to keep, in pixels — or in kpc when ``units`` is
        given. Both bounds inclusive; at least one is required, and both are
        keyword-only. An annulus is the one criterion here whose bounds read
        naturally in the other order — ``Distance(centre, 30.)`` looks like a
        radius and would be a minimum — so neither may be given positionally.
    units : dawis.selection.Units, optional
        A units frame. With it, ``minimum`` and ``maximum`` are read as kpc.
    centroid : bool, optional
        False to measure from every atom's bounding-box centre rather than
        its flux-weighted centroid. See :func:`atom_position`.
    """

    def __init__(self, centre, *, minimum = None, maximum = None, units = None,
                 centroid: bool = True):
        _bounds(minimum, maximum, 'a distance criterion')
        minimum, maximum = _as_pixels(units, minimum, maximum)
        centre_x, centre_y = (float(v) for v in centre)

        def test(index):
            x, y = atom_position(index, centroid)
            return _in_range(np.hypot(x - centre_x, y - centre_y), minimum, maximum)

        super().__init__(test)


class InMask(Criterion):
    """Atoms whose position falls on a nonzero pixel of a mask.

    The general form of CFIS's ICL, BCG, star and satellite masks, which are
    all ``mask[xco, yco] > 0`` on an atom position (``:526``, ``:529``,
    ``:549``) — and the star mask's veto is this criterion negated.

    The position is **rounded** to the nearest pixel, not truncated. CFIS cast
    it with ``int()`` (``:500``), which truncates, so a centroid of 7.8 named
    pixel 7 rather than the pixel the atom sits on.

    A position exactly between two pixels rounds **up**, not to the even one.
    ``np.rint`` rounds half to even, and a bounding-box centre is exactly a
    half-integer whenever the box side is even — so with ``np.rint`` the pixel
    a box lands on would depend on the parity of its origin, and two atoms of
    the same shape on the same mask edge would sort differently.

    An atom whose position falls outside the mask is not in it. The mask need
    not have the frame's shape, so that is how a mask smaller than the image
    behaves, not only how the frame's own edge does.

    Parameters
    ----------
    mask : array_like
        A 2D mask, on the axes of the frame: ``mask[x, y]``. Any nonzero
        value is inside it, so a boolean mask and a labelled one both work —
        including a labelled mask that uses a negative sentinel.
    centroid : bool, optional
        False to test the bounding-box centre. See :func:`atom_position`.
    """

    def __init__(self, mask, centroid: bool = True):
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError("a mask criterion needs a 2D mask on the frame's axes: "
                             'got shape %s' % (mask.shape,))

        def test(index):
            x, y = atom_position(index, centroid)
            xi = np.floor(x + 0.5).astype(int)
            yi = np.floor(y + 0.5).astype(int)
            inside = ((xi >= 0) & (xi < mask.shape[0])
                      & (yi >= 0) & (yi < mask.shape[1]))
            keep = np.zeros(len(index), dtype = bool)
            keep[inside] = mask[xi[inside], yi[inside]] != 0
            return keep

        super().__init__(test)


class Kurtosis(Criterion):
    """Atoms whose excess kurtosis is in a range, both bounds inclusive.

    ``Kurtosis(minimum = 0.)`` is the CFIS artefact cut kept the right way up:
    ``:505-507`` calls an atom of **negative** excess kurtosis an artefact, so
    the atoms worth keeping are the ones this criterion keeps.

    The column is the **excess** kurtosis over every pixel of the bounding
    box, which is what ``scipy.stats.kurtosis(..., fisher = True)`` returns
    and what the pipeline writes (#125). It is NaN for an atom that has none —
    one pixel, or every pixel the same — and a NaN never passes, here as in
    every other criterion.

    **To claim the artefacts in a cascade, do not negate this.** ``~Kurtosis``
    is True wherever the column is NaN, so a rule spelled
    ``(~Kurtosis(0.), 'artefacts')`` calls every atom whose kurtosis is
    *undefined* an artefact and takes it out of the selection before any other
    rule sees it — which is the opposite of CFIS, whose ``k < 0`` was False
    for a NaN and left the atom in play. Use an upper bound instead::

        below_zero = np.nextafter(0., -1.)      # :505's test is exclusive
        Cascade([(Kurtosis(maximum = below_zero), 'artefacts'), ...])

    Parameters
    ----------
    minimum, maximum : float, optional
        The range to keep. At least one is required.
    """

    def __init__(self, minimum = None, maximum = None):
        _bounds(minimum, maximum, 'a kurtosis criterion')
        super().__init__(lambda index: _in_range(index['kurtosis'], minimum, maximum))


class Eccentricity(Criterion):
    """Atoms whose eccentricity is in a range, both bounds inclusive.

    The morphology cut #17 names at point 4. The column is the atom's own
    ``eccentricity``, which the index has carried since #122 — so this
    criterion runs on a store written before #125 as well.

    Parameters
    ----------
    minimum, maximum : float, optional
        The range to keep, 0 for a circle and 1 for a line. At least one is
        required.
    """

    def __init__(self, minimum = None, maximum = None):
        _bounds(minimum, maximum, 'an eccentricity criterion')
        super().__init__(lambda index: _in_range(index['eccentricity'], minimum, maximum))
