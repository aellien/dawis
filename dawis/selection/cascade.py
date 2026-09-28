"""The ordering construct, and the mask sort that is one line of it (#126).

:func:`~dawis.selection.split` separates one group from the rest and says
nothing about order. That is not enough to express what selection actually
did: CFIS is an ordered **three-pass cascade** with carry-over lists
(``noticl_al`` at ``:483``, ``unclass_al`` at ``:484``, an ICL rescue at
``:569``), where the BCG mask claims an atom with no level or size test at all
(``:529``) and the satellite rule tests level *below* the very threshold the
ICL rule tests *above* (``:536``, ``:549``). Compose the same criteria in
another order and the partition is different, so **the order is part of the
selection** (#17, point 2).

:class:`Cascade` is that construct, and it is the only one: :func:`mask_sort`
is a wrapper over it, not a second implementation, so a mask sort cannot drift
away from the cascade it is (#17, point 5).
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from dawis.selection.criteria import InMask, Test, split

#: What a cascade calls the atoms no rule claimed.
UNCLASSIFIED = 'unclassified'


class Cascade:
    """An ordered list of ``(criterion, name)`` rules, applied top to bottom.

    Each atom walks the list from the top and the **first** matching rule
    claims it; whatever no rule claims falls through to
    :data:`UNCLASSIFIED`. A claimed atom leaves the pool, so the result is a
    partition **by construction** — no atom in two groups, no atom lost —
    which is the map's standing invariant that an atom belongs to at most one
    source model.

    A name may appear more than once. CFIS gives ICL three separate rules
    (``:526``, ``:529``, ``:569``), so the rows they claim are concatenated
    into one group, in the order the rules ran::

        cascade = Cascade([
            (InMask(bcg), 'icl'),
            (Level(5) & Size(size_sep_pix), 'icl'),
            (~InMask(star) & ~Level(5) & InMask(satellites), 'satellites'),
        ])
        groups = cascade(index)

    Parameters
    ----------
    rules : iterable of (callable, str)
        The rules, in the order they are applied. Each criterion is anything
        :class:`~dawis.selection.Criterion` accepts, including a plain
        ``def``.
    unclassified : str, optional
        What to call the fall-through group. It is always in the result, even
        when it is empty — an empty leftover group is a fact about the
        selection, not an absence.

    Raises
    ------
    ValueError
        If ``unclassified`` is also a rule's name. The two would merge
        silently, and "no rule claimed it" and "this rule claimed it" are the
        one distinction a cascade exists to make.
    """

    def __init__(self, rules: Iterable[tuple[Test, str]],
                 unclassified: str = UNCLASSIFIED):
        self.rules = [(test, str(name)) for test, name in rules]
        self.unclassified = str(unclassified)
        if any(name == self.unclassified for _, name in self.rules):
            raise ValueError('a rule is named %r, which is also what this cascade calls '
                             'the atoms no rule claimed: the two would merge and the '
                             'leftovers would stop being readable'
                             % (self.unclassified,))

    def __call__(self, index: np.ndarray) -> dict[str, np.ndarray]:
        """The groups of ``index``, in rule order, leftovers last.

        Parameters
        ----------
        index : numpy.ndarray
            Index rows, from :func:`dawis.read_atom_index`.

        Returns
        -------
        dict of str to numpy.ndarray
            One entry per distinct rule name, in the order the names first
            appear, then :attr:`unclassified`. Every row of ``index`` is in
            exactly one of them.
        """

        groups: dict[str, np.ndarray] = {}
        rest = index
        for test, name in self.rules:
            kept, rest = split(rest, test)
            groups[name] = (np.concatenate([groups[name], kept])
                            if name in groups else kept)
        groups[self.unclassified] = rest
        return groups


def mask_sort(index: np.ndarray, masks: Mapping[str, np.ndarray],
              centroid: bool = True,
              unclassified: str = UNCLASSIFIED) -> dict[str, np.ndarray]:
    """Sort atoms into groups by which mask their position falls in.

    The general form of the ICL / BCG / satellite split, and a **named
    convenience over a cascade of mask criteria** rather than a mechanism of
    its own: ``masks`` is walked in order, the first mask an atom's position
    falls in claims it, and an atom no mask claims is unclassified.

    Parameters
    ----------
    index : numpy.ndarray
        Index rows, from :func:`dawis.read_atom_index`.
    masks : mapping of str to array_like
        Group name to 2D mask, in the order they are tested. Order matters
        wherever two masks overlap, which is why this takes a mapping and not
        a set.
    centroid : bool, optional
        False to test each atom's bounding-box centre rather than its
        flux-weighted centroid. See
        :func:`~dawis.selection.atom_position`.
    unclassified : str, optional
        What to call the atoms no mask claimed.

    Returns
    -------
    dict of str to numpy.ndarray
        As :class:`Cascade` returns: one entry per mask, then the leftovers.
    """

    return Cascade([(InMask(mask, centroid), name) for name, mask in masks.items()],
                   unclassified)(index)
