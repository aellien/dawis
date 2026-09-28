"""Tier 0 — the ordering construct, and the mask sort over it (#126).

A bag of independent predicates cannot express what selection did: CFIS is an
ordered three-pass cascade with carry-over lists, so the **order is part of
the selection** (#17, point 2). These pin what :class:`~dawis.selection.Cascade`
promises — first match claims, leftovers fall through, a partition by
construction, a repeated name concatenated — that
:func:`~dawis.selection.mask_sort` is that same construct and not a second
one (point 5), and the thing the map's weakest claim turns on: that the CFIS
cascade is **expressible** in the shipped blocks without adding to them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from test_selection_criteria import an_index  # noqa: E402

from dawis.selection import (  # noqa: E402
    Cascade,
    Distance,
    InMask,
    Kurtosis,
    Level,
    Size,
    mask_sort,
)


def a_mask(*points, shape=(100, 100)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for x, y in points:
        mask[x, y] = True
    return mask


def test_the_first_matching_rule_claims_the_atom():
    index = an_index(levels=(1, 5, 9))

    groups = Cascade([(Level(5), 'big'), (Level(9), 'huge')])(index)

    assert list(groups['big']['level']) == [5, 9]
    assert list(groups['huge']['level']) == []


def test_swapping_two_rules_gives_a_different_partition():
    """Which is the whole reason an ordering construct exists (#17)."""
    index = an_index(levels=(1, 5, 9))
    rules = [(Level(5), 'big'), (Level(9), 'huge')]

    first = Cascade(rules)(index)
    second = Cascade(rules[::-1])(index)

    assert list(first['huge']['level']) == []
    assert list(second['huge']['level']) == [9]


def test_a_repeated_name_collects_every_rule_that_uses_it():
    """CFIS gives ICL three separate rules (:526, :529, :569)."""
    index = an_index(levels=(1, 2, 3, 4))

    groups = Cascade([(Level(4), 'icl'),
                      (Level(maximum=1), 'icl'),
                      (Level(2, 2), 'satellites')])(index)

    assert list(groups['icl']['level']) == [4, 1]
    assert list(groups['satellites']['level']) == [2]
    assert list(groups['unclassified']['level']) == [3]


def test_every_atom_lands_in_exactly_one_group():
    index = an_index(levels=(1, 2, 3, 4, 5, 6, 7, 8))

    groups = Cascade([(Level(6), 'a'), (Level(3, 4), 'b'), (Level(maximum=1), 'c')])(index)

    landed = np.concatenate([rows['atom_id'] for rows in groups.values()])
    assert sorted(landed.tolist()) == list(range(len(index)))


def test_the_leftovers_group_is_there_even_when_it_is_empty():
    """An empty leftover is a fact about the selection, not an absence."""
    index = an_index(levels=(5, 6))

    groups = Cascade([(Level(1), 'everything')])(index)

    assert 'unclassified' in groups
    assert len(groups['unclassified']) == 0


def test_an_empty_index_still_gives_every_group():
    groups = Cascade([(Level(5), 'big')])(an_index(levels=(), fluxes=()))

    assert sorted(groups) == ['big', 'unclassified']
    assert all(len(rows) == 0 for rows in groups.values())


def test_a_rule_named_like_the_leftovers_is_refused():
    """Otherwise 'no rule claimed it' and 'this rule did' merge silently."""
    with pytest.raises(ValueError, match='the atoms no rule claimed'):
        Cascade([(Level(5), 'unclassified')])


def test_the_leftovers_group_can_be_renamed():
    groups = Cascade([(Level(5), 'big')], unclassified='rest')(an_index())

    assert sorted(groups) == ['big', 'rest']


def test_mask_sort_is_a_cascade_of_mask_criteria_and_nothing_else():
    """One code path, so the two cannot drift apart (#17, point 5)."""
    index = an_index(levels=(1, 2, 3), centroids=[(1., 1.), (5., 5.), (9., 9.)])
    masks = {'icl': a_mask((1, 1), shape=(16, 16)),
             'satellites': a_mask((5, 5), shape=(16, 16))}

    sorted_groups = mask_sort(index, masks)
    by_hand = Cascade([(InMask(mask), name) for name, mask in masks.items()])(index)

    assert list(sorted_groups) == list(by_hand)
    for name in by_hand:
        np.testing.assert_array_equal(sorted_groups[name]['atom_id'],
                                      by_hand[name]['atom_id'])
    assert list(sorted_groups['unclassified']['atom_id']) == [2]


def test_mask_sort_gives_an_overlapping_atom_to_the_first_mask():
    index = an_index(levels=(1,), centroids=[(3., 3.)])
    both = {'first': a_mask((3, 3), shape=(16, 16)),
            'second': a_mask((3, 3), shape=(16, 16))}

    groups = mask_sort(index, both)

    assert len(groups['first']) == 1
    assert len(groups['second']) == 0


# --- the claim this ticket is really about ---------------------------------


CENTRE = (50., 50.)
RADIUS = 30.          # R, :521-523
LVL_SEP = 5           # :536
LVL_SEP_BCG = 4       # :569
SIZE_SEP_PIX = 6.     # :536, and — as a *distance* — :569
BELOW_ZERO = np.nextafter(0., -1.)


def cfis_index() -> np.ndarray:
    """Nine atoms, one per branch of the CFIS cascade."""
    positions = [(50., 50.), (95., 95.), (60., 60.), (40., 40.), (52., 52.),
                 (30., 30.), (45., 45.), (50., 53.), (45., 65.)]
    half = [1, 1, 1, 1, 1, 4, 1, 1, 1]
    bboxes = [(int(x) - h, int(y) - h, int(x) + h, int(y) + h)
              for (x, y), h in zip(positions, half)]
    return an_index(levels=(2, 2, 2, 2, 1, 6, 3, 4, 2),
                    fluxes=tuple(10. for _ in positions),
                    bboxes=bboxes, centroids=positions,
                    kurtoses=(-1.,) + tuple(1. for _ in positions[1:]))


def cfis_cascade(icl_mask, bcg, stars, satellites) -> Cascade:
    """``synthesis_bcgwavsizesep_with_masks`` written in the shipped blocks.

    Three passes with carry-over lists is one ordered rule list: pass 2's
    input is whatever pass 1 did not claim, which is what a rule further down
    the list sees. Nothing here is a construct #17 did not name.
    """

    return Cascade([
        # :505-507's test is ``k < 0``, so the bound is exclusive — and an
        # upper bound rather than a negated cut, because ``~Kurtosis`` would
        # also claim an atom whose kurtosis is undefined, which CFIS left in
        # play.
        (Kurtosis(maximum = BELOW_ZERO), 'artefacts'),
        (~Distance(CENTRE, maximum = RADIUS), 'outside'),               # :521-523
        (~InMask(stars) & InMask(icl_mask), 'icl'),                     # :526
        (InMask(bcg), 'icl'),                                           # :529
        (Level(LVL_SEP) & Size(SIZE_SEP_PIX), 'icl'),                    # :536
        (~InMask(stars) & ~Level(LVL_SEP) & InMask(satellites),
         'satellites'),                                                 # :549
        (Level(LVL_SEP_BCG) & Distance(CENTRE, maximum = SIZE_SEP_PIX),
         'icl'),                                                        # :569
    ])


def test_the_cfis_cascade_is_expressible_in_the_shipped_blocks():
    """The map's weakest claim, made checkable.

    Every rule of the published pipeline is one of the six criteria, its
    negation, or an ``&`` of them, and the three passes are one ordered list.
    If this needed a construct #17 did not name, the spec would be wrong.
    """
    groups = cfis_cascade(icl_mask=a_mask((40, 40), (60, 60)),
                          bcg=a_mask((52, 52)),
                          stars=a_mask((60, 60)),
                          satellites=a_mask((45, 45)))(cfis_index())

    assert list(groups['artefacts']['atom_id']) == [0]
    assert list(groups['outside']['atom_id']) == [1]
    assert list(groups['icl']['atom_id']) == [3, 4, 5, 7]
    assert list(groups['satellites']['atom_id']) == [6]
    assert list(groups['unclassified']['atom_id']) == [2, 8]


def test_the_cfis_bcg_rule_short_circuits_the_level_and_size_test():
    """:529 claims a BCG atom with no level or size test at all."""
    groups = cfis_cascade(icl_mask=a_mask((0, 0)), bcg=a_mask((52, 52)),
                          stars=a_mask((0, 0)), satellites=a_mask((0, 0)))(cfis_index())

    claimed = cfis_index()[4]
    assert claimed['level'] < LVL_SEP
    assert claimed['atom_id'] in groups['icl']['atom_id']


def test_the_cfis_rescue_pass_only_sees_what_two_passes_rejected():
    """:569 runs over ``unclass_al``, which is the tail of the rule list.

    Give the same atom to the ICL mask and the rescue rule stops mattering
    for it — the proof that the rescue is last, not independent.
    """
    index = cfis_index()
    masks = dict(bcg=a_mask((0, 0)), stars=a_mask((0, 0)), satellites=a_mask((0, 0)))

    rescued = cfis_cascade(icl_mask=a_mask((0, 0)), **masks)(index)
    claimed_earlier = cfis_cascade(icl_mask=a_mask((50, 53)), **masks)(index)

    assert list(rescued['icl']['atom_id']) == [5, 7]
    assert list(claimed_earlier['icl']['atom_id']) == [7, 5]


def test_an_atom_with_no_kurtosis_is_not_called_an_artefact():
    """CFIS's ``k < 0`` was False for a NaN, so the atom stayed in play.

    ``~Kurtosis(0.)`` is True on a NaN, so spelling the artefact rule as a
    negated cut would take a one-pixel atom out of the selection before any
    other rule saw it. The shipped cascade uses an upper bound instead.
    """
    index = cfis_index()
    index['kurtosis'][3] = np.nan          # the atom the ICL mask claims

    groups = cfis_cascade(icl_mask=a_mask((40, 40)), bcg=a_mask((0, 0)),
                          stars=a_mask((0, 0)), satellites=a_mask((0, 0)))(index)

    assert 3 not in groups['artefacts']['atom_id']
    assert 3 in groups['icl']['atom_id']
