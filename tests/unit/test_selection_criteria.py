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
from dawis.selection import (  # noqa: E402
    Criterion,
    Distance,
    Eccentricity,
    InMask,
    Kurtosis,
    Level,
    Size,
    Units,
    atom_position,
    split,
)


def an_index(levels=(1, 2, 3, 4), fluxes=None, bboxes=None,
             centroids=None, kurtoses=None, eccentricities=None) -> np.ndarray:
    """An index of ``len(levels)`` atoms, every column a criterion reads set.

    The defaults are deliberately dull — a 2x2 box at the origin, a centroid
    at its centre, zero kurtosis — so that a test that cares about one column
    can set that column and nothing else. ``levels`` fixes the atom count and
    every other default follows it.
    """
    n = len(levels)
    rows = np.zeros(n, dtype=INDEX_DTYPE)
    rows['level'] = levels
    rows['flux'] = 10. * (np.arange(n) + 1) if fluxes is None else fluxes
    rows['atom_id'] = np.arange(n)
    rows['bbox'] = (np.asarray(bboxes, dtype=int).reshape(n, 4) if bboxes is not None
                    else np.tile([0, 0, 2, 2], (n, 1)))
    if centroids is None:
        centroids = np.column_stack([(rows['bbox'][:, 0] + rows['bbox'][:, 2] - 1) / 2.,
                                     (rows['bbox'][:, 1] + rows['bbox'][:, 3] - 1) / 2.])
    centroids = np.asarray(centroids, dtype=float).reshape(n, 2)
    rows['centroid_x'], rows['centroid_y'] = centroids[:, 0], centroids[:, 1]
    rows['kurtosis'] = 0. if kurtoses is None else kurtoses
    rows['eccentricity'] = 0. if eccentricities is None else eccentricities
    return rows


def an_old_index(levels=(1, 3, 4)) -> np.ndarray:
    """An index written before #125: no flux, centroid or kurtosis column."""
    rows = np.zeros(len(levels), dtype=[(name, INDEX_DTYPE[name])
                                        for name in INDEX_DTYPE.names
                                        if name not in COMPUTED_COLUMNS])
    rows['level'] = levels
    rows['bbox'] = np.tile([0, 0, 2, 2], (len(levels), 1))
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
    old = an_old_index()

    np.testing.assert_array_equal(big(old), [False, True, True])
    with pytest.raises(ValueError, match='this criterion reads flux'):
        bright(old)


def test_a_criterion_with_a_bug_of_its_own_is_not_blamed_on_the_store():
    """An old index does not make every ValueError the store's fault.

    The rewrite only fires when the failure names a column the index lacks;
    a broadcast that does not fit is the criterion's own bug, and telling the
    user to re-run the pipeline for it would be the worse error.
    """
    old = an_old_index()
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


# --- the six criteria that ship (#126, #17 point 4) ------------------------


def test_level_keeps_a_band_with_both_bounds_inclusive():
    index = an_index(levels=(1, 2, 3, 4))

    np.testing.assert_array_equal(Level(2)(index), [False, True, True, True])
    np.testing.assert_array_equal(Level(maximum=2)(index), [True, True, False, False])
    np.testing.assert_array_equal(Level(2, 3)(index), [False, True, True, False])


def test_a_level_cut_and_its_negation_are_the_cfis_pair():
    """``lvlo >= lvl_sep`` at :536 and ``lvlo < lvl_sep`` at :549, one object."""
    index = an_index(levels=(3, 4, 5, 6))

    icl, satellites = split(index, Level(5))

    assert list(icl['level']) == [5, 6]
    assert list(satellites['level']) == [3, 4]


@pytest.mark.parametrize('bounded', [
    Level, Kurtosis, Eccentricity, Size,
    lambda *a, **kw: Distance((0., 0.), *a, **kw),
])
def test_a_criterion_with_no_bound_at_all_is_refused(bounded):
    """A cut that keeps every atom is the one that looks like it worked."""
    with pytest.raises(ValueError, match='minimum, a maximum, or both'):
        bounded()


@pytest.mark.parametrize('bounded', [
    Level, Kurtosis, Eccentricity, Size,
    lambda *a, **kw: Distance((0., 0.), *a, **kw),
])
def test_a_criterion_with_its_bounds_the_wrong_way_round_is_refused(bounded):
    """An inverted band keeps nothing, and a Cascade would not say why."""
    with pytest.raises(ValueError, match='the wrong way round'):
        bounded(minimum=5., maximum=3.)


def test_a_distance_bound_cannot_be_given_positionally():
    """``Distance(centre, 30.)`` reads as a radius and would be a minimum."""
    with pytest.raises(TypeError):
        Distance((0., 0.), 30.)


def test_size_needs_both_bbox_sides_by_default_and_either_on_request():
    """The two CFIS papers disagreed, so the choice is the user's (#17)."""
    index = an_index(levels=(1, 2, 3),
                     bboxes=[(0, 0, 10, 10), (0, 0, 10, 2), (0, 0, 2, 2)])

    np.testing.assert_array_equal(Size(5)(index), [True, False, False])
    np.testing.assert_array_equal(Size(5, either=True)(index), [True, True, False])


def test_size_reads_the_two_bbox_sides_and_not_the_pixel_count():
    index = an_index(levels=(1,), bboxes=[(3, 7, 9, 12)])

    np.testing.assert_array_equal(Size(5, 6)(index), [True])
    np.testing.assert_array_equal(Size(6)(index), [False])


def test_distance_measures_from_the_atom_position():
    """CFIS's analysis radius, :521-523 — ``dR > R`` drops the atom."""
    index = an_index(levels=(1, 2, 3),
                     centroids=[(100., 100.), (103., 104.), (100., 130.)])

    np.testing.assert_array_equal(Distance((100., 100.), maximum=5.)(index),
                                  [True, True, False])
    np.testing.assert_array_equal(Distance((100., 100.), minimum=5.)(index),
                                  [False, True, True])


def test_atom_position_falls_back_to_the_bbox_centre():
    """An atom of no flux has no centroid; it still has a place (CONTEXT.md)."""
    index = an_index(levels=(1, 2), bboxes=[(0, 0, 4, 6), (10, 20, 12, 22)],
                     centroids=[(np.nan, np.nan), (11.25, 20.75)])

    x, y = atom_position(index)

    np.testing.assert_allclose(x, [1.5, 11.25])
    np.testing.assert_allclose(y, [2.5, 20.75])


def test_atom_position_uses_the_bbox_centre_throughout_on_request():
    index = an_index(levels=(1,), bboxes=[(10, 20, 12, 22)],
                     centroids=[(11.25, 20.75)])

    x, y = atom_position(index, centroid=False)

    np.testing.assert_allclose([x, y], [[10.5], [20.5]])


def test_in_mask_rounds_the_position_where_cfis_truncated():
    """:500 cast with ``int()``; a centroid of 7.8 sits in pixel 8, not 7."""
    mask = np.zeros((16, 16), dtype=bool)
    mask[8, 3] = True
    index = an_index(levels=(1,), bboxes=[(0, 0, 16, 16)], centroids=[(7.8, 2.6)])

    np.testing.assert_array_equal(InMask(mask)(index), [True])


def test_in_mask_does_not_claim_an_atom_whose_position_is_outside_it():
    """A mask smaller than the frame is not a crash and not a match."""
    mask = np.ones((4, 4), dtype=bool)
    index = an_index(levels=(1, 2), centroids=[(1., 1.), (40., 40.)])

    np.testing.assert_array_equal(InMask(mask)(index), [True, False])


def test_in_mask_negated_is_the_star_veto():
    """:526's ``mstar[xco, yco] != 1``, which is this criterion inverted."""
    stars = np.zeros((8, 8), dtype=bool)
    stars[1, 1] = True
    index = an_index(levels=(1, 2), centroids=[(1., 1.), (5., 5.)])

    np.testing.assert_array_equal((~InMask(stars))(index), [False, True])


def test_in_mask_takes_a_labelled_mask_as_well_as_a_boolean_one():
    labels = np.zeros((8, 8), dtype=int)
    labels[1, 1] = 7
    labels[2, 2] = -1
    index = an_index(levels=(1, 2, 3), centroids=[(1., 1.), (2., 2.), (5., 5.)])

    np.testing.assert_array_equal(InMask(labels)(index), [True, True, False])


def test_a_bbox_centre_between_two_pixels_rounds_the_same_way_every_time():
    """An even-sided box sits on a half-integer; parity must not decide.

    ``np.rint`` rounds half to even, so boxes of the same shape at origins 0,
    1 and 2 would land on pixels 0, 2 and 2 — and two identical atoms on the
    same mask edge would sort into different groups.
    """
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 1] = mask[2, 2] = mask[3, 3] = True
    index = an_index(levels=(1, 2, 3),
                     bboxes=[(0, 0, 2, 2), (1, 1, 3, 3), (2, 2, 4, 4)],
                     centroids=[(np.nan, np.nan)] * 3)

    x, y = atom_position(index)

    np.testing.assert_allclose(x, [0.5, 1.5, 2.5])
    np.testing.assert_array_equal(InMask(mask)(index), [True, True, True])


def test_a_composed_criterion_does_not_rewrite_the_store_message_twice():
    """The sentence is written once, however deep the composition goes."""
    composed = Level(1) & Kurtosis(0.)

    with pytest.raises(ValueError) as raised:
        composed(an_old_index())

    assert str(raised.value).count('this atom store does not carry') == 1


def test_a_mask_that_is_not_two_dimensional_is_refused():
    with pytest.raises(ValueError, match='2D mask'):
        InMask(np.zeros(8))


def test_kurtosis_keeps_the_atoms_the_cfis_artefact_cut_kept():
    """:505-507 calls negative excess kurtosis an artefact."""
    index = an_index(levels=(1, 2, 3), kurtoses=(-0.5, 0., 3.))

    np.testing.assert_array_equal(Kurtosis(0.)(index), [False, True, True])


def test_an_undefined_kurtosis_never_passes_a_cut():
    """NaN is 'this atom has none', not 'this atom is inside the range'."""
    index = an_index(levels=(1, 2), kurtoses=(np.nan, 1.))

    np.testing.assert_array_equal(Kurtosis(0.)(index), [False, True])
    np.testing.assert_array_equal(Kurtosis(maximum=0.)(index), [False, False])


def test_eccentricity_cuts_on_shape():
    index = an_index(levels=(1, 2, 3), eccentricities=(0.1, 0.5, 0.95))

    np.testing.assert_array_equal(Eccentricity(maximum=0.6)(index),
                                  [True, True, False])


def test_the_criteria_that_read_no_computed_column_run_on_an_old_store():
    """Level, size and eccentricity cuts still work on a pre-#125 store."""
    old = an_old_index(levels=(1, 3, 4))

    np.testing.assert_array_equal(Level(3)(old), [False, True, True])
    np.testing.assert_array_equal(Size(1)(old), [True, True, True])
    np.testing.assert_array_equal(Eccentricity(maximum=1.)(old), [True, True, True])


def test_a_position_criterion_on_an_old_store_raises_and_names_it():
    with pytest.raises(ValueError, match='this criterion reads centroid_x'):
        InMask(np.ones((8, 8), dtype=bool))(an_old_index())


def test_a_shipped_criterion_composes_with_the_algebra_and_with_a_def():
    """The point of making them criteria: they are not a separate species."""
    index = an_index(levels=(3, 5, 5), kurtoses=(1., -1., 1.))

    icl = Level(5) & Kurtosis(0.) & (lambda i: i['flux'] > 0.)

    np.testing.assert_array_equal(icl(index), [False, False, True])


# --- the units frame (#126, #17 point 6) -----------------------------------


def test_kpc_to_pix_converts_at_the_frames_redshift():
    """Pinned against Planck18: a cosmology change should fail here loudly."""
    frame = Units(pixel_scale=0.168, redshift=0.3)

    assert frame.kpc_to_pix(10.) == pytest.approx(12.9479036, rel=1e-6)
    assert frame.kpc_to_pix(20.) == pytest.approx(2 * 12.9479036, rel=1e-6)


def test_kpc_to_pix_honours_a_cosmology_the_user_gives():
    from astropy.cosmology import Planck15, Planck18

    same = Units(pixel_scale=0.168, redshift=0.3, cosmology=Planck18)
    other = Units(pixel_scale=0.168, redshift=0.3, cosmology=Planck15)

    assert same.kpc_to_pix(10.) == pytest.approx(12.9479036, rel=1e-6)
    assert other.kpc_to_pix(10.) != pytest.approx(same.kpc_to_pix(10.), rel=1e-6)


@pytest.mark.parametrize('frame, match', [
    (dict(pixel_scale=0.), 'pixel scale must be positive'),
    (dict(pixel_scale=-0.168), 'pixel scale must be positive'),
])
def test_a_units_frame_refuses_a_pixel_scale_that_is_not_a_scale(frame, match):
    with pytest.raises(ValueError, match=match):
        Units(**frame)


@pytest.mark.parametrize('redshift', [None, 0.])
def test_kpc_to_pix_without_a_redshift_raises_rather_than_returning_zero(redshift):
    with pytest.raises(ValueError, match='positive redshift'):
        Units(pixel_scale=0.168, redshift=redshift).kpc_to_pix(10.)


def test_a_size_in_kpc_is_the_same_cut_as_that_size_in_pixels():
    """The frame converts once, in the criterion; nothing else knows about kpc."""
    frame = Units(pixel_scale=0.168, redshift=0.3)
    index = an_index(levels=(1, 2), bboxes=[(0, 0, 20, 20), (0, 0, 5, 5)])

    np.testing.assert_array_equal(Size(10., units=frame)(index),
                                  Size(frame.kpc_to_pix(10.))(index))
    np.testing.assert_array_equal(Size(10., units=frame)(index), [True, False])


def test_a_distance_in_kpc_is_the_same_cut_as_that_distance_in_pixels():
    frame = Units(pixel_scale=0.168, redshift=0.3)
    index = an_index(levels=(1, 2), centroids=[(0., 5.), (0., 100.)])

    np.testing.assert_array_equal(Distance((0., 0.), maximum=10., units=frame)(index),
                                  [True, False])


def test_a_units_frame_records_the_cosmology_it_used():
    """A default that stays None is an assumption nobody can read back."""
    from astropy.cosmology import Planck18

    assert Units(pixel_scale=0.168).cosmology is Planck18
    assert 'Planck18' in repr(Units(pixel_scale=0.168))
