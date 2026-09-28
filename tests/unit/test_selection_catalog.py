"""Tier 0 — a selection's catalog and its two error methods (#128).

``to_table()`` gives one row per source model (#17, point 10). The flux floor is
an argument at the call site and moves the flux and its error together (point
12). Two error methods ship: a real bootstrap, the default, and the CFIS
borderline resampling, which is not one (point 11).
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

from dawis.io import AtomStore, read_atom_index  # noqa: E402
from dawis.selection import (  # noqa: E402
    Level,
    MissingColumn,
    SourceModel,
    bootstrap,
    mask_sort,
    resample_borderline,
    source_models,
    to_table,
)

#: Levels 2, 2, 4 and 3; the level 3 atom is far brighter than the rest.
ATOMS = {1: [an_atom(0.), an_atom(100., (3, 4, 9, 9))],
         2: [an_atom(200., (0, 0, 1, 5), level=4), an_atom(5000., (20, 20, 22, 23), level=3)]}


@pytest.fixture
def outpath(tmp_path):
    outpath = str(tmp_path / 'run')
    with AtomStore(outpath + '.atoms.hdf5', SHAPE) as store:
        for iteration, atoms in ATOMS.items():
            store.append(iteration, atoms, [a_tree() for _ in atoms])
    return outpath


def model(outpath, criterion) -> SourceModel:
    index = read_atom_index(outpath)
    return SourceModel('m', index[criterion(index)], outpath)


def test_flux_is_the_sum_of_the_image(outpath):
    small = model(outpath, Level(2))

    assert small.flux() == pytest.approx(small.image().sum(), rel=1e-12)


def test_the_floor_drops_the_pixels_below_it(outpath):
    small = model(outpath, Level(2))
    image = small.image()

    assert small.flux(floor=110.) == pytest.approx(image[image >= 110.].sum(), rel=1e-12)
    assert small.flux(floor=110.) < small.flux()


def test_the_table_has_one_row_per_source_model(outpath):
    index = read_atom_index(outpath)
    groups = mask_sort(index, {'top': np.pad(np.ones((10, 32), bool), ((0, 22), (0, 0)))})
    models = source_models(groups, outpath)

    table = to_table(models, n=10, rng=0)

    assert list(table.colnames) == ['name', 'n_atoms', 'flux', 'flux_err', 'det_err',
                                    'centroid_x', 'centroid_y', 'bbox', 'level_min',
                                    'level_max']
    assert list(table['name']) == ['top'], 'unclassified atoms get no row'
    row = table[0]
    top = models['top']
    image = top.image()
    x, y = np.indices(SHAPE)
    assert row['n_atoms'] == 3
    assert row['flux'] == pytest.approx(image.sum(), rel=1e-12)
    assert row['centroid_x'] == pytest.approx((x * image).sum() / image.sum(), rel=1e-12)
    assert row['centroid_y'] == pytest.approx((y * image).sum() / image.sum(), rel=1e-12)
    assert list(row['bbox']) == [0, 0, 9, 9]
    assert (row['level_min'], row['level_max']) == (2, 4)


def test_det_err_adds_the_det_err_image_in_quadrature(outpath):
    small = model(outpath, Level(2))

    row = to_table([small], n=2, rng=0)[0]

    assert row['det_err'] == pytest.approx(np.sqrt((small.det_err_image() ** 2).sum()))


def test_the_floor_is_recorded_on_the_table_not_on_the_model(outpath):
    small = model(outpath, Level(2))

    table = to_table([small], n=2, floor=110., rng=0)

    assert table[0]['flux'] == small.flux(floor=110.)
    assert table.meta['floor'] == 110.
    assert small.flux() > small.flux(floor=110.), 'the model keeps no floor'


def test_an_empty_source_model_is_a_row_of_nothing(outpath):
    empty = model(outpath, Level(9))

    row = to_table([empty], n=2, rng=0)[0]

    assert (row['n_atoms'], row['flux'], row['flux_err'], row['det_err']) == (0, 0., 0., 0.)
    assert np.isnan(row['centroid_x']) and np.isnan(row['centroid_y'])
    assert list(row['bbox']) == [-1] * 4
    assert (row['level_min'], row['level_max']) == (-1, -1)


def test_a_one_atom_model_has_no_bootstrap_error(outpath):
    assert bootstrap(model(outpath, Level(4)), n=20, rng=0) == 0.
    assert bootstrap(model(outpath, Level(4)), n=20, floor=0., rng=0) == 0.


def test_the_bootstrap_draws_with_replacement(outpath):
    everything = model(outpath, lambda index: np.ones(len(index), bool))
    fluxes = everything.rows['flux']

    # The standard error of a sum of N draws with replacement.
    expected = np.sqrt(len(fluxes)) * fluxes.std()

    assert bootstrap(everything, n=4000, rng=1) == pytest.approx(expected, rel=0.05)


def test_the_floor_path_draws_what_the_index_path_draws(outpath):
    everything = model(outpath, lambda index: np.ones(len(index), bool))

    assert bootstrap(everything, n=30, floor=-np.inf, rng=2) == pytest.approx(
        bootstrap(everything, n=30, rng=2), rel=1e-12)


def test_no_spread_resamples_the_whole_group_and_nothing_else(outpath):
    small = model(outpath, Level(2))
    borderline = model(outpath, Level(3)).rows

    assert resample_borderline(small, borderline, n=20, spread=0., rng=0) == \
        pytest.approx((small.flux(),) * 3, rel=1e-12)


def test_borderline_atoms_are_swapped_in(outpath):
    index = read_atom_index(outpath)
    group = SourceModel('m', index[index['level'] != 3], outpath)
    bright = index[Level(3)(index)]

    median, low, high = resample_borderline(group, bright, n=400, spread=1., rng=3)

    assert low <= median <= high
    assert low < group.flux(), 'some draws hold fewer atoms'
    assert high > bright['flux'].sum(), 'some draws swap the bright borderline atom in'


def test_a_store_without_the_computed_columns_is_refused_by_name(outpath):
    index = read_atom_index(outpath)
    old = index[[n for n in index.dtype.names if n not in ('flux', 'centroid_x')]]

    with pytest.raises(MissingColumn, match='flux, centroid_x'):
        to_table([SourceModel('m', old, outpath)], n=2)


def test_the_bootstrap_needs_two_resamplings(outpath):
    with pytest.raises(ValueError, match='n >= 2'):
        bootstrap(model(outpath, Level(2)), n=1)


def test_the_spread_lies_between_zero_and_one(outpath):
    small = model(outpath, Level(2))

    with pytest.raises(ValueError, match='spread'):
        resample_borderline(small, small.rows[:0], spread=1.5)


def test_a_borderline_atom_already_in_the_model_warns(outpath):
    small = model(outpath, Level(2))

    with pytest.warns(UserWarning, match='1 borderline atom is also in the source model'):
        resample_borderline(small, small.rows[:1], n=2, rng=0)

