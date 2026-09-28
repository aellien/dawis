"""Tier 0 — source models and the images they synthesise (#127).

A source model is a name, its index rows and the run's path, so it is a
complete thing on its own and reads pixels only when an image is asked for
(#17, point 7). Its image and its ``det_err`` image are the sums of its atoms,
the same sums :class:`~dawis.restoration.Reconstruction` accumulates during
the run (point 8). Unclassified atoms synthesise to an image and are no source
model (point 9). And assembling source models from hand-built groups warns if
an atom is in two of them (point 3).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from test_dataclasses import a_tree  # noqa: E402
from test_store import SHAPE, an_atom  # noqa: E402

from dawis.io import AtomStore, read_atom_index, read_synthesised_image  # noqa: E402
from dawis.restoration import Reconstruction  # noqa: E402
from dawis.selection import Level, SourceModel, mask_sort, source_models  # noqa: E402

#: Two iterations, two atoms overlapping in the first, levels 2, 2 and 4.
ATOMS = {1: [an_atom(0.), an_atom(100., (3, 4, 9, 9))],
         2: [an_atom(200., (0, 0, 1, 5), level=4)]}


@pytest.fixture
def outpath(tmp_path):
    outpath = str(tmp_path / 'run')
    with AtomStore(outpath + '.atoms.hdf5', SHAPE) as store:
        for iteration, atoms in ATOMS.items():
            store.append(iteration, atoms, [a_tree() for _ in atoms])
    return outpath


def summed(atoms) -> Reconstruction:
    run = Reconstruction(np.zeros(SHAPE), n_levels=5)
    run.add(atoms)
    return run


def test_a_source_model_synthesises_what_the_run_accumulated(outpath):
    index = read_atom_index(outpath)
    model = SourceModel('small', index[index['level'] == 2], outpath)

    expected = summed(ATOMS[1])

    np.testing.assert_array_equal(model.image(), expected.restored)
    np.testing.assert_array_equal(model.det_err_image(), expected.det_err)


def test_every_group_adds_back_to_the_full_field(outpath):
    groups = mask_sort(read_atom_index(outpath), {'corner': np.eye(*SHAPE, dtype=bool)})
    models = source_models(groups, outpath)

    total = sum(m.image() for m in models.values())
    total = total + read_synthesised_image(outpath, groups['unclassified'])

    np.testing.assert_array_equal(total, summed(ATOMS[1] + ATOMS[2]).restored)


def test_unclassified_atoms_are_no_source_model(outpath):
    groups = mask_sort(read_atom_index(outpath), {'nothing': np.zeros(SHAPE, dtype=bool)})

    models = source_models(groups, outpath)

    assert list(models) == ['nothing']
    assert models['nothing'].name == 'nothing'
    assert models['nothing'].image().shape == SHAPE
    assert not models['nothing'].image().any()


def test_a_cascade_partition_assembles_without_a_warning(outpath):
    index = read_atom_index(outpath)
    groups = mask_sort(index, {'all': np.ones(SHAPE, dtype=bool)})

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        source_models(groups, outpath)


def test_an_atom_in_two_hand_built_groups_warns(outpath):
    index = read_atom_index(outpath)
    groups = {'big': index[Level(2)(index)], 'deep': index[Level(4)(index)]}

    with pytest.warns(UserWarning, match='1 atom is found more than once'):
        models = source_models(groups, outpath)

    assert list(models) == ['big', 'deep'], 'a warning, not a refusal'


def test_an_atom_twice_in_one_group_warns(outpath):
    index = read_atom_index(outpath)

    with pytest.warns(UserWarning, match='2 atoms are found more than once'):
        source_models({'twice': np.concatenate([index[:2], index[:2]])}, outpath)
