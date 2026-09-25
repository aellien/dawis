"""Tier 0: the cube family, the interscale tree and the atom are dataclasses.

#4 decided every 1.0 class is a ``@dataclass`` (#64 converted the seven that
were moved rather than written). Two things a generated class would change
by default are pinned here as *not* changed: equality stays identity, because
every one of these holds arrays, and the HDF5 store still rebuilds a tree and
an atom from nothing but their stored attributes.
"""

from __future__ import annotations

import dataclasses

import h5py
import numpy as np
import pytest

from dawis.cubes import CoarseCube, Cube, LabelCube, SupportCube, WaveletCube
from dawis.detection import region
from dawis.io.store import (
    read_itl_from_hdf5,
    read_ol_from_hdf5,
    write_itl_to_hdf5,
    write_ol_to_hdf5,
)
from dawis.restoration import Atom
from dawis.restoration.synthesis import UNKNOWN_PROVENANCE
from dawis.trees import InterscaleTree

CONVERTED = (Cube, WaveletCube, CoarseCube, SupportCube, LabelCube,
         InterscaleTree, Atom)


def a_tree() -> InterscaleTree:
    """A three-scale tree on a 32 by 32 frame, built the way deblending does."""
    shape, n_levels = (32, 32), 3
    labels = np.zeros(shape + (n_levels,))
    regions = []
    for level, (low, high) in enumerate(((13, 19), (11, 21), (9, 23))):
        coords = np.array([(x, y) for x in range(low, high)
                           for y in range(low, high)])
        labels[low:high, low:high, level] = 1
        regions.append(region(label = 1, level = level, coords = coords,
                              x_max = 16, y_max = 16, eccentricity = 0.1,
                              area = coords.shape[0],
                              bbox = (low, low, high, high)))

    return InterscaleTree(regions[-1], regions,
                          WaveletCube(np.ones(shape + (n_levels,)), 'BSPL'),
                          LabelCube(labels, [1, 1, 1]),
                          np.zeros(n_levels), 16)


def an_atom() -> Atom:
    return Atom(np.ones((4, 4)), np.zeros((4, 4)), [0, 0, 4, 4], 2, 0.5,
                'BSPL', True, 1.0, 2.0, 'bijaoui_rue', 'SUM')


@pytest.mark.parametrize('cls', CONVERTED, ids = lambda cls: cls.__name__)
def test_the_class_is_a_dataclass(cls):
    assert dataclasses.is_dataclass(cls)


def test_equality_is_identity_and_instances_stay_hashable():
    """A generated ``__eq__`` would compare arrays and raise."""
    first = WaveletCube(np.zeros((4, 4, 3)), 'BSPL')
    second = WaveletCube(np.zeros((4, 4, 3)), 'BSPL')

    assert first == first and first != second
    assert len({first, second, an_atom(), an_atom()}) == 4


def test_the_constructors_keep_their_coercions():
    atom = an_atom()
    assert atom.bbox == (0, 0, 4, 4)

    cube = SupportCube(np.zeros((8, 6, 4)), 'HARD', np.zeros(3))
    assert (cube.dctype, cube.shape, cube.z_size) == ('SUPPORT', (8, 6, 4), 4)
    assert cube.fheader['TRSHTYPE'] == 'HARD'


def test_the_tree_keeps_its_geometry_and_drops_the_cubes():
    tree = a_tree()

    assert (tree.n_regions, tree.span_levels, tree.bbox) == (3, 3, (2, 2, 32, 32))
    assert not hasattr(tree, 'wavelet_cube')
    assert not hasattr(tree, 'label_cube')


def test_the_hdf5_store_round_trips_a_tree_and_an_atom(tmp_path):
    tree, atom = a_tree(), an_atom()
    write_itl_to_hdf5([tree], tmp_path / 'itl.hdf5')
    write_ol_to_hdf5([atom], tmp_path / 'ol.hdf5')

    [tree_back] = read_itl_from_hdf5(tmp_path / 'itl.hdf5')
    [atom_back] = read_ol_from_hdf5(tmp_path / 'ol.hdf5')

    assert type(tree_back) is InterscaleTree
    assert tuple(tree_back.bbox) == tree.bbox
    assert tree_back.extent == tree.extent
    assert [r.level for r in tree_back.region_list] == [0, 1, 2]

    assert type(atom_back) is Atom
    np.testing.assert_array_equal(atom_back.image, atom.image)
    assert (atom_back.filter, atom_back.algorithm) == ('BSPL', 'bijaoui_rue')


def test_an_atom_written_before_provenance_reads_unknown(tmp_path):
    """#66: no ``algorithm`` dataset falls through to the class default."""
    write_ol_to_hdf5([an_atom()], tmp_path / 'ol.hdf5')
    with h5py.File(tmp_path / 'ol.hdf5', 'a') as f:
        del f['o0/algorithm'], f['o0/synthesis_operator']

    [atom] = read_ol_from_hdf5(tmp_path / 'ol.hdf5')

    assert atom.algorithm == atom.synthesis_operator == UNKNOWN_PROVENANCE


def test_a_support_cube_reads_back_from_fits(tmp_path):
    """The file holds no noise sample, so the one read back has none."""
    SupportCube(np.ones((4, 4, 3)), 'HARD', np.zeros(3)).to_fits(tmp_path / 's.fits')

    cube = SupportCube.from_fits(tmp_path / 's.fits')

    assert (cube.threshold_type, cube.noise_pixels) == ('HARD', None)
