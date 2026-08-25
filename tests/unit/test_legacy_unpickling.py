"""Tier 0 — pickles written before the carve still read back (#50).

A pickle records the *module path* of every class it holds, so a run written
before the carve names ``dawis.restore_objects.restored_object``,
``dawis.make_interscale_trees.interscale_tree``, ``dawis.make_regions.region``
and the cube classes under ``dawis.datacube``. Those paths resolved only
because the flat shims existed. This ticket deletes them, and the published
``v0.0.7-h`` outputs are recorded in exactly that format — the regression
harness (#3) reads its goldens through :func:`read_ol_from_pickle`, and old
runs resume through the same call. So :mod:`dawis.io.legacy` has to map the
old paths onto the carved classes itself.

Nothing in the tree can be tested against a real pre-carve file, because none
is checked in. Instead the classes are pickled *under their old paths* — the
one thing a shim used to make possible — and read back with the shims gone.
"""

from __future__ import annotations

import contextlib
import importlib
import pickle
import sys
import types

import numpy as np
import pytest

import dawis.io.legacy as legacy
from dawis.cubes import (
    coarse_datacube,
    datacube,
    label_datacube,
    support_datacube,
    wavelet_datacube,
)
from dawis.detection import region
from dawis.io.legacy import (
    read_itl_from_pickle,
    read_ol_from_pickle,
    read_regions_from_pickle,
    write_itl_to_pickle,
    write_ol_to_pickle,
    write_regions_to_pickle,
)
from dawis.restoration import restored_object
from dawis.trees import interscale_tree

#: Every class a pre-carve run can have written into a pickle.
PICKLED_CLASSES = (
    restored_object,
    interscale_tree,
    region,
    datacube,
    wavelet_datacube,
    coarse_datacube,
    support_datacube,
    label_datacube,
)

#: Where each of them used to live, read *off the map the reader uses* rather
#: than written out again here. Restating it would let the two drift, and the
#: test would then pin a mapping nothing in the package performs.
PRE_CARVE_PATHS = {
    cls: {new: old for old, new in legacy._PRE_CARVE_MODULES.items()}[
        cls.__module__
    ]
    for cls in PICKLED_CLASSES
}


def test_every_mapped_module_is_exercised():
    """No entry in the reader's map goes unchecked.

    The map is the only thing standing between a pre-carve file and a
    ``ModuleNotFoundError``, so an entry no case below reaches is an entry
    that could be wrong without any test noticing.
    """
    assert set(PRE_CARVE_PATHS.values()) == set(legacy._PRE_CARVE_MODULES)


@contextlib.contextmanager
def written_before_the_carve():
    """Make :mod:`pickle` record the pre-carve module path of every class.

    Pickle names a class by ``__module__`` and ``__qualname__``, so pointing
    ``__module__`` back at the flat module — and putting a stand-in for that
    module in :data:`sys.modules`, which pickle checks the class against —
    produces the bytes a pre-carve run really wrote.
    """

    originals = {cls: cls.__module__ for cls in PRE_CARVE_PATHS}
    for old_path in sorted(set(PRE_CARVE_PATHS.values())):
        assert old_path not in sys.modules, (
            '%s is still importable, so this test would pass through the shim '
            'rather than through the mapping that replaces it' % old_path
        )
        sys.modules[old_path] = types.ModuleType(old_path)
    for cls, old_path in PRE_CARVE_PATHS.items():
        setattr(sys.modules[old_path], cls.__name__, cls)
        cls.__module__ = old_path
    try:
        yield
    finally:
        for cls, module_name in originals.items():
            cls.__module__ = module_name
        for old_path in set(PRE_CARVE_PATHS.values()):
            sys.modules.pop(old_path, None)


def a_region(label=1, level=2):
    """One measured detection, with the properties a tree reads off it."""
    return region(
        label=label,
        level=level,
        bbox=(0, 0, 4, 4),
        eccentricity=0.5,
        area=16,
    )


def an_atom():
    """One synthesised atom."""
    return restored_object(
        image=np.ones((4, 4)),
        det_err_image=np.zeros((4, 4)),
        bbox=(0, 0, 4, 4),
        level=2,
        eccentricity=0.5,
        filter_kw='BSPL',
        flag_convergence=True,
        sum_wr=1.0,
        norm_wr=2.0,
    )


def a_tree():
    """One interscale tree, built attribute by attribute as the store does."""
    tree = interscale_tree(blank=True)
    tree.interscale_maximum = a_region()
    tree.region_list = [a_region()]
    tree.bbox = (0, 0, 4, 4)
    tree.det_err_array = np.zeros(3)
    return tree


def test_the_flat_modules_are_gone():
    """The premise of every case below: no shim is left to resolve the path."""
    for old_path in sorted(set(PRE_CARVE_PATHS.values())):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(old_path)


def test_atoms_written_before_the_carve_read_back_as_the_carved_class(tmp_path):
    """``dawis.restore_objects.restored_object`` still resolves."""
    path = tmp_path / 'ol.pkl'
    with written_before_the_carve():
        write_ol_to_pickle([an_atom()], str(path))

    assert b'dawis.restore_objects' in path.read_bytes()

    [atom] = read_ol_from_pickle(str(path))

    assert type(atom) is restored_object
    assert atom.filter == 'BSPL'
    np.testing.assert_array_equal(atom.image, np.ones((4, 4)))


def test_trees_written_before_the_carve_read_back_as_the_carved_classes(tmp_path):
    """A tree names two moved classes: its own, and the regions inside it."""
    path = tmp_path / 'itl.pkl'
    with written_before_the_carve():
        write_itl_to_pickle([a_tree()], str(path))

    assert b'dawis.make_interscale_trees' in path.read_bytes()
    assert b'dawis.make_regions' in path.read_bytes()

    [tree] = read_itl_from_pickle(str(path))

    assert type(tree) is interscale_tree
    assert type(tree.interscale_maximum) is region
    assert [type(member) for member in tree.region_list] == [region]


def test_regions_written_before_the_carve_read_back_as_the_carved_class(tmp_path):
    """``dawis.make_regions.region`` still resolves."""
    path = tmp_path / 'regions.pkl'
    with written_before_the_carve():
        write_regions_to_pickle([a_region(label=7)], str(path))

    assert b'dawis.make_regions' in path.read_bytes()

    [measured] = read_regions_from_pickle(str(path))

    assert type(measured) is region
    assert measured.label == 7


@pytest.mark.parametrize(
    'cube',
    [
        datacube(np.zeros((4, 4, 3)), dctype='NONE'),
        wavelet_datacube(np.zeros((4, 4, 3)), 'BSPL'),
        coarse_datacube(np.zeros((4, 4, 3)), 'BSPL'),
        support_datacube(np.zeros((4, 4, 3)), 'HARD', np.zeros(3)),
        label_datacube(np.zeros((4, 4, 3)), np.zeros(3)),
    ],
    ids=['base', 'wavelet', 'coarse', 'support', 'label'],
)
def test_cubes_written_before_the_carve_read_back_as_the_carved_class(
    tmp_path, cube
):
    """The cube classes moved too, and a pre-carve pickle can hold one."""
    expected = type(cube)
    path = tmp_path / 'cube.pkl'
    with written_before_the_carve():
        with open(path, 'wb') as f:
            pickle.dump(cube, f)

    assert b'dawis.datacube' in path.read_bytes()

    with open(path, 'rb') as f:
        read = legacy._load_legacy(f)

    assert type(read) is expected


def test_a_class_that_never_moved_is_left_alone(tmp_path):
    """The mapping renames only what the carve moved."""
    path = tmp_path / 'plain.pkl'
    with open(path, 'wb') as f:
        pickle.dump({'array': np.arange(3)}, f)

    with open(path, 'rb') as f:
        read = legacy._load_legacy(f)

    np.testing.assert_array_equal(read['array'], np.arange(3))
