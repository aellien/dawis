"""Tier 0 — pickles written before the carve still read back (#50, #51).

A pickle records the *module path and class name* of every class it holds, so
a run written before the carve names ``dawis.restore_objects.restored_object``,
``dawis.make_interscale_trees.interscale_tree``, ``dawis.make_regions.region``
and the cube classes under ``dawis.datacube``. Those paths resolved only
because the flat shims existed. #50 deleted them, and #51 and #59 renamed the classes,
and the published ``v0.0.7-h`` outputs are recorded in exactly the old format —
the regression harness (#3) reads its goldens through
:func:`read_ol_from_pickle`, and old runs resume through the same call. So
:mod:`dawis.io.legacy` has to map both the old paths and the old names onto the
carved classes itself.

Nothing in the tree can be tested against a real pre-carve file, because none
is checked in. Instead the classes are pickled *under their old paths and old
names* — the bytes a pre-carve run really wrote — and read back with the shims
gone and the names changed.
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
    CoarseCube,
    Cube,
    LabelCube,
    SupportCube,
    WaveletCube,
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
from dawis.restoration import Atom
from dawis.trees import InterscaleTree

#: Every class a pre-carve run can have written into a pickle.
PICKLED_CLASSES = (
    Atom,
    InterscaleTree,
    region,
    Cube,
    WaveletCube,
    CoarseCube,
    SupportCube,
    LabelCube,
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


#: What each of them used to be called, read off the reader's own map for the
#: same reason. A class #51 left alone maps to the name it already has, so the
#: forging below can treat every class the same way.
_OLD_NAME_OF = {
    (module, new): old
    for (module, old), new in legacy._PRE_RENAME_CLASSES.items()
}

PRE_RENAME_NAMES = {
    cls: _OLD_NAME_OF.get((cls.__module__, cls.__name__), cls.__name__)
    for cls in PICKLED_CLASSES
}


def test_every_mapped_module_is_exercised():
    """No entry in the reader's map goes unchecked.

    The map is the only thing standing between a pre-carve file and a
    ``ModuleNotFoundError``, so an entry no case below reaches is an entry
    that could be wrong without any test noticing.
    """
    assert set(PRE_CARVE_PATHS.values()) == set(legacy._PRE_CARVE_MODULES)


def test_every_renamed_class_is_exercised():
    """Likewise for the name map, which stands between a file and an
    ``AttributeError``.
    """
    exercised = {
        (cls.__module__, old_name)
        for cls, old_name in PRE_RENAME_NAMES.items()
        if old_name != cls.__name__
    }
    assert exercised == set(legacy._PRE_RENAME_CLASSES)


@contextlib.contextmanager
def written_before_the_carve():
    """Make :mod:`pickle` record the pre-carve path *and name* of every class.

    Pickle names a class by ``__module__`` and ``__qualname__``, so pointing
    both back at the flat module and the old class name — and putting a
    stand-in for that module in :data:`sys.modules`, which pickle checks the
    class against — produces the bytes a pre-carve run really wrote. Forging
    the module alone would leave the current name in the stream, and the name
    map would then go untested by every case below.
    """

    originals = {
        cls: (cls.__module__, cls.__qualname__) for cls in PRE_CARVE_PATHS
    }
    for old_path in sorted(set(PRE_CARVE_PATHS.values())):
        assert old_path not in sys.modules, (
            '%s is still importable, so this test would pass through the shim '
            'rather than through the mapping that replaces it' % old_path
        )
        sys.modules[old_path] = types.ModuleType(old_path)
    for cls, old_path in PRE_CARVE_PATHS.items():
        old_name = PRE_RENAME_NAMES[cls]
        setattr(sys.modules[old_path], old_name, cls)
        cls.__module__ = old_path
        cls.__qualname__ = old_name
    try:
        yield
    finally:
        for cls, (module_name, qualname) in originals.items():
            cls.__module__ = module_name
            cls.__qualname__ = qualname
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
    return Atom(
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
    tree = InterscaleTree(blank=True)
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

    assert type(atom) is Atom
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

    assert type(tree) is InterscaleTree
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
        Cube(np.zeros((4, 4, 3)), dctype='NONE'),
        WaveletCube(np.zeros((4, 4, 3)), 'BSPL'),
        CoarseCube(np.zeros((4, 4, 3)), 'BSPL'),
        SupportCube(np.zeros((4, 4, 3)), 'HARD', np.zeros(3)),
        LabelCube(np.zeros((4, 4, 3)), np.zeros(3)),
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
