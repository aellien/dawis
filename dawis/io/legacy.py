"""The legacy pickle format.

The format DAWIS wrote before the HDF5 object store, and the one the
published ``v0.0.7-h`` outputs are recorded in. It is kept readable, not
recommended: the regression harness (#3) reads its goldens through
:func:`read_ol_from_pickle`, and old runs resume through the same call.

This module is the leaf of :mod:`dawis.io`. It knows about pickle and
nothing else — no h5py, no astropy — so :mod:`dawis.io.store` can import it
to read a legacy run without the edge ever pointing back.

A pickle records the *module path and class name* of every class it holds, so
a run written before the carve names ``dawis.restore_objects.restored_object``,
``dawis.make_interscale_trees.interscale_tree``, ``dawis.make_regions.region``
and the cube classes under ``dawis.datacube``. Those paths resolved only
because the flat shims existed, and the contract ticket (#50) deleted them; the
rename sweep (#51) and the Phase A gate (#59) then changed the class names
themselves. So this module maps
both halves itself, in :class:`_LegacyUnpickler`. Without the module map a
pre-carve file raises ``ModuleNotFoundError``; without the class map it raises
``AttributeError`` — either way the published ``v0.0.7-h`` outputs stop being
readable, which is the one thing this module exists to prevent.

The maps are a *read* concession only: what this module writes is written under
the carved paths and the current names, so a file written today needs no entry.

.. warning::

   Unpickling executes whatever the file says. Read only files you produced.
"""

from __future__ import annotations

import gc
import pickle
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:  # annotations only, so no runtime edge to the other blocks
    from dawis.detection import region
    from dawis.restoration import Atom
    from dawis.trees import InterscaleTree


#: Where the classes a pre-carve pickle names used to live, and where each
#: was carved to. Keyed by module rather than by class because a module moved
#: as a whole: every class the old module held is now in the new one.
_PRE_CARVE_MODULES = {
    'dawis.restore_objects': 'dawis.restoration.synthesis',
    'dawis.make_interscale_trees': 'dawis.trees.construction',
    'dawis.make_regions': 'dawis.detection.regions',
    'dawis.datacube': 'dawis.cubes',
}

#: What the classes the two rename passes touched used to be called, keyed by
#: the module they live in *now* and their old name. Keying on the carved
#: module rather than the flat one means one table covers both a pre-carve file
#: (whose module is rewritten by :data:`_PRE_CARVE_MODULES` first) and a file
#: written between the carve and a rename. The sweep of #51 renamed the cube
#: family and ``restored_object``; the Phase A gate (#59) then renamed
#: ``interscale_tree``, the one class #4 asked for that #51 left out. ``region``
#: is absent because neither pass touched its name.
_PRE_RENAME_CLASSES = {
    ('dawis.restoration.synthesis', 'restored_object'): 'Atom',
    ('dawis.trees.construction', 'interscale_tree'): 'InterscaleTree',
    ('dawis.cubes', 'datacube'): 'Cube',
    ('dawis.cubes', 'wavelet_datacube'): 'WaveletCube',
    ('dawis.cubes', 'coarse_datacube'): 'CoarseCube',
    ('dawis.cubes', 'support_datacube'): 'SupportCube',
    ('dawis.cubes', 'label_datacube'): 'LabelCube',
}


class _LegacyUnpickler(pickle.Unpickler):
    """An unpickler that reads the module paths and names of the flat package.

    :meth:`pickle.Unpickler.find_class` is the single point every class in a
    pickle stream is resolved through, so rewriting the four moved modules and
    the seven renamed classes there covers atoms, trees, regions and cubes at any
    depth of nesting. The module is rewritten first and the name is looked up
    against the result, so the two maps compose for a file that predates both
    changes. A module or name outside the maps is passed through untouched, so
    numpy and the standard library resolve exactly as they did.
    """

    def find_class(self, module: str, name: str) -> type:
        module = _PRE_CARVE_MODULES.get(module, module)
        name = _PRE_RENAME_CLASSES.get((module, name), name)
        return super().find_class(module, name)


def _load_legacy(infile: BinaryIO) -> object:
    """Read one pickle stream, resolving pre-carve module paths.

    The one door every reader in this module goes through, so that a format
    concession is made in a single place rather than three.
    """

    return _LegacyUnpickler(infile).load()


def write_ol_to_pickle(
    object_list: list[Atom],
    filename: str,
    overwrite: bool = True,
) -> None:
    """Write a list of atoms to a pickle file.

    Parameters
    ----------
    object_list : list of Atom
        The atoms of one iteration.
    filename : str
        Path to write to.
    overwrite : bool, optional
        Replace the file. Default is True; False appends a second pickle
        stream to the existing file, which :func:`read_ol_from_pickle` will
        not read back.
    """

    if overwrite == True:
        mode = 'wb'
    else:
        mode = 'ab'

    with open(filename, mode) as outfile:
        gc.collect()
        pickle.dump(object_list, outfile)


def read_ol_from_pickle(filename: str) -> list[Atom]:
    """Read a list of atoms back from a pickle file.

    Parameters
    ----------
    filename : str
        Path written by :func:`write_ol_to_pickle`.

    Returns
    -------
    list of Atom
        The atoms, in the order they were written.

    Notes
    -----
    Reads one pickle stream, so a file appended to holds atoms this does not
    return.

    Unpickling executes whatever the file says. Read only files you produced.
    """

    with open(filename, 'rb') as infile:
        gc.collect()
        object_list = _load_legacy(infile)

    return object_list


def write_itl_to_pickle(
    interscale_tree_list: list[InterscaleTree],
    filename: str,
    overwrite: bool = True,
) -> None:
    """Write a list of interscale trees to a pickle file.

    Parameters
    ----------
    interscale_tree_list : list of InterscaleTree
        The trees of one iteration.
    filename : str
        Path to write to.
    overwrite : bool, optional
        Replace the file. Default is True; False appends a second pickle
        stream to the existing file, which :func:`read_itl_from_pickle` will
        not read back.
    """

    if overwrite == True:
        mode = 'wb'
    else:
        mode = 'ab'

    with open(filename, mode) as outfile:
        gc.collect()
        pickle.dump(interscale_tree_list, outfile)


def read_itl_from_pickle(filename: str) -> list[InterscaleTree]:
    """Read a list of interscale trees back from a pickle file.

    Parameters
    ----------
    filename : str
        Path written by :func:`write_itl_to_pickle`.

    Returns
    -------
    list of InterscaleTree
        The trees, in the order they were written.

    Notes
    -----
    Reads one pickle stream, so a file appended to holds trees this does not
    return.

    Unpickling executes whatever the file says. Read only files you produced.
    """

    with open(filename, 'rb') as infile:
        gc.collect()
        interscale_tree_list = _load_legacy(infile)

    return interscale_tree_list


def write_regions_to_pickle(
    region_list: list[region],
    filename: str,
    overwrite: bool = True,
) -> None:
    """Write a region list to a pickle file.

    Parameters
    ----------
    region_list : list of region
        The regions to write.
    filename : str
        Path to write to.
    overwrite : bool, optional
        Replace the file. Default is True; False appends a second pickle
        stream to the existing file, which
        :func:`read_regions_from_pickle` will not read back.
    """

    if overwrite == True:
        mode = 'wb'
    else:
        mode = 'ab'

    with open(filename, mode) as outfile:
        pickle.dump(region_list, outfile)

def read_regions_from_pickle(filename: str) -> list[region]:
    """Read a region list back from a pickle file.

    Parameters
    ----------
    filename : str
        Path written by :func:`write_regions_to_pickle`.

    Returns
    -------
    list of region
        The regions, in the order they were written.

    Notes
    -----
    Reads one pickle stream, so a file appended to holds regions this does
    not return.

    Unpickling executes whatever the file says. Read only files you produced.
    """

    with open(filename, 'rb') as infile:
        region_list = _load_legacy(infile)

    return region_list
