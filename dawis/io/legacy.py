"""The legacy pickle format.

The format DAWIS wrote before the HDF5 object store, and the one the
published ``v0.0.7-h`` outputs are recorded in. It is kept readable, not
recommended: the regression harness (#3) reads its goldens through
:func:`read_ol_from_pickle`, and old runs resume through the same call.

This module is the leaf of :mod:`dawis.io`. It knows about pickle and
nothing else — no h5py, no astropy — so :mod:`dawis.io.store` can import it
to read a legacy run without the edge ever pointing back.

A pickle records the *module path* of every class it holds, so a run written
before the carve names ``dawis.restore_objects.restored_object``,
``dawis.make_interscale_trees.interscale_tree``, ``dawis.make_regions.region``
and the cube classes under ``dawis.datacube``. Those paths resolved only
because the flat shims existed, and the contract ticket (#50) deleted them —
so this module maps them itself, in :class:`_LegacyUnpickler`. Without that
map a pre-carve file raises ``ModuleNotFoundError`` and the published
``v0.0.7-h`` outputs stop being readable, which is the one thing this module
exists to prevent.

The map is a *read* concession only: what this module writes is written under
the carved paths, so a file written today needs no entry.

.. warning::

   Unpickling executes whatever the file says. Read only files you produced.
"""

from __future__ import annotations

import gc
import pickle
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:  # annotations only, so no runtime edge to the other blocks
    from dawis.detection import region
    from dawis.restoration import restored_object
    from dawis.trees import interscale_tree


#: Where the classes a pre-carve pickle names used to live, and where each
#: was carved to. Keyed by module rather than by class because a module moved
#: as a whole: every class the old module held is now in the new one, under
#: the same name. Nothing is renamed here — the rename sweep is #51 — so a
#: name that resolved before the carve resolves after it.
_PRE_CARVE_MODULES = {
    'dawis.restore_objects': 'dawis.restoration.synthesis',
    'dawis.make_interscale_trees': 'dawis.trees.construction',
    'dawis.make_regions': 'dawis.detection.regions',
    'dawis.datacube': 'dawis.cubes',
}


class _LegacyUnpickler(pickle.Unpickler):
    """An unpickler that reads the module paths of the flat package.

    :meth:`pickle.Unpickler.find_class` is the single point every class in a
    pickle stream is resolved through, so redirecting the four moved modules
    there covers atoms, trees, regions and cubes at any depth of nesting. A
    module outside the map is passed through untouched, so numpy and the
    standard library resolve exactly as they did.
    """

    def find_class(self, module: str, name: str) -> type:
        return super().find_class(_PRE_CARVE_MODULES.get(module, module), name)


def _load_legacy(infile: BinaryIO) -> object:
    """Read one pickle stream, resolving pre-carve module paths.

    The one door every reader in this module goes through, so that a format
    concession is made in a single place rather than three.
    """

    return _LegacyUnpickler(infile).load()


def write_ol_to_pickle(
    object_list: list[restored_object],
    filename: str,
    overwrite: bool = True,
) -> None:
    """Write a list of atoms to a pickle file.

    Parameters
    ----------
    object_list : list of restored_object
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


def read_ol_from_pickle(filename: str) -> list[restored_object]:
    """Read a list of atoms back from a pickle file.

    Parameters
    ----------
    filename : str
        Path written by :func:`write_ol_to_pickle`.

    Returns
    -------
    list of restored_object
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
    interscale_tree_list: list[interscale_tree],
    filename: str,
    overwrite: bool = True,
) -> None:
    """Write a list of interscale trees to a pickle file.

    Parameters
    ----------
    interscale_tree_list : list of interscale_tree
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


def read_itl_from_pickle(filename: str) -> list[interscale_tree]:
    """Read a list of interscale trees back from a pickle file.

    Parameters
    ----------
    filename : str
        Path written by :func:`write_itl_to_pickle`.

    Returns
    -------
    list of interscale_tree
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
