"""The two formats DAWIS wrote before 1.0: pickle, and per-iteration HDF5.

A pre-1.0 run left one atom file and one tree file per iteration,
``<name>.ol.itNNN`` and ``<name>.itl.itNNN``, as ``.pkl`` or ``.hdf5``. The
published ``v0.0.7-h`` outputs are recorded as pickles, and the regression
harness (#3) reads its goldens through :func:`read_ol_from_pickle`. Both
formats are kept readable, not recommended: a 1.0 run writes the atom store
of :mod:`dawis.io.store` instead (#122), and resumes from it.
:func:`read_legacy_run` reads a whole pre-1.0 run.

The per-iteration HDF5 writer is generic. :func:`object_recursive_save_to_hdf5`
walks an object's ``__dict__`` and mirrors it into an HDF5 group. The readers
rebuild an atom and a tree the way a pickle does, with ``cls.__new__(cls)``
and no constructor call. The round trip squeezes a stored length-1 array back
to a scalar.

A pickle records the *module path and class name* of every class it holds, so
a run written before the carve names ``dawis.restore_objects.restored_object``,
``dawis.make_interscale_trees.interscale_tree``, ``dawis.make_regions.region``
and the cube classes under ``dawis.datacube``. Those paths resolved only
because the flat shims existed, and the contract ticket (#50) deleted them; the
rename sweep (#51) and the Phase A gate (#59) then changed the class names
themselves. So this module maps both halves itself, in
:class:`_LegacyUnpickler`. Without the module map a
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
import glob
import logging
import pickle
from typing import BinaryIO

import h5py
import numpy as np

from dawis.detection import region
from dawis.restoration import Atom
from dawis.trees import InterscaleTree

log = logging.getLogger(__name__)


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


def _load_hdf5_attribute(dataset: h5py.Dataset) -> object:
    """Read one stored attribute back, as the type it was written as.

    The one door every HDF5 group loader in this module reads a dataset
    through, so that the two concessions the format asks for are made once
    rather than three times. Private, unlike the three loaders it serves: it
    is how they read, not a way to read a run.

    Parameters
    ----------
    dataset : h5py.Dataset
        A dataset written by :func:`object_recursive_save_to_hdf5`.

    Returns
    -------
    object
        The stored value: an array as an array, a number as a scalar, and a
        string as :class:`str`.

    Notes
    -----
    Two concessions. The writer stores a scalar as a length-1 array, so this
    squeezes; a genuine length-1 array therefore comes back as a scalar, which
    is the round trip's one lossy corner and predates this function.

    And h5py hands a stored string back as :class:`bytes`, so
    ``atom.filter`` read out of HDF5 used to be ``b'BSPL'`` where the same atom
    read out of a pickle was ``'BSPL'`` — one attribute with two types
    depending on which file it came from. Decoding here makes the two agree
    (#66). The goldens are recorded from pickles, where the filter was already
    text, so nothing recorded can move.
    """

    value = np.squeeze(dataset[()])

    # `S` is numpy's fixed-width bytes kind, which is what a stored string
    # squeezes to: h5py hands one back as `bytes` and `np.squeeze` wraps that
    # in a 0-d array. So the test is on the array, not on `bytes` itself.
    if isinstance(value, np.ndarray) and value.dtype.kind == 'S':
        text = value.astype(str)
        return text.item() if text.ndim == 0 else text

    return value


def object_recursive_save_to_hdf5(o: object, h5grp: h5py.Group) -> None:
    """Recursively save a Python object to an HDF5 group.

    Every attribute of ``o`` becomes a dataset if h5py accepts it directly,
    and a subgroup otherwise — one subgroup per nested object, or one
    ``r%d`` subgroup per element of a list of objects.

    Parameters
    ----------
    o : object
        The object to save. Read through its ``__dict__``.
    h5grp : h5py.Group
        The group to save into.

    Notes
    -----
    Recursion stops at depth two: a nested object's own nested attributes are
    written as arrays, and an attribute that is neither array-like, a list nor
    an object is logged and dropped rather than raising.
    """

    for attr, val in o.__dict__.items():
                
        # Test if standard data structure
        try:
            h5grp.create_dataset(attr, data = val)
            
        # If not standard, should be a list of objects or an object
        except TypeError:
            
            # Create subgroup
            sub_h5grp = h5grp.create_group(attr)
            
            # recursive call
            # If data structure is a list of objects
            if isinstance(val, list):
                
                for i, sub_val in enumerate(val):
                    sub_sub_h5grp = sub_h5grp.create_group('r%d'%i)
                    object_recursive_save_to_hdf5(sub_val, sub_sub_h5grp)
                    
        
            # If data structure is an object
            elif hasattr(val, '__dict__'):
                
                for sub_attr, sub_val in val.__dict__.items():
                    # Test if standard data structure
                    try:
                        
                        if isinstance(sub_val, np.ndarray):
                            sub_h5grp.create_dataset(sub_attr, data = sub_val)
                        else:
                            sub_h5grp.create_dataset(sub_attr, data = np.array([sub_val]))

                    # If not standard, should be a list of objects, or an object
                    except TypeError:
                        object_recursive_save_to_hdf5(sub_val, sub_h5grp)       
            else:
                log.info('Unknown data structure for %s. NOT SAVING TO HDF5 FILE.' %(attr))


def write_itl_to_hdf5(itl: list[InterscaleTree], filename: str) -> None:
    """Write a list of interscale trees to an HDF5 file.

    Parameters
    ----------
    itl : list of InterscaleTree
        The trees of one iteration. Tree ``i`` becomes group ``it{i}``.
    filename : str
        Path to write to. Any existing file is replaced.
    """
    with h5py.File(filename, "w") as f:
        gc.collect()
        for i, o in enumerate(itl):
            grp = f.create_group(f'it{i}')
            object_recursive_save_to_hdf5(o, grp)


def read_itl_from_hdf5(filename: str) -> list[InterscaleTree]:
    """Read a list of interscale trees back from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path written by :func:`write_itl_to_hdf5`.

    Returns
    -------
    list of InterscaleTree
        The trees, in the order h5py yields the groups, which is
        alphabetical: ``it10`` precedes ``it2``.
    """
    itl = []
    with h5py.File(filename, "r") as f:
        gc.collect()
        for grp_name in f.keys():
            grp = f[grp_name]
            it = load_interscale_tree_from_hdf5_group(grp)
            itl.append(it)
    return itl


def load_interscale_tree_from_hdf5_group(grp: h5py.Group) -> InterscaleTree:
    """Rebuild one interscale tree from an HDF5 group.

    Parameters
    ----------
    grp : h5py.Group
        A group written by :func:`object_recursive_save_to_hdf5`.

    Returns
    -------
    InterscaleTree
        An empty tree with the stored attributes set on it.

    Notes
    -----
    Only ``region_list`` and ``interscale_maximum`` are restored among the
    subgroups; any other subgroup is skipped.
    """
    it = InterscaleTree.__new__(InterscaleTree)
    for attr_name in grp.keys():
        attr = grp[attr_name]
        
        # If dataset --> standard attribute
        if isinstance(attr, h5py.Dataset):
            setattr(it, attr_name, _load_hdf5_attribute(attr))
            
        # If group --> a list of regions, or a region
        elif isinstance(attr, h5py.Group):
            
            if attr_name == 'region_list':  
                rl = []
                for rnum in attr.keys():
                    rgrp = grp['region_list'][rnum]
                    r = load_region_from_hdf5_group(rgrp)
                    rl.append(r)
                setattr(it, attr_name, rl)
                
            elif attr_name == 'interscale_maximum':
                interscale_maximum = load_region_from_hdf5_group(attr)
                setattr(it, attr_name, interscale_maximum)
    return it


def load_region_from_hdf5_group(grp: h5py.Group) -> region:
    """Rebuild one region from an HDF5 group.

    Parameters
    ----------
    grp : h5py.Group
        A group holding one region's attributes as datasets.

    Returns
    -------
    region
        A default region with the stored attributes set on it.
    """
    r = region()
    for attr_name in grp.keys():
        attr = grp[attr_name]
        setattr(r, attr_name, _load_hdf5_attribute(attr))
    return r


def write_ol_to_hdf5(object_list: list[Atom], filename: str) -> None:
    """Write a list of atoms to an HDF5 file.

    Parameters
    ----------
    object_list : list of Atom
        The atoms of one iteration. Atom ``i`` becomes group ``o{i}``.
    filename : str
        Path to write to. Any existing file is replaced.
    """
    with h5py.File(filename, "w") as f:
        gc.collect()
        for i, o in enumerate(object_list):
            grp = f.create_group(f'o{i}')
            object_recursive_save_to_hdf5(o, grp)


def read_ol_from_hdf5(filename: str) -> list[Atom]:
    """Read a list of atoms back from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path written by :func:`write_ol_to_hdf5`.

    Returns
    -------
    list of Atom
        The atoms, in the order h5py yields the groups, which is
        alphabetical: ``o10`` precedes ``o2``.
    """
    ol = []
    with h5py.File(filename, "r") as f:
        gc.collect()
        for grp_name in f.keys():
            grp = f[grp_name]
            o = load_atom_from_hdf5_group(grp)
            ol.append(o)
    return ol


def load_atom_from_hdf5_group(grp: h5py.Group) -> Atom:
    """Rebuild one atom from an HDF5 group.

    Parameters
    ----------
    grp : h5py.Group
        A group holding one atom's attributes as datasets.

    Returns
    -------
    Atom
        An empty atom with the stored attributes set on it.

    Notes
    -----
    An atom written before #66 holds no ``algorithm`` and no
    ``synthesis_operator`` dataset, so the empty instance keeps the class
    attributes :class:`~dawis.restoration.synthesis.Atom` declares and both
    read ``'UNKNOWN'`` rather than raising.
    """
    o = Atom.__new__(Atom)
    for attr_name in grp.keys():
        
        attr = grp[attr_name]
        setattr(o, attr_name, _load_hdf5_attribute(attr))
    return o


def pkl_to_hdf5(pkl_ol: str, pkl_itl: str) -> None:
    """Convert one iteration's legacy pickle files to the HDF5 store.

    Parameters
    ----------
    pkl_ol : str
        Path to an atom list written by
        :func:`~dawis.io.legacy.write_ol_to_pickle`.
    pkl_itl : str
        Path to the matching tree list written by
        :func:`~dawis.io.legacy.write_itl_to_pickle`.

    Notes
    -----
    The outputs are the same paths with the last three characters replaced by
    ``hdf5``, so both arguments must end in ``.pkl``.
    """
    
    ol = read_ol_from_pickle(pkl_ol)
    itl = read_itl_from_pickle(pkl_itl)
    write_ol_to_hdf5(ol, pkl_ol[:-3] + 'hdf5')
    write_itl_to_hdf5(itl, pkl_itl[:-3] + 'hdf5')


def read_legacy_run(
    nfp: str,
    file_format: str = 'hdf5',
    filter_it: str | None = None,
    verbose: bool = True,
) -> tuple[list[Atom], list[InterscaleTree]]:
    """Read a whole pre-1.0 run's atoms back off disk, with the tree behind each one.

    Parameters
    ----------
    nfp : str
        Path prefix the run was written under — everything before
        ``.ol.itNNN``.
    file_format : {'hdf5', 'pkl'}, optional
        Which of the two pre-1.0 formats to read. Default is ``'hdf5'``.
    filter_it : str or None, optional
        Read one iteration only, as the zero-padded string it appears as in
        the filename, e.g. ``'003'``. Default None reads every iteration.
    verbose : bool, optional
        Log the two glob patterns and print a progress counter. Default True.

    Returns
    -------
    tol : list of Atom
        Every atom of the run, iterations concatenated in filename order.
    titl : list of InterscaleTree
        The tree each atom was restored from, at the same index.

    Notes
    -----
    The two file lists are sorted as strings and then zipped, so iteration
    ``10`` precedes iteration ``2`` and a run missing one of the two files for
    some iteration pairs atoms with the wrong trees rather than raising.
    """

    if file_format == 'pkl':
        
        if filter_it == None:
            opath = nfp + '*ol.it*.pkl'
            itpath = nfp + '*itl.it*.pkl'
        else:
            opath = nfp + '*ol.it' + filter_it  + '.pkl'
            itpath = nfp + '*itl.it' + filter_it + '.pkl'
            
    elif file_format == 'hdf5':
        
        if filter_it == None:
            opath = nfp + '*ol.it*.hdf5'
            itpath = nfp + '*itl.it*.hdf5'
        else:
            opath = nfp + '*ol.it' + filter_it  + '.hdf5'
            itpath = nfp + '*itl.it' + filter_it + '.hdf5'

    # Object lists
    opathl = glob.glob(opath)
    opathl.sort()

    # Interscale tree lists
    itpathl = glob.glob(itpath)
    itpathl.sort()

    tol = []
    titl = []

    if verbose:
        log.info('Reading %s.'%(opath))
        log.info('Reading %s.'%(itpath))

    for i, ( op, itlp ) in enumerate( zip( opathl, itpathl )):
        
        if verbose :
            log.debug('Iteration %d' %(i))
            
        if file_format == 'pkl':
            ol = read_ol_from_pickle( op )
            itl = read_itl_from_pickle( itlp )

        elif file_format == 'hdf5':
            ol = read_ol_from_hdf5( op)
            itl = read_itl_from_hdf5(itlp)
            
        for j, o in enumerate(ol):

            tol.append(o)
            titl.append(itl[j])

    return tol, titl
