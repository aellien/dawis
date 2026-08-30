"""The object store: what one iteration leaves on disk.

The pipeline writes two things per iteration — the atoms it restored and the
interscale trees they came from — and reads them back to resume. This module
is that store, in its current HDF5 form.

The writer is deliberately generic. :func:`object_recursive_save_to_hdf5`
walks an object's ``__dict__`` and mirrors it into an HDF5 group: arrays and
scalars become datasets, a nested object or a list of objects becomes a
subgroup. Nothing here names an attribute of an atom or a tree, so the store
does not have to be edited when one of them gains a field.

The readers are not generic, because reading has to know what class to build.
:func:`read_ol_from_hdf5` and :func:`read_itl_from_hdf5` each rebuild their
own type from a blank instance, which is why
:class:`~dawis.restoration.synthesis.Atom`,
:class:`~dawis.trees.construction.InterscaleTree` and
:class:`~dawis.detection.regions.region` all take a ``blank`` constructor
flag. Those three imports are the only edges this module has to the other
blocks, and they run one way: no block imports :mod:`dawis.io`.

The round trip is lossy in one respect worth knowing. The writer stores a
scalar as a length-1 array, so every reader squeezes on the way back; a
genuine length-1 array therefore returns as a scalar.

:func:`read_image_atoms` is the run-level entry: it globs a whole run's
per-iteration files and returns every atom with the tree that produced it. It
reads the legacy pickle format as well as HDF5, through
:mod:`dawis.io.legacy`.

Phase B replaces this with a first-class ``ObjectStore``; through Phase A the
on-disk format is unchanged, which is what makes the goldens diagnostic.
"""

from __future__ import annotations

import gc
import glob
import logging

import h5py
import numpy as np

from dawis.detection import region
from dawis.io.legacy import read_itl_from_pickle, read_ol_from_pickle
from dawis.restoration import Atom
from dawis.trees import InterscaleTree

log = logging.getLogger(__name__)


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
        A blank tree with the stored attributes set on it.

    Notes
    -----
    Only ``region_list`` and ``interscale_maximum`` are restored among the
    subgroups; any other subgroup is skipped.
    """
    it = InterscaleTree(blank = True)
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
        A blank atom with the stored attributes set on it.

    Notes
    -----
    An atom written before #66 holds no ``algorithm`` and no
    ``synthesis_operator`` dataset, so the blank instance keeps the class
    attributes :class:`~dawis.restoration.synthesis.Atom` declares and both
    read ``'UNKNOWN'`` rather than raising.
    """
    o = Atom(blank = True)
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


def read_image_atoms(
    nfp: str,
    file_format: str = 'hdf5',
    filter_it: str | None = None,
    verbose: bool = True,
) -> tuple[list[Atom], list[InterscaleTree]]:
    """Read a whole run's atoms back off disk, with the tree behind each one.

    Parameters
    ----------
    nfp : str
        Path prefix the run was written under — everything before
        ``.ol.itNNN``.
    file_format : {'hdf5', 'pkl'}, optional
        Which format to read. Default is ``'hdf5'``; ``'pkl'`` reads a legacy
        run through :mod:`dawis.io.legacy`.
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


if __name__ == '__main__':

    import sys
    filename_ol = sys.argv[1] # .pkl
    filename_itl = sys.argv[2] # .pkl
    pkl_to_hdf5(filename_ol, filename_itl)
