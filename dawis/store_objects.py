#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:38:40 2024

@author: aellien
"""
from dawis.make_interscale_trees import interscale_tree
from dawis.restore_objects import restored_object
from dawis.make_regions import region
import h5py
import numpy as np
import logging
import pickle
import sys
import os
import gc
import glob

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def object_recursive_save_to_hdf5(o, h5grp):
    """
    Recursively save a Python object to an HDF5 file.

    This function takes a Python object and an HDF5 group, and recursively saves 
    the object's attributes to the group. It handles standard data structures 
    like NumPy arrays and lists, as well as nested objects.

    Args:
        o (object): The Python object to be saved.
        h5grp (h5py.Group): The HDF5 group to save the object to.

    Raises:
        TypeError: If the object contains an unsupported data structure.
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
                log = logging.getLogger(__name__)
                log.info('Unknown data structure for %s. NOT SAVING TO HDF5 FILE.' %(attr))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
def write_itl_to_hdf5(itl, filename):
    """
    Save a list of interscale tree instances to an HDF5 file.

    This function takes a list of interscale trees and saves them to an HDF5 file. 
    It uses the `object_recursive_save_to_hdf5` function to recursively save each 
    object's attributes to the HDF5 file.

    Args:
        itl (list): A list of interscale tree instances
        filename (str): The name of the HDF5 file to be created.
    """
    with h5py.File(filename, "w") as f:
        gc.collect()
        for i, o in enumerate(itl):
            grp = f.create_group(f'it{i}')
            object_recursive_save_to_hdf5(o, grp)
          
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_itl_from_hdf5(filename):
    """
    Load a list of interscale tree instances from an HDF5 file.
    
    Args:
        filename (str): The name of the HDF5 file to be read.
    
    Returns:
        itl: A list of interscale tree instances loaded from the HDF5 file.
    """
    itl = []
    with h5py.File(filename, "r") as f:
        gc.collect()
        for grp_name in f.keys():
            grp = f[grp_name]
            it = load_interscale_tree_from_hdf5_group(grp)
            itl.append(it)
    return itl

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def load_interscale_tree_from_hdf5_group(grp):
    """
    Load a Python object from an HDF5 group.
    
    Args:
        grp (h5py.Group): The HDF5 group containing the object data.
    
    Returns:
        object: The Python object loaded from the HDF5 group.
    """
    it = interscale_tree(blank = True)
    for attr_name in grp.keys():
        attr = grp[attr_name]
        
        # If dataset --> standard attribute
        if isinstance(attr, h5py.Dataset):
            setattr(it, attr_name, np.squeeze(attr[()])) # squeeze to remove the 
                                                         # useless dimension added when creating HDF5
            
        # If group --> a list of regions, or a region
        elif isinstance(attr, h5py.Group):
            
            if attr_name == 'region_list':  
                rl = []
                #import pdb;pdb.set_trace()
                for rnum in attr.keys():
                    rgrp = grp['region_list'][rnum]
                    r = load_region_from_hdf5_group(rgrp)
                    rl.append(r)
                setattr(it, attr_name, rl)
                
            elif attr_name == 'interscale_maximum':
                interscale_maximum = load_region_from_hdf5_group(attr)
                setattr(it, attr_name, interscale_maximum)
    return it

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def load_region_from_hdf5_group(grp):
    """Load a region object from an HDF5 group.

    Args:
        grp (h5py.Group): The HDF5 group containing the region data.

    Returns:
        d.region: A region object with attributes loaded from the HDF5 group.
    """
    r = region()
    for attr_name in grp.keys():
        attr = grp[attr_name]
        setattr(r, attr_name, np.squeeze(attr[()]))
    return r
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
def write_ol_to_hdf5(object_list, filename):
    """
    Save a list of restored objects to an HDF5 file.

    This function takes a list of restored objects and saves them to an HDF5 file. 
    It uses the `object_recursive_save_to_hdf5` function to recursively save each 
    object's attributes to the HDF5 file.

    Args:
        objects (list): A list of restored objects to be saved.
        filename (str): The name of the HDF5 file to be created.
    """
    with h5py.File(filename, "w") as f:
        gc.collect()
        for i, o in enumerate(object_list):
            grp = f.create_group(f'o{i}')
            object_recursive_save_to_hdf5(o, grp)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_ol_from_hdf5(filename):
    """
    Load a list of Python objects from an HDF5 file.
    
    Args:
        filename (str): The name of the HDF5 file to be read.
    
    Returns:
        list: A list of Python objects loaded from the HDF5 file.
    """
    ol = []
    with h5py.File(filename, "r") as f:
        gc.collect()
        for grp_name in f.keys():
            grp = f[grp_name]
            o = load_restored_object_from_hdf5_group(grp)
            ol.append(o)
    return ol

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def load_restored_object_from_hdf5_group(grp):
    """Load a restored object from an HDF5 group.

    Args:
        grp (h5py.Group): The HDF5 group containing the restored object data.

    Returns:
        d.restored_object: A restored object with attributes loaded from the HDF5 group.
    """
    o = restored_object(blank = True)
    for attr_name in grp.keys():
        
        attr = grp[attr_name]
        #print(attr_name, attr)
        setattr(o, attr_name, np.squeeze(attr[()]))
    return o

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_ol_to_pickle(object_list, filename, overwrite = True):

    if overwrite == True:
        mode = 'wb'
    else:
        mode = 'ab'

    with open(filename, mode) as outfile:
        gc.collect()
        pickle.dump(object_list, outfile)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_ol_from_pickle(filename):

    with open(filename, 'rb') as infile:
        gc.collect()
        object_list = pickle.load(infile)

    return object_list

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_itl_to_pickle(interscale_tree_list, filename, overwrite = True):

    if overwrite == True:
        mode = 'wb'
    else:
        mode = 'ab'

    with open(filename, mode) as outfile:
        gc.collect()
        pickle.dump(interscale_tree_list, outfile)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_itl_from_pickle(filename):

    with open(filename, 'rb') as infile:
        gc.collect()
        interscale_tree_list = pickle.load(infile)
        
    return interscale_tree_list

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def pkl_to_hdf5(pkl_ol, pkl_itl):
    
   ol = read_ol_from_pickle(pkl_ol)
   itl = read_itl_from_pickle(pkl_itl)
   write_ol_to_hdf5(ol, pkl_ol[:-3] + 'hdf5')
   write_itl_to_hdf5(itl, pkl_itl[:-3] + 'hdf5')
   
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_image_atoms( nfp, file_format = 'hdf5', filter_it = None, verbose = True ):

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
        log = logging.getLogger(__name__)
        log.info('Reading %s.'%(opath))
        log.info('Reading %s.'%(itpath))

    for i, ( op, itlp ) in enumerate( zip( opathl, itpathl )):
        
        if verbose :
            print('Iteration %d' %(i), end ='\r')
            
        if file_format == 'pkl':
            ol = read_objects_from_pickle( op )
            itl = read_interscale_trees_from_pickle( itlp )

        elif file_format == 'hdf5':
            ol = read_ol_from_hdf5( op)
            itl = read_itl_from_hdf5(itlp)
            
        for j, o in enumerate(ol):

            tol.append(o)
            titl.append(itl[j])

    return tol, titl
   
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

    import sys
    filename_ol = sys.argv[1] # .pkl
    filename_itl = sys.argv[2] # .pkl
    pkl_to_hdf5(filename_ol, filename_itl)
    
    
    
    