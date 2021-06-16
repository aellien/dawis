#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# This file is part of DAWIS.
#
# NAME : restore_objects.py
# AUTHOR : Amael Ellien
# LAST MODIFICATION : 07/2021
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from datetime import datetime
import pdb
import matplotlib.pyplot as plt
import pickle
import numpy as np
import ray
import logging

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class restored_object(object):
    """docstring for restored_object."""

    def __init__(self, image, bbox, level, eccentricity):
        self.bbox = bbox
        self.image = image
        self.level = level
        self.eccentricity = eccentricity

def write_objects_to_pickle(object_list, filename, overwrite = True):

    if overwrite == True:
        mode = 'wb'
    else:
        mode = 'ab'

    with open(filename, mode) as outfile:
        pickle.dump(object_list, outfile)

def read_objects_from_pickle(filename):

    with open(filename, 'rb') as infile:
        object_list = pickle.load(infile)

    return object_list


@ray.remote
def restore_patch(interscale_tree_patch, wavelet_datacube, label_datacube, extent_sep, lvl_sep_lin, lvl_sep_big):

    object_patch = []
    for tree in interscale_tree_patch:

        image = restore_object( tree, \
                                wavelet_datacube, \
                                label_datacube, \
                                extent_sep, \
                                lvl_sep_lin, lvl_sep_big )

        object_patch.append(restored_object(image, tree.bbox, \
                                            tree.interscale_maximum.level, \
                                            tree.interscale_maximum.eccentricity))

    return object_patch

def restore_object( interscale_tree, wavelet_datacube, label_datacube, extent_sep, lvl_sep_lin, lvl_sep_big ):

    bspl = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
    haar = 1 / 2. * np.array([ 1, 0, 1 ])

    if ( interscale_tree.extent < extent_sep ) & ( interscale_tree.interscale_maximum.level < lvl_sep_lin ) :
        image = interscale_tree.SD_minimization( wavelet_datacube, label_datacube, filter = haar, \
                                                        synthesis_operator = 'ADJOINT' )

    else:
        image = interscale_tree.SD_minimization( wavelet_datacube, label_datacube, filter = bspl, \
                                                    synthesis_operator = 'ADJOINT' )

    return image


def restore_objects_default(interscale_tree_list, wavelet_datacube, label_datacube, lvl_sep_big, extent_sep, lvl_sep_lin, size_patch_small = 50, size_patch_big = 5, size_big_objects = 512, n_cpus = 1 ):

    if len(interscale_tree_list) < size_patch_big:
        # params_minimization = strategy(tree)
        object_list = []
        bspl = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
        for tree in interscale_tree_list:
            image = restore_object( tree, \
                                    wavelet_datacube, \
                                    label_datacube, \
                                    extent_sep, \
                                    lvl_sep_lin,
                                    lvl_sep_big )

            object_list.append( restored_object(image, tree.bbox, \
                                                tree.interscale_maximum.level, \
                                                tree.interscale_maximum.eccentricity) )

    else:
        ray.init(num_cpus = 4)
        id_wdc = ray.put(wavelet_datacube)
        id_ldc = ray.put(label_datacube)
        id_extent_sep = ray.put(extent_sep)
        id_lvl_sep_lin = ray.put(lvl_sep_lin)
        id_lvl_sep_big = ray.put(lvl_sep_big)

        interscale_tree_list.sort(key = lambda x: x.interscale_maximum.area, reverse = True)

        small_interscale_tree_patch = []
        small_object_patch = []
        big_interscale_tree_patch = []
        big_object_patch = []

        for tree in interscale_tree_list:

            if tree.x_size > size_big_objects & tree.y_size > size_big_objects:
                big_interscale_tree_patch.append(tree)

                if len(big_interscale_tree_patch) >= size_patch_big:
                    big_object_patch.append( restore_patch.remote( big_interscale_tree_patch, id_wdc, id_ldc, id_extent_sep, id_lvl_sep_lin, id_lvl_sep_big))
                    big_interscale_tree_patch = []

            else:

                if big_interscale_tree_patch:
                        big_object_patch.append( restore_patch.remote( big_interscale_tree_patch, id_wdc, id_ldc, id_extent_sep, id_lvl_sep_lin, id_lvl_sep_big ))

                small_interscale_tree_patch.append(tree)

                if len(small_interscale_tree_patch) >= size_patch_small:
                    small_object_patch.append( restore_patch.remote( small_interscale_tree_patch, id_wdc, id_ldc, id_extent_sep, id_lvl_sep_lin, id_lvl_sep_big ))
                    small_interscale_tree_patch = []

        if small_interscale_tree_patch:
            small_object_patch.append( restore_patch.remote( small_interscale_tree_patch, id_wdc, id_ldc, id_extent_sep, id_lvl_sep_lin, id_lvl_sep_big ))

        object_list = []
        for id_patch in big_object_patch:
            patch = ray.get( id_patch )
            object_list = object_list + patch
        for id_patch in small_object_patch:
            patch = ray.get( id_patch )
            object_list = object_list + patch

        ray.shutdown()

    return object_list

if __name__ == '__main__':

    from make_regions import *
    from datacube import *
    from matplotlib.colors import SymLogNorm, LogNorm
    from astropy.io import fits
    from atrous import atrous
    from make_regions import make_regions_full_props
    from wavelet_transforms import bspl_atrous, wavelet_datacube
    from hard_threshold import hard_threshold
    from label_regions import label_regions
    from anscombe_transforms import *
    from skimage.metrics import structural_similarity as ssim
    from make_interscale_trees import make_interscale_trees

    hdu = fits.open('/home/ellien/devd/gallery/A1365.rebin.fits')
    im = hdu[0].data
    header = hdu[0].header
    n_levels = np.int(np.min(np.floor(np.log2(im.shape))))
    print(im.shape, n_levels, 2**n_levels)
    n_levels = 3

    sigma, mean, gain = pg_noise_bissection(im, max_err = 1E-3, n_sigmas = 3)
    aim = anscombe_transform(im, sigma, mean, gain)
    acdc, awdc = bspl_atrous(aim, n_levels, header)
    sdc = hard_threshold(awdc, n_sigmas = 5)
    ldc = label_regions(sdc)
    cdc, wdc = bspl_atrous(im, n_levels, header)
    rl = make_regions_full_props(wdc, ldc, verbose = True)
    itl = make_interscale_trees(rl, ldc, tau = 0.8, min_span = 3, verbose = True)
    ol = restore_objects_default(itl, wdc,ldc, size_patch_small = 50, \
                                               size_patch_big = 5, \
                                               size_big_objects = 512, \
                                               n_cpus = 4 )
