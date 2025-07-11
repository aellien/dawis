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
import _pickle as pickle
import gc
import numpy as np
import ray
import logging

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class restored_object(object):
    """docstring for restored_object."""

    def __init__(self, image = None, det_err_image = None, bbox = None, level = None, eccentricity = None, filter_kw = None, flag_convergence = None, sum_wr = None, norm_wr = None, blank = False):
        
        if blank == False:
            
            self.bbox = tuple(bbox)
            self.image = image
            self.det_err_image = det_err_image
            self.level = level
            self.eccentricity = eccentricity
            self.filter = str(filter_kw)
            self.flag_convergence = flag_convergence
            self.sum_wr = sum_wr
            self.norm_wr = norm_wr

@ray.remote
def restore_patch(interscale_tree_patch, wavelet_datacube, label_datacube, extent_sep, ecc_sep, lvl_sep_lin, lvl_sep_big,  lvl_sep_op, gamma, rm_gamma_for_big, deconv):

    object_patch = []
    for tree in interscale_tree_patch:

        image, det_err_image, filter_kw, flag_convergence, sum_wr, norm_wr = restore_object( tree, \
                                                                              wavelet_datacube, \
                                                                              label_datacube, \
                                                                              extent_sep, \
                                                                              ecc_sep, \
                                                                              lvl_sep_lin, \
                                                                              lvl_sep_big, \
                                                                              lvl_sep_op, \
                                                                              deconv )

        if (rm_gamma_for_big == False) or (tree.interscale_maximum.level < lvl_sep_big):
            image = image * gamma  # add attenuation factor



        object_patch.append(restored_object(image, det_err_image, tree.bbox, \
                                            tree.interscale_maximum.level, \
                                            tree.interscale_maximum.eccentricity, \
                                            filter_kw, \
                                            flag_convergence, \
                                            sum_wr, \
                                            norm_wr ))

    return object_patch

def restore_object( interscale_tree, wavelet_datacube, label_datacube, extent_sep, ecc_sep, lvl_sep_lin, lvl_sep_big, lvl_sep_op = 2, deconv = True, verbose = False ):

    bspl = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
    haar = 1 / 2. * np.array([ 1, 0, 1 ])

    flag_convergence = True

    if (( interscale_tree.extent < extent_sep ) or (  interscale_tree.eccentricity >= ecc_sep )) & ( interscale_tree.interscale_maximum.level < lvl_sep_lin ) :
        
        image, sum_wr, norm_wr = interscale_tree.CG_minimization( wavelet_datacube, label_datacube, filter = haar, \
                                                        synthesis_operator = 'SUM', deconv = deconv, verbose = verbose )
        filter_kw = 'HAAR'

    else:
        if interscale_tree.interscale_maximum.level <= lvl_sep_op:
            image, sum_wr, norm_wr = interscale_tree.CG_minimization( wavelet_datacube, label_datacube, filter = bspl, \
                                                    synthesis_operator = 'SUM', deconv = deconv, verbose = verbose )
        else:
            image, sum_wr, norm_wr = interscale_tree.CG_minimization( wavelet_datacube, label_datacube, filter = bspl, \
                                                    synthesis_operator = 'ADJOINT', deconv = deconv, verbose = verbose )
        filter_kw = 'BSPL'

    if interscale_tree.det_err_tube(wavelet_datacube, label_datacube).shape[2] > 1:
        det_err_image = np.sum(interscale_tree.det_err_tube(wavelet_datacube, label_datacube), axis = 2)
    else:
        det_err_image = np.squeeze(interscale_tree.det_err_tube(wavelet_datacube, label_datacube), 2) # from (x, y, 1) to (x, y)    

    # Security
    if np.isnan(np.sum(image)):
        image = np.zeros( image.shape )
        flag_convergence = False

    return image, det_err_image, filter_kw, flag_convergence, sum_wr, norm_wr

def restore_objects_default(interscale_tree_list, oimage, cg_gamma, niter, wavelet_datacube, label_datacube, lvl_sep_big, rm_gamma_for_big,  extent_sep, ecc_sep, lvl_sep_lin, lvl_sep_op, deconv, size_patch_small = 1, gamma = 1, size_patch_big = 1, size_big_objects = 512, n_cpus = 1 ):

    if (len(interscale_tree_list) < size_patch_big) or (n_cpus == 1):

        object_list = []
        bspl = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
        logging.info('Not parallelizing - %d Objects.'%(len(interscale_tree_list)))
        for tree in interscale_tree_list:
            image, det_err_image, filter_kw, flag_convergence, sum_wr, norm_wr = restore_object( tree, \
                                                                                  wavelet_datacube, \
                                                                                  label_datacube, \
                                                                                  extent_sep, \
                                                                                  ecc_sep, \
                                                                                  lvl_sep_lin, \
                                                                                  lvl_sep_big, \
                                                                                  lvl_sep_op, \
                                                                                  verbose = True )

            if (rm_gamma_for_big == False) or (tree.interscale_maximum.level < lvl_sep_big):
                image = image * gamma  # add attenuation factor

            # new quality check
            # some restored objects with the adjoint operator are very bad
            # use residual values to quality check and leave the reconstructed object if too bad
            # hard coded values for tests
            if (tree.interscale_maximum.level > lvl_sep_op) & (sum_wr > 1000) & (norm_wr > 100):
                image = np.zeros(image.shape)
                flag_convergence = False
            object_list.append( restored_object(image, det_err_image, tree.bbox, \
                                                tree.interscale_maximum.level, \
                                                tree.interscale_maximum.eccentricity, \
                                                filter_kw, \
                                                flag_convergence, \
                                                sum_wr, \
                                                norm_wr ))

    else:
        logging.info('Size tree patch (%d) greater than %d, activating Ray store.'%(len(interscale_tree_list), size_patch_big ))
        id_wdc = ray.put(wavelet_datacube)
        id_ldc = ray.put(label_datacube)
        id_extent_sep = ray.put(extent_sep)
        id_ecc_sep = ray.put(ecc_sep)
        id_lvl_sep_lin = ray.put(lvl_sep_lin)
        id_lvl_sep_big = ray.put(lvl_sep_big)
        id_rm_gamma_for_big = ray.put(rm_gamma_for_big)
        id_gamma = ray.put(gamma)
        id_deconv = ray.put(deconv)
        id_lvl_sep_op = ray.put(lvl_sep_op)

        interscale_tree_list.sort(key = lambda x: x.interscale_maximum.area, reverse = True)

        small_interscale_tree_patch = []
        small_object_patch = []
        big_interscale_tree_patch = []
        big_object_patch = []

        for tree in interscale_tree_list:

            if tree.x_size > size_big_objects & tree.y_size > size_big_objects:
                big_interscale_tree_patch.append(tree)

                if len(big_interscale_tree_patch) >= size_patch_big:
                    big_object_patch.append( restore_patch.remote( big_interscale_tree_patch, id_wdc, id_ldc, id_extent_sep, id_ecc_sep, id_lvl_sep_lin, id_lvl_sep_big, id_lvl_sep_op, id_gamma, id_rm_gamma_for_big, id_deconv ))
                    big_interscale_tree_patch = []

            else:

                if big_interscale_tree_patch:
                        big_object_patch.append( restore_patch.remote( big_interscale_tree_patch, id_wdc, id_ldc, id_extent_sep, id_ecc_sep, id_lvl_sep_lin, id_lvl_sep_big, id_lvl_sep_op, id_gamma, id_rm_gamma_for_big, id_deconv ))

                small_interscale_tree_patch.append(tree)

                if len(small_interscale_tree_patch) >= size_patch_small:
                    small_object_patch.append( restore_patch.remote( small_interscale_tree_patch, id_wdc, id_ldc, id_extent_sep, id_ecc_sep, id_lvl_sep_lin, id_lvl_sep_big, id_lvl_sep_op, id_gamma, id_rm_gamma_for_big, id_deconv ))
                    small_interscale_tree_patch = []

        if small_interscale_tree_patch:
            small_object_patch.append( restore_patch.remote( small_interscale_tree_patch, id_wdc, id_ldc, id_extent_sep, id_ecc_sep, id_lvl_sep_lin, id_lvl_sep_big, id_lvl_sep_op, id_gamma, id_rm_gamma_for_big, id_deconv ))

        object_list = []
        for id_patch in big_object_patch:
            patch = ray.get( id_patch )
            object_list = object_list + patch
        for id_patch in small_object_patch:
            patch = ray.get( id_patch )
            object_list = object_list + patch

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
    n_levels = int(np.min(np.floor(np.log2(im.shape))))
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
