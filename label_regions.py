#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# this file is part of dawis 3.0.
#
# name : label_regions.py
# author : amael ellien
# last modification : 05/2019
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# labels every regions of the support. it means that a number is given to each
# region of the support (packs of pixels with a value of 1 surrounded by pixels
# with a value of 0), starting at 1. note that the labbeling is made so each
# region is unique through the whole wavelet space.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from dawis.datacube import *
from skimage.measure import label
import numpy as np
import matplotlib.pyplot as plt
import logging

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def label_regions(support_datacube, verbose = False):

    '''
    labelling of the regions (pixels surrounded by non-zero pixels) of the
    multiscale support. note that the labbeling is made so each region is unique
    through the whole wavelet space.
    '''

    label_array = np.zeros(support_datacube.shape, order = 'f')
    lab_counts = np.zeros(support_datacube.z_size)

    if verbose == True:
        log = logging.getLogger(__name__)
        log.info('\nLabelling regions')

    for level in range(0, support_datacube.z_size):
        label_array[:, :, level], lab_counts[level] = label(support_datacube.array[:,:,level], \
                                                      connectivity = 1, \
                                                      return_num = True)
        if level != 0:
            reg = np.where( label_array[:, :, level] != 0 )               # here we make
            temp = label_array[:, :, level]                           # sure that the
            temp[reg] = temp[reg] + np.sum( lab_counts[0:level] ) # labels are
            label_array[:, :, level] = temp                           # unique through
                                                            # the whole
                                                            # wavelet space.
        if verbose == True:
            log.info('level %02d nregions %d labels %d - %d'\
                    %(level+1,lab_counts[level],\
                    np.max(label_array[:,:,level])-lab_counts[level]+1,\
                    np.max(label_array[:,:,level])))

    del temp, reg

    label_dc = label_datacube(label_array, lab_counts, support_datacube.fheader)

    return label_dc

if __name__ == '__main__':

    from atrous import atrous
    from numpy.random import normal
    from astropy.io import fits
    from wavelet_transforms import bspl_atrous, wavelet_datacube
    from hard_threshold import hard_threshold

    hdu = fits.open('/home/ellien/wavelets/A1365.rebin.fits')
    im = hdu[0].data
    header = hdu[0].header
    cdc, wdc = bspl_atrous(im, 10, header)
    sdc = hard_threshold(wdc, n_sigmas = 5)
    ldc = label_regions(sdc)
    ldc.label_counts_plot()
