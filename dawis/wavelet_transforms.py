#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This file is part of DAWIS (Detection Algorithm for Intracluster light Studies).
# Author: Amaël Ellien
# Last modification: 02/07/2021
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODULES

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib is the optional [viz] extra
    from dawis._optional import MissingDependency
    plt = MissingDependency('matplotlib', 'viz')
try:
    from matplotlib.colors import SymLogNorm
except ImportError:  # matplotlib is the optional [viz] extra
    from dawis._optional import MissingDependency
    SymLogNorm = MissingDependency('matplotlib.colors.SymLogNorm', 'viz')
from astropy.io import fits
from dawis.exceptions import DawisDimensionError
from dawis.datacube import *
from dawis.atrous import atrous, boundary_conditions, ser_iuwt_decomposition
from dawis.exceptions import DawisWrongType
import pdb
import logging

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def bspl_atrous(image, n_levels, fheader = None, conditions = 'loop', verbose = False):
    '''doc to do'''

    #filter = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
    #coarse_array, wavelet_array = atrous(image = image, n_levels = n_levels, filter = filter, conditions = 'loop')
    wavelet_array, coarse_array = ser_iuwt_decomposition(image, n_levels, 0, store_smoothed = True)
    #coarse_dc = coarse_datacube(array = coarse_array, wavelet_type = 'BSPL', fheader = fheader)
    wavelet_dc = wavelet_datacube(array = wavelet_array, wavelet_type = 'BSPL', fheader = fheader)

    return wavelet_dc

def haar_atrous(image, n_levels, fheader = None, verbose = False):
    '''doc to do'''

    filter = 1 / 2. * np.array([ 1, 0, 1 ])
    coarse_array, wavelet_array = atrous(image = image, n_levels = n_levels, filter = filter, conditions = 'loop')

    coarse_dc = coarse_datacube(array = coarse_array, wavelet_type = 'HAAR', fheader = fheader)
    wavelet_dc = wavelet_datacube(array = wavelet_array, wavelet_type = 'HAAR', fheader = fheader)

    return coarse_dc, wavelet_dc

if __name__ == '__main__':

    from atrous import atrous
    from numpy.random import normal
    from astropy.io import fits

    im = normal(0, 1,(1024, 1024))
    hdu = fits.open('/home/ellien/wavelets/A1365.rebin.fits')
    im = hdu[0].data
    header = hdu[0].header
    cdc, wdc = haar_atrous(im, 10, header)

    wdc.waveplot(cmap = 'hot')
    cdc.waveplot()
