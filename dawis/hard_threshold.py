#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# This file is part of DAWIS 3.0.
#
# NAME : hard_threshold.py
# AUTHOR : Amael Ellien
# LAST MODIFICATION : 07/2021
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Hard thresholding process for the multiscale support. The threshold is a
# simple constant given by N_SIGMA*SIGMA.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from dawis.table_noise_stds import table_noise_stds
from dawis.datacube import *
from scipy.stats import sigmaclip
from scipy.stats import normaltest
import numpy as np
import matplotlib.pyplot as plt
import logging

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def hard_threshold(wavelet_datacube, n_sigmas = 3, wavelet_type = None, keep_neg = False, verbose = False, display = False):

    '''
    Hard thresholding process for the multiscale support. The threshold is a simple constant given by N_SIGMA*SIGMA.
    '''

    # Add exceptions

    x_axis, y_axis, n_levels = wavelet_datacube.shape

    support = np.zeros((x_axis, y_axis, n_levels))

    if wavelet_type:
        conv_table = table_noise_stds(n_levels, wavelet_type = wavelet_type)
    else:
        wavelet_type = wavelet_datacube.fheader['WAVTYPE']
        conv_table = table_noise_stds(n_levels, wavelet_type = wavelet_type)

    if verbose == True:
        log = logging.getLogger(__name__)
        log.info('\ncreating support datacube : hard thresholding')

    noise_pixels = sigmaclip(wavelet_datacube.array[:,:,0], low = n_sigmas, high = n_sigmas)[0]
    sigma = np.std(noise_pixels)

    alpha = 1E-3
    if noise_pixels[0].size > 0:
        k, pval = normaltest( noise_pixels )
        if pval > alpha:
            log.info('/!\ Warning /!\ noise does not look Gaussian : p = %1.5f alpha = %1.5f'%(pval, alpha))

    if display == True:
        plt.figure(1)
        plt.hist(noise_pixels[0].flatten(),  bins = 'auto',  histtype = 'bar', rwidth = 10. , align='left')
        plt.show()

    for level in range(0, n_levels):

        if keep_neg == True:
            non_masked_pixels = np.where( abs( wavelet_datacube.array[:,:, level] )\
                         >= (n_sigmas * sigma * conv_table[level] ))

        elif keep_neg == False:
            non_masked_pixels = np.where( wavelet_datacube.array[:,:,level]\
                         >= (n_sigmas * sigma * conv_table[level] ))

        temp = support[:,:,level]
        temp[non_masked_pixels] = 1.0
        support[:,:,level] = temp

        if verbose == True:
            log.info('level %02d sigma %f' %(level+1, n_sigmas*sigma*conv_table[level] ))

    support_dc = support_datacube(support, threshold_type = 'HARD', \
                                         noise_pixels = noise_pixels, \
                                         fheader = wavelet_datacube.fheader)
    return support_dc

if __name__ == '__main__':

    #wdc = datacube.from_fits('/home/ellien/devd/tests/cat_gal_004_z_0.1_Ficl_0.4_Re_050_noise_megacam_dcw.fits')
    #wdc.fheader['WAVTYPE'] = 'LBSPL'
    #wdc.to_fits('/home/ellien/devd/tests/cat_gal_004_z_0.1_Ficl_0.4_Re_050_noise_megacam_dcw.fits', overwrite = True)

    from atrous import atrous
    from numpy.random import normal
    from astropy.io import fits
    from wavelet_transforms import bspl_atrous, wavelet_datacube

    hdu = fits.open('/home/ellien/wavelets/A1365.rebin.fits')
    im = hdu[0].data
    header = hdu[0].header
    cdc, wdc = bspl_atrous(im, 10, header)
    cdc.to_fits('/home/ellien/devd/tests/A1365.rebin.cdc.fits', overwrite = True)
    wdc.to_fits('/home/ellien/devd/tests/A1365.rebin.wdc.fits', overwrite = True)
    #wdc.std_plot(name = 'test', markerstyle = 'o', color = 'black', linestyle = '-')
    wdc = wavelet_datacube.from_fits('/home/ellien/devd/tests/A1365.rebin.wdc.fits')
    sdc = hard_threshold(wdc, n_sigmas = 5)
    #sdc.waveplot(cmap = 'binary')
    #sdc.histogram_noise()
    #sdc.test_gaussianity()
    wdc.fheader
