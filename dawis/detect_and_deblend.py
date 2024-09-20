#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# This file is part of DAWIS.
#
# NAME : ms_detect_and_deblend.py
# AUTHOR : Amael Ellien
# LAST MODIFICATION : 02/2024
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import matplotlib.pyplot as plt
import logging
from dawis.table_noise_stds import table_noise_stds
from dawis.datacube import label_datacube
from astropy.stats import sigma_clip
from photutils.segmentation import detect_sources, deblend_sources

def ms_detect_and_deblend(wavelet_datacube, n_sigmas = 3, wavelet_type = 'BSPL', lvl_deblend = 0, npixels = 10, deblend_contrast = 0.3, verbose = False):
    '''
    Parameters
    ----------
    wavelet_datacube : TYPE
        DESCRIPTION.
    n_sigmas : TYPE, optional
        DESCRIPTION. The default is 3.
    wavelet_type : TYPE, optional
        DESCRIPTION. The default is 'BSPL'.
    lvl_deblend : TYPE, optional
        DESCRIPTION. The default is 0.
    npixels : TYPE, optional
        DESCRIPTION. The default is 10.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    if verbose == True:
        log = logging.getLogger(__name__)
        log.info('Labelling regions')
    
    x_axis, y_axis, n_levels = wavelet_datacube.shape
    
    conv_table = table_noise_stds(n_levels, wavelet_type = wavelet_type)
    noise_pixels = np.copy(wavelet_datacube.array[:,:,0]).flatten()
    noise_pixels = sigma_clip(noise_pixels, sigma = n_sigmas, sigma_lower = n_sigmas, masked = False)
    sigma = np.std(noise_pixels)
    
    # Detection error
    det_err_array = []
        
    # Labels 
    label_array = np.zeros(wavelet_datacube.shape)
    lab_counts = np.zeros(wavelet_datacube.z_size)
        
    for level in range(0, n_levels):
            
        threshold = n_sigmas * sigma * conv_table[level]
        det_err_array.append(sigma * conv_table[level]) # uncertainty is taken as 1-sigma std of noise at given wavelet scale
        segment_map = detect_sources(wavelet_datacube.array[:,:,level], threshold, npixels = npixels)
        
        if level >= lvl_deblend:
            if segment_map:
                segment_map = deblend_sources(wavelet_datacube.array[:,:,level], segment_map, npixels = npixels, contrast = deblend_contrast, nproc = 1)
        
        
        if segment_map:
            label_array[:,:,level] = segment_map.data
            lab_counts[level] = segment_map.labels.max()
            

        if level != 0:
            reg = np.where( label_array[:, :, level] != 0 ) 
            temp = label_array[:, :, level]
            temp[reg] = temp[reg] + np.sum( lab_counts[0:level] )
            label_array[:, :, level] = temp
        
        if verbose == True:
            log.info('level %02d nregions %d labels %d - %d'\
                    %(level+1,lab_counts[level],\
                    np.max(label_array[:,:,level])-lab_counts[level]+1,\
                    np.max(label_array[:,:,level])))
            
    return label_datacube(label_array, lab_counts, wavelet_datacube.fheader), np.array(det_err_array)

if __name__ == '__main__':

    from atrous import atrous
    from numpy.random import normal
    from astropy.io import fits
    from wavelet_transforms import bspl_atrous, wavelet_datacube
    from hard_threshold import hard_threshold

    hdu = fits.open('/home/aellien/test/dawis_test/data/star_test_H.fits')
    im = hdu[0].data
    header = hdu[0].header
    wdc = bspl_atrous(im, 10, header)
    ldc = ms_detect_and_deblend(wdc, 3, 'BSPL', 4, 10, True)
    ldc.label_counts_plot()
    ldc.waveplot()
