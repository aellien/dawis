#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# This file is part of DAWIS.
#
# NAME : inpainting.py
# AUTHOR : Amael Ellien
# LAST MODIFICATION : 02/2024
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import logging
from astropy.stats import sigma_clip
from skimage.feature import peak_local_max

def sample_noise(im, n_sigmas = 3, bins = 100):
    '''
    Sample the noise distribution of an image. A sigma clipping algorithm is
    first applied, before the peak of the distribution is determined. Only values
    lower than the peak are kept and miroired to get a noise distribution without
    any source signal.

    Parameters
    ----------
    im : TYPE
        DESCRIPTION.
    n_sigmas : TYPE, optional
        DESCRIPTION. The default is 3.
    bins : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    None.

    '''    
    noise_pixels = sigma_clip(im, sigma = n_sigmas, sigma_lower = n_sigmas, masked = False)
    hist = np.histogram(noise_pixels, bins = bins, range = (noise_pixels.min(), noise_pixels.max()))
    idmax = peak_local_max( hist[0], min_distance = 1, num_peaks = 1, exclude_border = False)[0][0]
    valmax = hist[1][idmax]
    noise_pixels = noise_pixels[ noise_pixels <= valmax ]
    noise_pixels = np.append( noise_pixels, noise_pixels + 2 * ( valmax - noise_pixels ) )
    return noise_pixels, valmax


def inpaint_with_gaussian_noise(image, mean, sigma, iptd_sigma):
    '''
    Inpaint all pixels with values lower than iptd_sigma * sigma with values drawn
    from a normal ditribution of standard deviation sigma and centered on 0. 
    /!\ Assumes sky background is removed.

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    mean : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    iptd_sigma : TYPE
        DESCRIPTION.

    Returns
    -------
    image : TYPE
        DESCRIPTION.

    '''
    mask = np.zeros(image.shape)
    mask[ image < -abs(iptd_sigma * sigma) ] = 1.
    mask[np.where(np.isnan(image) == True)] = 1.
    draws = np.random.normal(0, sigma, image.shape)
    mask *= draws
    image[ image < -abs(iptd_sigma * sigma) ] = 0.
    image[ np.where(np.isnan(image) == True)] = 0.
    image += mask
    
    return image