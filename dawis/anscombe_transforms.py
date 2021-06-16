#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# This file is part of DAWIS.
#
# name : anscombe_transform.py
# author : amael ellien
# last modification : 07/2021
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# anscombe transform of an image.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
from scipy.stats import sigmaclip
import logging

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def pg_noise_bissection(image, max_err = 1E-6, n_sigmas = 3, verbose = False):
    '''
    doc tot do    '''

    noise_pts = np.copy(image)
    noise_pixels = sigmaclip(image, low = n_sigmas, high = n_sigmas)[0] # the max parameters are given

    sigma_max = np.std(noise_pixels)           # by the parameters of a gaussian
    mean_max = np.mean(noise_pixels)              # noise estimation on the input
    gain_max = np.max(noise_pixels) - np.min(noise_pixels)              # image.

    sigma_min = 0.0          # the min parameters are given
    mean_min = 0.0           # by 0.
    gain_min = 0.0           #

    error = 1.0    # initial parameters for the
    step = 0       # bissection-like method.

    if verbose == True:
        log = logging.getLogger(__name__)
        log.info('bissection-like noise estimation')

    while error > max_err:

        sigma = ( sigma_min + sigma_max ) / 2.0 # the noise parameters for this
        mean = ( mean_min + mean_max ) / 2.0    # iteration are the mean of max
        gain = ( gain_min + gain_max ) / 2.0    # and min noise parameters.

        sigma_ansc = np.std(sigmaclip(anscombe_transform(noise_pts, sigma = sigma, mean = mean, gain = gain), \
                                        low = n_sigmas, high = n_sigmas)[0])

        if sigma_ansc <= 1.0:  # gaussian distribution countepart
            sigma_max = sigma  # overestimated.
            gain_max = gain
            mean_max = mean
        else:
            sigma_min = sigma # gaussian distribution
            gain_min = gain   # underestimated.
            mean_min = mean

        error = sigma_max - sigma_min # convergence of the algorithm.
        step += 1

        if verbose == True:
            log.info('step %02d sigma_ansc %f sigma %f mean %f gain %f' %(step,sigma_ansc,sigma,mean,gain))
    if verbose == True:
        log.info('done.')

    return sigma, mean, gain


def anscombe_transform(image, sigma = 0.0, mean = 0.0, gain = 1.0):

    '''
    apply anscombe transform to input data. doc to do
    '''

    out_image = np.copy( image )
    out_image = ( 2.0 / gain ) * np.sqrt( gain * out_image + ( 3. / 8. ) \
                        * ( gain ** 2 ) + ( sigma ** 2 ) - gain * mean )
    out_image[ np.where( np.isnan( out_image ) == True ) ] = 0.0

    return out_image

if __name__ == '__main__':

    from atrous import atrous
    from numpy.random import normal
    from astropy.io import fits
    from wavelet_transforms import bspl_atrous, wavelet_datacube
    from hard_threshold import hard_threshold
    from label_regions import label_regions

    hdu = fits.open('/home/ellien/wavelets/A1365.rebin.fits')
    im = hdu[0].data
    header = hdu[0].header

    sigma, mean, gain = pg_noise_bissection(im, max_err = 1E-6, n_sigmas = 3)
    aim = anscombe_transform(im, sigma, mean, gain)
    acdc, awdc = bspl_atrous(aim, 10, header)
    sdc = hard_threshold(awdc, n_sigmas = 5)
    ldc = label_regions(sdc)

    cdc, wdc = bspl_atrous(im, 10, header)

    acdc.to_fits('/home/ellien/devd/tests/A1365.rebin.acdc.fits', overwrite = True)
    cdc.to_fits('/home/ellien/devd/tests/A1365.rebin.cdc.fits', overwrite = True)
    awdc.to_fits('/home/ellien/devd/tests/A1365.rebin.awdc.fits', overwrite = True)
    wdc.to_fits('/home/ellien/devd/tests/A1365.rebin.wdc.fits', overwrite = True)
    sdc.to_fits('/home/ellien/devd/tests/A1365.rebin.sdc.fits', overwrite = True)
    ldc.to_fits('/home/ellien/devd/tests/A1365.rebin.ldc.fits', overwrite = True)
