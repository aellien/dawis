"""Hard thresholding of the wavelet cube into a significance support.

The first detection step. Noise is measured once, on the finest wavelet plane
where structure is faintest, and turned into a per-scale threshold through the
conversion table in :mod:`dawis.detection.noise`. Every coefficient above its
scale's threshold is marked significant; the result is a binary cube — the
*multiscale support* — with the same shape as the wavelet cube.

This is the simple detector. :mod:`dawis.detection.deblending` is the one the
pipeline runs by default; it detects and separates touching sources in the
same pass.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import normaltest, sigmaclip

from dawis.cubes import support_datacube, wavelet_datacube
from dawis.detection.noise import table_noise_stds

try:
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib is the optional [viz] extra
    from dawis._optional import MissingDependency
    plt = MissingDependency('matplotlib', 'viz')


def hard_threshold(
    wavelet_datacube: wavelet_datacube,
    n_sigmas: float = 3,
    wavelet_type: str | None = None,
    keep_neg: bool = False,
    verbose: bool = False,
    display: bool = False,
) -> support_datacube:
    """Threshold a wavelet cube into a binary significance support.

    The noise spread ``sigma`` is estimated once, by sigma-clipping the
    finest wavelet plane. The threshold applied at scale ``l`` is then
    ``n_sigmas * sigma * conv_table[l]``, where the conversion table accounts
    for the shrinking of noise with scale.

    Parameters
    ----------
    wavelet_datacube : dawis.cubes.wavelet_datacube
        The wavelet planes to threshold.
    n_sigmas : float, optional
        Detection threshold in units of the image noise spread. Also the
        clipping bound used to measure that spread. Default is 3.
    wavelet_type : str, optional
        Wavelet family naming the conversion table. Default is None, which
        reads ``WAVTYPE`` off the cube's FITS header.
    keep_neg : bool, optional
        If True, threshold on ``abs(coefficient)``, so significant negative
        coefficients are kept. Default is False, which keeps positive
        detections only.
    verbose : bool, optional
        Log the threshold applied at each scale. Default is False.
    display : bool, optional
        Show a histogram of the clipped noise pixels. Default is False.

    Returns
    -------
    dawis.cubes.support_datacube
        Binary cube, 1 where the coefficient is significant, tagged
        ``'HARD'`` and carrying the clipped noise pixels used for the
        estimate.

    Notes
    -----
    Zero-valued pixels are dropped from the noise sample before clipping, so
    a masked or padded border does not pull the estimate down.

    A Gaussianity test is run on the noise sample and a warning logged,
    whatever ``verbose`` is. The test is reported, never enforced — and the
    warning is inverted: ``normaltest`` takes normality as its null
    hypothesis, so a *large* p-value means the sample does look Gaussian,
    yet that is exactly when the warning fires. Read it as the opposite of
    what it says. The comparison is left as published rather than corrected
    inside a move.

    Two more defects in the same block are carried through unchanged. The
    guard ``noise_pixels[0].size > 0`` inspects a scalar, so it is always
    true, and on a plane whose sample clips away to nothing the indexing
    raises ``IndexError`` before the guard can do anything. The ``display``
    path histograms that same scalar rather than the sample, so the plot it
    draws is empty of information; it is unreachable from the pipeline,
    which never sets ``display``.
    """

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

    noise_pixels = np.copy(wavelet_datacube.array[:,:,0]).flatten()
    noise_pixels = np.delete(noise_pixels, np.where(noise_pixels == 0) )
    noise_pixels = sigmaclip(noise_pixels, low = n_sigmas, high = n_sigmas)[0]
    sigma = np.std(noise_pixels)

    alpha = 1E-3
    if noise_pixels[0].size > 0:
        k, pval = normaltest( noise_pixels )
        if pval > alpha:
            log = logging.getLogger(__name__)
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
