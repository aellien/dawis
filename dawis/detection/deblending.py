"""Multiscale detection and deblending, in one pass.

The detector the pipeline runs. It replaces the
:mod:`~dawis.detection.thresholding` plus :mod:`~dawis.detection.labelling`
pair with a single sweep built on photutils: each wavelet plane is segmented
above its own threshold, then segments that merge two peaks are split apart.

Deblending is what stops a bright galaxy and its faint neighbour from entering
the tree builder as one region. It can be switched off at the finest scales,
where the noise makes splitting meaningless, through ``lvl_deblend``.

The function also returns the per-scale detection uncertainty, which the
restoration carries onto every atom it synthesises.
"""

from __future__ import annotations

import logging

import numpy as np
from astropy.stats import sigma_clip
from photutils.segmentation import deblend_sources, detect_sources

from dawis.cubes import LabelCube, WaveletCube
from dawis.detection.noise import table_noise_stds

log = logging.getLogger(__name__)


def ms_detect_and_deblend(
    wavelet_cube: WaveletCube,
    n_sigmas: float = 3,
    wavelet_type: str = 'BSPL',
    lvl_deblend: int = 0,
    npixels: int = 10,
    deblend_contrast: float = 0.3,
    verbose: bool = False,
) -> tuple[LabelCube, np.ndarray]:
    """Detect and deblend regions on every plane of a wavelet cube.

    The noise spread is measured once by sigma-clipping the finest plane, and
    the threshold at scale ``l`` is ``n_sigmas * sigma * conv_table[l]``.
    Each plane is segmented above that threshold, keeping only segments of at
    least ``npixels`` connected pixels, and then deblended. As in
    :func:`~dawis.detection.labelling.label_regions`, labels are offset per
    scale so they are unique across the whole cube.

    Parameters
    ----------
    wavelet_cube : dawis.cubes.WaveletCube
        The wavelet planes to detect on.
    n_sigmas : float, optional
        Detection threshold in units of the image noise spread. Also the
        clipping bound used to measure that spread. Default is 3.
    wavelet_type : str, optional
        Wavelet family naming the conversion table. Default is ``'BSPL'``.
        Unlike :func:`~dawis.detection.thresholding.hard_threshold`, this is
        never read off the cube header.
    lvl_deblend : int, optional
        First scale index at which deblending is applied. Scales below it are
        detected but not split. Default is 0, deblending everywhere.
    npixels : int, optional
        Minimum number of connected pixels for a detection to survive, passed
        to both detection and deblending. Default is 10.
    deblend_contrast : float, optional
        Fraction of the total source flux a local peak must hold to be split
        off as its own region. Default is 0.3.
    verbose : bool, optional
        Log the region count and label range of each scale. Default is False.

    Returns
    -------
    label_cube : dawis.cubes.LabelCube
        Integer cube of labels, 0 outside any region, carrying the per-scale
        region counts.
    detection_errors : numpy.ndarray
        One value per scale, ``sigma * conv_table[l]`` — the 1-sigma noise
        spread of that plane, taken as the uncertainty on anything detected
        there.

    Notes
    -----
    Unlike :func:`~dawis.detection.thresholding.hard_threshold`, zero-valued
    pixels are kept in the noise sample, so a padded or masked border biases
    the estimate low.

    Deblending runs single-process (``nproc = 1``) whatever the pipeline's
    worker count, so detection is unaffected by the parallelism seam.
    """

    if verbose == True:
        log.info('Labelling regions')
    
    x_axis, y_axis, n_levels = wavelet_cube.shape
    
    conv_table = table_noise_stds(n_levels, wavelet_type = wavelet_type)
    noise_pixels = np.copy(wavelet_cube.array[:,:,0]).flatten()
    noise_pixels = sigma_clip(noise_pixels, sigma = n_sigmas, sigma_lower = n_sigmas, masked = False)
    sigma = np.std(noise_pixels)
    
    # Detection error
    det_err_array = []
        
    # Labels 
    label_array = np.zeros(wavelet_cube.shape)
    lab_counts = np.zeros(wavelet_cube.z_size)
        
    for level in range(0, n_levels):
            
        threshold = n_sigmas * sigma * conv_table[level]
        det_err_array.append(sigma * conv_table[level]) # uncertainty is taken as 1-sigma std of noise at given wavelet scale
        segment_map = detect_sources(wavelet_cube.array[:,:,level], threshold, npixels = npixels)
        
        if level >= lvl_deblend:
            if segment_map:
                segment_map = deblend_sources(wavelet_cube.array[:,:,level], segment_map, npixels = npixels, contrast = deblend_contrast, nproc = 1)
        
        
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
            
    return LabelCube(label_array, lab_counts, fheader = wavelet_cube.fheader), np.array(det_err_array)
