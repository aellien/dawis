"""Inpainting: putting noise back where the residual has gone negative.

The pipeline subtracts each atom from a running residual, and a slightly
over-restored atom digs a hole. Those holes are not noise, and the detection
of the next iteration would read them as structure, so before each iteration
the residual is inpainted: every pixel below a negative threshold, and every
NaN, is replaced with a fresh draw from the noise distribution of the image.

Two steps. :func:`sample_noise` measures that distribution from the image
itself, by keeping only the pixels below the peak of the sigma-clipped
histogram and mirroring them about it — the half of the histogram that source
flux cannot reach. :func:`inpaint_with_gaussian_noise` does the replacement
from a Gaussian of the width that measurement produced.

Both assume the sky background has already been removed, so that the noise is
centred on zero.
"""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clip
from skimage.feature import peak_local_max

def sample_noise(
    im: np.ndarray,
    n_sigmas: float = 3,
    bins: int = 100,
) -> tuple[np.ndarray, float]:
    """Sample the noise distribution of an image, free of source signal.

    The image is sigma-clipped, and the peak of the histogram of what survives
    is taken as the centre of the noise. Only pixels at or below that peak are
    kept — the side of the distribution that source flux cannot push into —
    and they are mirrored about it to rebuild a symmetric distribution.

    Parameters
    ----------
    im : numpy.ndarray
        Image to measure. Normally the residual, with the sky removed.
    n_sigmas : float, optional
        Clipping threshold, applied on both sides. Default 3.
    bins : int, optional
        Number of histogram bins used to locate the peak. Default 100.

    Returns
    -------
    noise_pixels : numpy.ndarray
        1D array of noise samples, symmetric about ``valmax``. Twice the
        length of the kept half.
    valmax : float
        Value of the histogram peak, the centre of the noise distribution.

    See Also
    --------
    inpaint_with_gaussian_noise : Consumes the width measured from this.
    """
    noise_pixels = sigma_clip(im, sigma = n_sigmas, sigma_lower = n_sigmas, masked = False)
    hist = np.histogram(noise_pixels, bins = bins, range = (noise_pixels.min(), noise_pixels.max()))
    idmax = peak_local_max( hist[0], min_distance = 1, num_peaks = 1, exclude_border = False)[0][0]
    valmax = hist[1][idmax]
    noise_pixels = noise_pixels[ noise_pixels <= valmax ]
    noise_pixels = np.append( noise_pixels, noise_pixels + 2 * ( valmax - noise_pixels ) )
    return noise_pixels, valmax


def inpaint_with_gaussian_noise(
    image: np.ndarray,
    mean: float,
    sigma: float,
    iptd_sigma: float,
) -> np.ndarray:
    """Replace deep negative pixels and NaNs with Gaussian noise.

    Every pixel below ``-abs(iptd_sigma * sigma)``, and every NaN, is
    overwritten with a draw from a normal distribution of width ``sigma``
    centred on zero. Pixels above the threshold are left untouched.

    The image is modified in place as well as returned.

    Parameters
    ----------
    image : numpy.ndarray
        Image to inpaint, normally the residual. Modified in place.
    mean : float
        Centre of the noise distribution. Not used; see Notes.
    sigma : float
        Standard deviation of the noise, as measured by :func:`sample_noise`.
    iptd_sigma : float
        Threshold in units of ``sigma``. Pixels below minus this many sigmas
        are inpainted.

    Returns
    -------
    numpy.ndarray
        The inpainted image, the same object that was passed in.

    See Also
    --------
    sample_noise : Measures the distribution this draws from.

    Notes
    -----
    Assumes the sky background has been removed, so that zero is the centre of
    the noise. ``mean`` is accepted for that reason and is not read: the draws
    are always centred on zero.
    """
    mask = np.zeros(image.shape)
    mask[ image < -abs(iptd_sigma * sigma) ] = 1.
    mask[np.where(np.isnan(image) == True)] = 1.
    draws = np.random.normal(0, sigma, image.shape)
    mask *= draws
    image[ image < -abs(iptd_sigma * sigma) ] = 0.
    image[ np.where(np.isnan(image) == True)] = 0.
    image += mask
    
    return image

def subtract_background(
    image: np.ndarray,
    n_sigmas: float = 3,
    bins: int = 200,
) -> tuple[np.ndarray, float, float]:
    """Measure the sky background of an image and take it out.

    Everything downstream assumes the noise is centred on zero — the
    inpainting draws are, the wavelet thresholds are quoted in sigmas of it —
    so this is the first thing a run does to its image.

    The image is modified in place as well as returned, which is what keeps its
    dtype: subtracting a float64 mean out of place would promote a float32
    image and move every number that follows.

    Parameters
    ----------
    image : numpy.ndarray
        The image to level. Modified in place.
    n_sigmas : float, optional
        Clipping threshold handed to :func:`sample_noise`. Default 3.
    bins : int, optional
        Histogram bins handed to :func:`sample_noise`. Default 200.

    Returns
    -------
    image : numpy.ndarray
        The image with the background removed — the same object that was
        passed in.
    mean : float
        The background that was subtracted.
    sigma : float
        Width of the noise distribution, which the inpainting draws from and
        the detection thresholds are quoted in.

    See Also
    --------
    sample_noise : Measures the distribution these two numbers describe.
    """

    noise_pixels, _ = sample_noise(image, n_sigmas = n_sigmas, bins = bins)
    mean, sigma = np.mean(noise_pixels), np.std(noise_pixels)
    image -= mean

    return image, mean, sigma
