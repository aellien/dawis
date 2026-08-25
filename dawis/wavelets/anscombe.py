"""The Anscombe variance-stabilising transform and its inverse.

Astronomical images carry Poisson-Gaussian noise, whose spread grows with the
signal. Every threshold in DAWIS is expressed in units of a single noise
sigma, which only means something if the spread is the same everywhere. The
Anscombe transform maps the image onto a domain where the noise has constant
unit spread, so one threshold is valid at every flux level.

:func:`pg_noise_bissection` estimates the three noise parameters the
transform needs, :func:`anscombe_transform` applies it, and
:func:`inverse_anscombe` undoes it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clip


def pg_noise_bissection(
    image: np.ndarray,
    max_err: float = 1E-6,
    n_sigmas: int = 3,
    verbose: bool = False,
    rng: np.random.RandomState | None = None,
) -> tuple[float, float, float]:
    """Estimate the Poisson-Gaussian noise parameters of an image.

    The three parameters ``sigma``, ``mean`` and ``gain`` are not measured
    directly. They are searched for: the triple is correct when applying
    :func:`anscombe_transform` with it leaves noise of unit standard
    deviation. The search is a bisection, halving the bracket at every step
    and moving the bound that the unit test lands on.

    Parameters
    ----------
    image : numpy.ndarray
        Image to estimate the noise of. Its noise pixels are isolated by
        sigma clipping, which assumes the image is mostly background.
    max_err : float, optional
        Stop once the bracket on ``sigma`` is narrower than this. Default is
        1e-6.
    n_sigmas : int, optional
        Clipping threshold used to separate noise pixels from signal, in
        standard deviations. Default is 3.
    verbose : bool, optional
        Log the bracket at every step. Default is False.
    rng : numpy.random.RandomState, optional
        Generator handed to every :func:`anscombe_transform` this search
        makes — one per bisection step, each drawing its own NaN fill. Those
        draws are part of a run's random stream, so a run threads its one
        generator in here as well. Default None uses the global
        :mod:`numpy.random` state.

    Returns
    -------
    sigma : float
        Standard deviation of the Gaussian read-noise component.
    mean : float
        Offset of the Gaussian component.
    gain : float
        Conversion between counts and the Poisson component.

    Notes
    -----
    The bracket is seeded from a plain Gaussian estimate on the clipped
    pixels, so the search only converges on an image whose noise those bounds
    actually enclose.
    """

    noise_pts = np.copy(image)
    noise_pixels = sigma_clip(image, sigma = n_sigmas, sigma_lower = n_sigmas, masked = False) # the max parameters are given
    
    sigma_max = np.nanstd(noise_pixels)           # by the parameters of a gaussian
    mean_max = np.nanmean(noise_pixels)              # noise estimation on the input
    gain_max = np.nanmax(noise_pixels) - np.nanmin(noise_pixels)              # image.

    sigma_min = 0
    mean_min = np.nanmin(noise_pixels) 
    gain_min = 0.

    error = 1.0    # initial parameters for the
    step = 0       # bissection-like method.

    if verbose == True:
        log = logging.getLogger(__name__)
        log.info('bissection-like noise estimation')

    while error > max_err:
        sigma = ( sigma_min + sigma_max ) / 2.0 # the noise parameters for this
        mean = ( mean_min + mean_max ) / 2.0    # iteration are the mean of max
        gain = ( gain_min + gain_max ) / 2.0    # and min noise parameters.
       
        sigma_ansc = np.nanstd(sigma_clip(anscombe_transform(noise_pts, sigma = sigma, mean = mean, gain = gain, rng = rng), \
                                        sigma = n_sigmas, sigma_lower = n_sigmas, masked = False))

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


def anscombe_transform(
    image: np.ndarray,
    sigma: float = 0.0,
    mean: float = 0.0,
    gain: float = 1.0,
    n_sigmas_clip: int = 3,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Stabilise the variance of a Poisson-Gaussian image.

    Applies

    .. math:: y = \\frac{2}{g}\\sqrt{g x + \\frac{3}{8}g^2 + s^2 - g m}

    where ``g``, ``s`` and ``m`` are the gain, sigma and mean returned by
    :func:`pg_noise_bissection`. Noise in ``y`` has unit standard deviation
    whatever the signal level in ``x``.

    Pixels whose argument under the square root is negative come back as NaN.
    Rather than propagate them, they are filled with draws from a Gaussian
    matched to the clipped statistics of the transformed image, so the output
    stays finite everywhere and the filled pixels look like background.

    Parameters
    ----------
    image : numpy.ndarray
        Image to transform. Not modified.
    sigma : float, optional
        Standard deviation of the Gaussian noise component. Default is 0.0.
    mean : float, optional
        Offset of the Gaussian noise component. Default is 0.0.
    gain : float, optional
        Poisson gain. Must not be zero. Default is 1.0.
    n_sigmas_clip : int, optional
        Clipping threshold, in standard deviations, used to measure the
        statistics the NaN fill is drawn from. Default is 3.
    rng : numpy.random.RandomState, optional
        Generator the NaN fill draws from. A run threads its own so the
        stabilisation is part of one reproducible stream (#53). Default None
        uses the global :mod:`numpy.random` state, which is what a caller
        using this transform on its own gets.

    Returns
    -------
    numpy.ndarray
        The stabilised image, same shape as the input.

    See Also
    --------
    inverse_anscombe : Undoes the transform.

    Notes
    -----
    Without ``rng`` the NaN fill uses the global numpy random state, so the
    function is only reproducible once that state has been seeded.
    """

    out_image = np.copy( image )
    out_image = ( 2.0 / gain ) * np.sqrt( gain * out_image + ( 3. / 8. ) \
                        * ( gain ** 2 ) + ( sigma ** 2 ) - gain * mean )
    clip_ans = sigma_clip(out_image, sigma = n_sigmas_clip, sigma_lower = n_sigmas_clip, masked = False)
    std_ans = np.nanstd(clip_ans)
    mean_ans = np.nanmean(clip_ans)
    mask = np.zeros(out_image.shape)
    mask[ np.where( np.isnan( out_image ) == True ) ] = 1
    draws = (np.random if rng is None else rng).normal(mean_ans, std_ans, mask.shape)
    mask *= draws
    out_image[ np.where( np.isnan( out_image ) == True ) ] = 0.0
    out_image += mask

    return out_image


def inverse_anscombe(
    image: np.ndarray,
    sigma: float = 0.0,
    mean: float = 0.0,
    gain: float = 1.0,
) -> np.ndarray:
    """Map a stabilised image back to the original flux domain.

    The algebraic inverse of :func:`anscombe_transform`

    .. math:: x = \\frac{1}{g}\\left(\\frac{g^2 y^2}{4}
                  - \\frac{3}{8}g^2 - s^2 + g m\\right)

    which is exact, and not the bias-corrected inverse usually recommended
    for denoised data. DAWIS restores atoms in the untransformed image, so
    the stabilised domain is only ever used for detection; nothing measured
    in it is mapped back.

    Parameters
    ----------
    image : numpy.ndarray
        A stabilised image, as returned by :func:`anscombe_transform`.
    sigma : float, optional
        Standard deviation of the Gaussian noise component. Must be the value
        the forward transform used. Default is 0.0.
    mean : float, optional
        Offset of the Gaussian noise component. Must match the forward
        transform. Default is 0.0.
    gain : float, optional
        Poisson gain. Must match the forward transform, and must not be zero.
        Default is 1.0.

    Returns
    -------
    numpy.ndarray
        The image in the original flux domain, same shape as the input.

    See Also
    --------
    anscombe_transform : The forward transform.

    Notes
    -----
    The roundtrip returns the input exactly only for pixels the forward
    transform left finite. Pixels it filled with noise draws come back as
    those draws, not as their original values.
    """

    return ( gain ** 2 * image ** 2 / 4.0
             - 3.0 * gain ** 2 / 8.0
             - sigma ** 2
             + gain * mean ) / gain


def _shifted(image: np.ndarray, offset: float) -> np.ndarray:
    """A copy of ``image`` with ``offset`` taken out of it, in place.

    In place on the copy, so the copy keeps the dtype of the image: an
    out-of-place subtraction of a float64 offset would promote a float32
    image and move every number computed after it.
    """

    shifted = np.copy(image)
    shifted -= offset

    return shifted


@dataclass(frozen = True)
class AnscombeStabilisation:
    """The variance-stabilising map of one run, measured once and reused.

    :func:`anscombe_transform` needs three noise parameters and a non-negative
    image, and both are properties of the image a run starts from rather than
    of the array handed to any one call. Measuring them per iteration would
    re-estimate the noise of a residual that the run is deliberately emptying,
    so a run fits this once, against its input image, and applies it to every
    residual afterwards.

    Parameters
    ----------
    offset : float
        Subtracted from an image before the transform, putting its minimum at
        zero. The transform takes a square root, so a negative pixel would
        return NaN.
    sigma, mean, gain : float
        The Poisson-Gaussian noise parameters, as
        :func:`pg_noise_bissection` searched them out.

    See Also
    --------
    pg_noise_bissection : The search :meth:`fit` runs.
    anscombe_transform : The map :meth:`transform` applies.

    Examples
    --------
    >>> stabilisation = AnscombeStabilisation.fit(image)     # doctest: +SKIP
    >>> stabilised = stabilisation.transform(residuals)      # doctest: +SKIP
    """

    offset: float
    sigma: float
    mean: float
    gain: float

    @classmethod
    def fit(
        cls,
        image: np.ndarray,
        max_err: float = 1E-4,
        n_sigmas: int = 3,
        rng: np.random.RandomState | None = None,
    ) -> AnscombeStabilisation:
        """Measure the map from the image a run starts from.

        Parameters
        ----------
        image : numpy.ndarray
            The image, with its sky background already removed. Not modified.
        max_err : float, optional
            Bisection tolerance handed to :func:`pg_noise_bissection`.
            Default 1e-4.
        n_sigmas : int, optional
            Clipping threshold handed to :func:`pg_noise_bissection`.
            Default 3.
        rng : numpy.random.RandomState, optional
            Generator handed to the bisection. Fitting draws — every step of
            the search stabilises the image and fills its NaNs — so a run
            threads its generator here too, before the loop starts. Default
            None uses the global :mod:`numpy.random` state.

        Returns
        -------
        AnscombeStabilisation
            The fitted map.
        """

        offset = np.nanmin(image)
        sigma, mean, gain = pg_noise_bissection(_shifted(image, offset),
                                                max_err = max_err,
                                                n_sigmas = n_sigmas,
                                                rng = rng)

        return cls(offset = offset, sigma = sigma, mean = mean, gain = gain)

    def transform(
        self,
        image: np.ndarray,
        rng: np.random.RandomState | None = None,
    ) -> np.ndarray:
        """Apply the fitted map to an image.

        Parameters
        ----------
        image : numpy.ndarray
            Image to stabilise, normally the run's current residual. Not
            modified.
        rng : numpy.random.RandomState, optional
            Generator the NaN fill draws from, threaded on to
            :func:`anscombe_transform`. Default None uses the global
            :mod:`numpy.random` state.

        Returns
        -------
        numpy.ndarray
            The stabilised image, whose noise has unit standard deviation at
            every flux level.
        """

        return anscombe_transform(_shifted(image, self.offset),
                                  self.sigma, self.mean, self.gain,
                                  rng = rng)
