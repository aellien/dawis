"""Richardson-Lucy deconvolution and the B3-spline kernel it deconvolves with.

The optional PSF step of the synthesis. When
:meth:`~dawis.trees.construction.interscale_tree.CG_minimization` runs with
``deconv = True`` it treats the à trous B3-spline kernel as the point spread
function of the tree it is reconstructing, and runs ten Richardson-Lucy
iterations against it. :func:`twod_B2spline` builds that kernel — at order 0
for a tree whose interscale maximum sits at or below level 2, and at order 2
above it — and :func:`richardson_lucy_np` does the deconvolution.

Only the ADJOINT branch of the solver uses it. Under the SUM operator the
kernel is still built and then never read.

:func:`richardson_lucy_np` is a general routine and not specific to dawis: it
takes any image and any PSF, handles non-circulant edges by padding, and can
mask pixels out of the estimate. :func:`pad` and :func:`unpad` are its centred
padding helpers.

Everything here is pure numpy on float64 — the NumPy FFT casts 32-bit input up
regardless, so passing float32 to save memory does not work.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.fft import fftn, ifftn, ifftshift

def pad(
    img: np.ndarray,
    paddedsize: Sequence[int],
    mode: str,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Pad an array up to a larger shape, centring the original in it.

    The extra rows and columns are split between the two sides of each axis;
    an odd difference puts the larger half before the array.

    Parameters
    ----------
    img : numpy.ndarray
        Array to pad.
    paddedsize : sequence of int
        Target shape. Must have one entry per axis of ``img``, each at least
        as large as the corresponding axis.
    mode : str
        Padding mode, passed straight to :func:`numpy.pad`.

    Returns
    -------
    padded : numpy.ndarray
        The padded array, of shape ``paddedsize``.
    padding : tuple of tuple of int
        Widths added before and after along each axis, in the form
        :func:`numpy.pad` takes. Kept so the result can be cropped back.

    See Also
    --------
    unpad : Crops a padded array back to its original shape.
    """
    padding = tuple(map(lambda i,j: ( math.ceil((i-j)/2), math.floor((i-j)/2) ),paddedsize,img.shape))
    return np.pad(img, padding,mode), padding


def unpad(padded: np.ndarray, imgsize: Sequence[int]) -> np.ndarray:
    """Crop a centre-padded array back to its original shape.

    The inverse of :func:`pad`, recomputed from the two shapes rather than
    from the widths ``pad`` returned.

    Parameters
    ----------
    padded : numpy.ndarray
        Array padded by :func:`pad`.
    imgsize : sequence of int
        Shape to crop back to, one entry per axis.

    Returns
    -------
    numpy.ndarray
        View of the centre of ``padded``, of shape ``imgsize``.

    See Also
    --------
    pad : The padding this undoes.
    """
    padding = tuple(map(lambda i,j: (math.ceil((i-j)/2), math.floor((i-j)/2)), padded.shape, imgsize))
    slices = tuple(slice(p[0], p[0]+s) for p, s in zip(padding, imgsize))

    return padded[slices]

def richardson_lucy_np(
    image: np.ndarray,
    psf: np.ndarray,
    num_iters: int,
    noncirc: bool = False,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Deconvolve an image with the Richardson-Lucy algorithm.

    Multiplicative expectation-maximisation for Poisson noise: the current
    estimate is reblurred with the PSF, the ratio of the measured image to
    that reblur is correlated back with the PSF, and the estimate is scaled by
    the result. Repeated ``num_iters`` times, with no early exit.

    Parameters
    ----------
    image : numpy.ndarray
        Image to deconvolve.
    psf : numpy.ndarray
        Point spread function. Padded up to the shape of ``image`` when the
        two differ and ``noncirc`` is False.
    num_iters : int
        Number of iterations to run.
    noncirc : bool, optional
        If True, pad image and PSF out by half a PSF width so the convolutions
        do not wrap around the edges, and crop back at the end. Default False,
        which treats the image as periodic.
    mask : numpy.ndarray, optional
        Multiplicative mask over ``image``. Zeroed pixels take no part in the
        deconvolution and get their original values back afterwards. Default
        None, which uses every pixel.

    Returns
    -------
    numpy.ndarray
        The deconvolved image, of the shape of ``image``.

    Notes
    -----
    The NumPy FFT always casts 32-bit arrays to float64, so passing float32 in
    to save memory does not work.
    """
    
    # if noncirc==False and (image.shape != psf.shape) then pad the psf
    if noncirc==False and (image.shape != psf.shape):
        print('padding psf')
        psf,_=pad(psf, image.shape, 'constant')
    
    HTones = np.ones_like(image)

    if (mask is not None):
        HTones = HTones * mask
        mask_values = image*(1-mask)
        image=image*mask
    
    # if noncirc==True then pad the image, psf and HTOnes array to the extended size
    if noncirc:
        # compute the extended size of the image and psf
        extended_size = [image.shape[i]+2*int(psf.shape[i]/2) for i in range(len(image.shape))]

        # pad the image, psf and HTOnes array to the extended size computed above
        original_size = image.shape
        image,_=pad(image, extended_size, 'constant')
        HTones,_=pad(HTones, extended_size, 'constant')
        psf,_=pad(psf, extended_size, 'constant')
    

    otf = fftn(ifftshift(psf))
    otf_ = np.conjugate(otf)

    if noncirc:
        estimate = np.ones_like(image)*np.mean(image)
    else:
        estimate = image

    HTones = np.real(np.fft.ifftn(np.fft.fftn(HTones) * otf_))
    HTones[HTones<1e-6] = 1

    for i in range(num_iters):
        
        reblurred = np.real(ifftn(fftn(estimate) * otf))

        ratio = image / (reblurred + 1e-12)
        correction=np.real((np.fft.ifftn(np.fft.fftn(ratio) * otf_)))
        estimate = estimate * correction/HTones        
    
    if noncirc:
        estimate = unpad(estimate, original_size)

    if (mask is not None):
        estimate = estimate*mask + mask_values
    
    return estimate

def B2spline(xin: np.ndarray, order: int = 0) -> np.ndarray:
    """Sample the 1D B3-spline scaling function at a given scale.

    The low-band filter of the à trous transform in continuous form. Raising
    ``order`` by one doubles the width of the kernel, matching the hole
    spacing of the corresponding wavelet plane.

    Parameters
    ----------
    xin : numpy.ndarray
        Positions to sample at, in pixels, centred on zero.
    order : int, optional
        Scale index. The abscissae are divided by ``2 ** order`` before
        sampling. Default 0, the finest scale.

    Returns
    -------
    numpy.ndarray
        The spline sampled at ``xin``, of the shape of ``xin``.

    See Also
    --------
    twod_B2spline : The separable 2D kernel built from this.
    """
    x = xin / 2**order
    return ( abs(x-2)**3 - 4*abs(x-1)**3 + 6 * abs(x)**3 - 4 * abs(x+1)**3 + abs(x+2)**3 ) / 16.

def twod_B2spline(xsize: int, ysize: int, order: int = 0) -> np.ndarray:
    """Build the 2D B3-spline kernel used as a PSF by the deconvolution.

    The outer product of :func:`B2spline` along each axis, centred in an array
    of the requested shape. Passed to :func:`richardson_lucy_np` as the point
    spread function of one wavelet plane.

    Parameters
    ----------
    xsize : int
        Width of the kernel, in pixels.
    ysize : int
        Height of the kernel, in pixels.
    order : int, optional
        Scale index, forwarded to :func:`B2spline`. Default 0.

    Returns
    -------
    numpy.ndarray
        Kernel of shape ``(xsize, ysize)``, not normalised to unit sum.

    See Also
    --------
    B2spline : The 1D spline this is the outer product of.
    """
    
    xc = int(xsize / 2.)
    yc = int(ysize / 2.)
    
    xa = np.linspace(- xc, xc, xsize )
    ya = np.linspace(- yc, yc, ysize )
    
    b2s_x = B2spline(xa, order = order)
    b2s_y = B2spline(ya, order = order)
    
    b2s_2d = np.outer(b2s_x, b2s_y)
    
    return b2s_2d
    