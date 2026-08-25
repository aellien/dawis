"""Entry points that turn an image into a wavelet cube.

The thin layer between the raw à trous kernels in
:mod:`dawis.wavelets.atrous` and the rest of DAWIS. A kernel returns plain
arrays; these functions name the wavelet family and wrap the result in the
cube the pipeline passes around.

:func:`bspl_atrous` is the one the pipeline runs.
"""

from __future__ import annotations

import numpy as np

from dawis.cubes import CoarseCube, WaveletCube
from dawis.wavelets.atrous import atrous, ser_iuwt_decomposition


def bspl_atrous(
    image: np.ndarray,
    n_levels: int,
    fheader=None,
    conditions: str = 'loop',
    verbose: bool = False,
) -> WaveletCube:
    """Decompose an image on the B3-spline à trous basis.

    The decomposition the pipeline runs at every iteration. The coarse plane
    is computed but not returned: the caller works on the wavelet planes
    alone.

    Parameters
    ----------
    image : numpy.ndarray
        The 2D image to decompose.
    n_levels : int
        Number of wavelet planes to compute.
    fheader : astropy.io.fits.Header, optional
        FITS header carried onto the cube, so the cube can be written back
        out with the astrometry of the image it came from. Default is None.
    conditions : {'loop', 'prolongation', 'mirror'}, optional
        Unused on this path: :func:`~dawis.wavelets.atrous.ser_iuwt_decomposition`
        always mirrors at the border. Kept because the pipeline threads a
        boundary condition through to every transform. Default is ``'loop'``.
    verbose : bool, optional
        Unused. Kept for call-site compatibility. Default is False.

    Returns
    -------
    WaveletCube
        The wavelet planes, of shape ``(nx, ny, n_levels)``, tagged
        ``'BSPL'``.
    """

    #filter = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
    #coarse_array, wavelet_array = atrous(image = image, n_levels = n_levels, filter = filter, conditions = 'loop')
    wavelet_array, coarse_array = ser_iuwt_decomposition(image, n_levels, 0, store_smoothed = True)
    #coarse_dc = CoarseCube(array = coarse_array, wavelet_type = 'BSPL', fheader = fheader)
    wavelet_dc = WaveletCube(array = wavelet_array, wavelet_type = 'BSPL', fheader = fheader)

    return wavelet_dc


def haar_atrous(
    image: np.ndarray,
    n_levels: int,
    fheader=None,
    verbose: bool = False,
) -> tuple[CoarseCube, WaveletCube]:
    """Decompose an image on the Haar à trous basis.

    Offered as an alternative basis to :func:`bspl_atrous`. The pipeline does
    not call it.

    Parameters
    ----------
    image : numpy.ndarray
        The 2D image to decompose.
    n_levels : int
        Number of wavelet planes to compute.
    fheader : astropy.io.fits.Header, optional
        FITS header carried onto both cubes. Default is None.
    verbose : bool, optional
        Unused. Kept for call-site compatibility. Default is False.

    Returns
    -------
    coarse_dc : CoarseCube
        The smoothed planes, of shape ``(nx, ny, n_levels + 1)``, tagged
        ``'HAAR'``.
    wavelet_dc : WaveletCube
        The wavelet planes, of shape ``(nx, ny, n_levels)``, tagged
        ``'HAAR'``.
    """

    filter = 1 / 2. * np.array([ 1, 0, 1 ])
    coarse_array, wavelet_array = atrous(image = image, n_levels = n_levels, filter = filter, conditions = 'loop')

    coarse_dc = CoarseCube(array = coarse_array, wavelet_type = 'HAAR', fheader = fheader)
    wavelet_dc = WaveletCube(array = wavelet_array, wavelet_type = 'HAAR', fheader = fheader)

    return coarse_dc, wavelet_dc


def max_levels(shape: tuple[int, ...]) -> int:
    """Deepest à trous decomposition an image of this shape supports.

    Each wavelet plane doubles the size of the structure it is sensitive to, so
    plane ``n`` describes structure about ``2**n`` pixels across. Past the point
    where that exceeds the image there is nothing left to describe, and this is
    that point: the largest ``n`` with ``2**n`` no wider than the shortest side.

    Parameters
    ----------
    shape : tuple of int
        Shape of the image, as ``numpy.ndarray.shape``.

    Returns
    -------
    int
        Number of wavelet planes, the ``n_levels`` the transforms take.

    Examples
    --------
    >>> max_levels((900, 900))
    9
    >>> max_levels((256, 256))
    8
    >>> max_levels((900, 256))
    8
    """

    return int(np.min(np.floor(np.log2(shape))))
