"""The à trous (starlet) transform, the kernel of the wavelet block.

Two decompositions live here. :func:`atrous` is the general one: it takes any
separable filter and returns both the coarse and the wavelet arrays. It is
still called from the interscale tree module, which reruns it on every atom
it synthesises. :func:`ser_iuwt_decomposition` is the fast B3-spline path
borrowed from PyMORESANE, and is what the pipeline runs on every iteration
through :func:`dawis.wavelets.transforms.bspl_atrous`.

Both are pure array in, array out. Nothing here knows about cubes.
"""

from __future__ import annotations

import numpy as np

from dawis.exceptions import (
    DawisDimensionError,
    DawisValueError,
)


def boundary_conditions(
    position_array: np.ndarray,
    sizes: int | np.ndarray,
    conditions: str = 'prolongation',
) -> np.ndarray:
    """Fold out-of-bounds sample positions back inside the image.

    The à trous filter reaches further from its centre at every scale, so at
    the coarse scales it samples well outside the image. This maps those
    positions back onto valid indices, in place.

    Parameters
    ----------
    position_array : numpy.ndarray
        Sample positions to fold, of dimension 1 or 3. A 3D array is the
        ``(2, ny, nx)`` meshgrid produced for a 2D filter; a 1D array is the
        offsets of a separable 1D filter along one axis. Modified in place.
    sizes : int or numpy.ndarray
        Image extent the positions must fall inside. A scalar for a 1D
        ``position_array``, or ``(nx, ny)`` for the 3D case.
    conditions : {'prolongation', 'mirror', 'loop'}, optional
        How to treat a position that falls outside. ``'prolongation'`` clamps
        it to the nearest edge, ``'mirror'`` reflects it back across the edge,
        and ``'loop'`` wraps it around to the opposite side. The pipeline runs
        ``'loop'``. Default is ``'prolongation'``.

    Returns
    -------
    numpy.ndarray
        ``position_array``, folded. The input array is modified in place and
        returned, not copied.

    Raises
    ------
    DawisDimensionError
        If ``position_array`` is not 1D or 3D, or ``sizes`` is not a scalar
        or 1D.
    TypeError
        If ``conditions`` is not a string.
    DawisValueError
        If ``conditions`` is not one of the three allowed values.
    """

    if np.ndim(position_array) not in [1, 3]:
        raise DawisDimensionError('Wrong dimensions for array -->', np.ndim(position_array), [1, 3])

    if np.ndim(sizes) not in [0, 1]:
        raise DawisDimensionError('Wrong dimensions for array -->', np.ndim(sizes), [0, 1])

    if type(conditions) != str:
        raise TypeError('Wrong type --> conditions is %s instead of str'
                        % type(conditions))

    allowed_conditions = ['mirror', 'prolongation', 'loop' ]

    if conditions not in allowed_conditions:
        raise DawisValueError('Not an allowed condition -->', conditions, allowed_conditions)

    if conditions == 'prolongation':

        position_array[np.where(position_array < 0)] = 0

        if ( np.ndim(position_array) == 3 ) & ( np.ndim(sizes) == 1 ):
            position_array[0, np.where(position_array[0,:] > sizes[0] - 1)] = sizes[0] - 1
            position_array[1, np.where(position_array[1,:] > sizes[1] - 1)] = sizes[1] - 1
        else:
            position_array[np.where(position_array > sizes - 1)] = sizes - 1

    if conditions == 'mirror':

        position_array[np.where(position_array < 0)] -= 2 * position_array[np.where(position_array < 0)]

        if ( np.ndim(position_array) == 3 ) & ( np.ndim(sizes) == 1 ):
            position_array[0][np.where(position_array[0,:] > sizes[0] - 1)] -= 2 * ( position_array[0][np.where(position_array[0,:] > sizes[0] - 1)] - sizes[0] )
            position_array[1][np.where(position_array[1,:] > sizes[1] - 1)] -= 2 * ( position_array[1][np.where(position_array[1,:] > sizes[1] - 1)] - sizes[1] )
        else:
            position_array[np.where(position_array > sizes - 1)] -= sizes - ( position_array[np.where(position_array > sizes - 1 )] - sizes )

    if conditions == 'loop':

        if ( np.ndim(position_array) == 3 ) & ( np.ndim(sizes) == 1 ):
            position_array[0][np.where(position_array[0,:] > sizes[0] - 1)] -= sizes[0]
            position_array[1][np.where(position_array[1,:] > sizes[1] - 1)] -= sizes[1]
            position_array[0][np.where(position_array[0,:] < 0)] += sizes[0]
            position_array[1][np.where(position_array[1,:] < 0)] += sizes[1]
        else:
            position_array[np.where(position_array > sizes - 1)] -= sizes
            position_array[np.where(position_array < 0)] += sizes

    return position_array

def atrous(
    image: np.ndarray,
    n_levels: int,
    filter: np.ndarray,
    conditions: str = 'loop',
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose an image with the à trous algorithm, for any filter.

    The image is smoothed repeatedly with the same filter, dilated by a
    factor of two at every level. Each wavelet plane is the difference
    between two successive smoothings, so the planes plus the last coarse
    plane sum back to the input exactly.

    Parameters
    ----------
    image : numpy.ndarray
        The 2D image to decompose.
    n_levels : int
        Number of wavelet planes to compute. Must be positive.
    filter : numpy.ndarray
        The scaling filter, 1D (applied separably along each axis) or 2D.
    conditions : {'loop', 'prolongation', 'mirror'}, optional
        Boundary treatment, passed to :func:`boundary_conditions`. Default is
        ``'loop'``.
    verbose : bool, optional
        Unused. Kept for call-site compatibility. Default is False.

    Returns
    -------
    coarse_array : numpy.ndarray
        Smoothed planes, of shape ``(nx, ny, n_levels + 1)`` and dtype
        float32. Index 0 is the input image; index ``n_levels`` is the
        coarsest plane.
    wavelet_array : numpy.ndarray
        Wavelet planes, of shape ``(nx, ny, n_levels)`` and dtype float32,
        from finest to coarsest.

    Raises
    ------
    DawisDimensionError
        If ``image`` is not 2D, or ``filter`` has more than two dimensions.
    DawisValueError
        If ``n_levels`` is not positive.

    See Also
    --------
    ser_iuwt_decomposition : The faster B3-spline-only path the pipeline runs.
    """

    #===========================================================================
    # Exceptions
    if image.ndim != 2:
        raise DawisDimensionError('Not an image -->', image.ndim, 2)
    image_x_size = image.shape[0]
    image_y_size = image.shape[1]

    if filter.ndim > 2:
        raise DawisDimensionError('Not a 1D or 2D array -->', image.ndim, 2)

    if n_levels <= 0:
        raise DawisValueError('Number of wavelet planes cannot be negative -->', \
                                                n_levels, 'positive integer')

    #===========================================================================
    # Position array for filter
    if filter.ndim == 1:
        filter_ctr_index = filter.size // 2 + 1
        filter_half_size = filter.size // 2
        positive_positions = np.logspace( 0, filter_half_size - 1, \
                                            num = filter_half_size, base = 2)
        position_array = np.concatenate((- np.flip(positive_positions), np.zeros(1),
                                                            positive_positions))
    elif filter.ndim == 2:
        filter_ctr_index = (filter.shape[0] // 2 + 1, filter.shape[1] // 2 + 1 )
        filter_half_size = (filter.shape[0] // 2, filter.shape[1] // 2 )
        positive_positions_x = np.logspace( 0, filter_half_size[0] - 1, \
                                            num = filter_half_size[0], base = 2)
        positive_positions_y = np.logspace( 0, filter_half_size[1] - 1, \
                                                    num = filter_half_size[1], base = 2)
        position_array_x = np.concatenate((- np.flip(positive_positions_x), np.zeros(1),
                                                            positive_positions_x))
        position_array_y = np.concatenate((- np.flip(positive_positions_y), np.zeros(1),
                                                            positive_positions_y))
        position_array = np.array(np.meshgrid(position_array_x, position_array_y))

    #===========================================================================
    # Arrays
    coarse_array = np.zeros((image_x_size, image_y_size, n_levels + 1), dtype = np.float32)
    wavelet_array = np.zeros((image_x_size, image_y_size, n_levels ), dtype = np.float32)
    coarse_array[:,:,0] = image

    #===========================================================================
    # Convolution
    for level in range(0, n_levels):

        position_array_lvl = np.array(position_array) * np.power(2, level)
        position_array_lvl = position_array_lvl.astype(int)

        if filter.ndim == 2:
            with np.nditer(coarse_array[:,:,level + 1], flags = ['multi_index'], op_flags = ['readwrite']) as it:
                for coeff in it:
                    position_array_coeff = np.copy(position_array_lvl)
                    position_array_coeff[0] += it.multi_index[0]
                    position_array_coeff[1] += it.multi_index[1]
                    position_array_coeff = boundary_conditions(position_array_coeff, image.shape, conditions)

                    coeff += np.sum(coarse_array[position_array_coeff[0], position_array_coeff[1], level] * filter)

        elif filter.ndim == 1:

            temporary_copy = np.copy(coarse_array[:,:,level + 1])
            # Over X axis

            for x in range(0, image_x_size):

                position_array_coeff = np.copy(position_array_lvl)
                position_array_coeff += x

                position_array_coeff = boundary_conditions(position_array_coeff, image.shape[0], conditions)

                temporary_copy[x,:] = np.matmul(np.array([filter]), coarse_array[position_array_coeff, :, level])

            for y in range(0, image_y_size):

                position_array_coeff = np.copy(position_array_lvl)
                position_array_coeff += y
                position_array_coeff = boundary_conditions(position_array_coeff, image.shape[1], conditions)

                coarse_array[:,y,level + 1] = np.matmul(np.array([filter]), temporary_copy[ :, position_array_coeff].T)[0]

    for level in range(0, n_levels):
        wavelet_array[:,:,level] = coarse_array[:,:,level] - coarse_array[:,:,level+1]
    #wavelet_array[:,:,n_levels - 1] = np.copy(coarse_array[:,:,n_levels - 1])

    return coarse_array, wavelet_array

def ser_a_trous(C0, filter, scale):
    """
    The following is a serial implementation of the a trous algorithm. Accepts the following parameters:

    INPUTS:
    filter      (no default):   The filter-bank which is applied to the components of the transform.
    C0          (no default):   The current array on which filtering is to be performed.
    scale       (no default):   The scale for which the decomposition is being carried out.

    OUTPUTS:
    C1                          The result of applying the a trous algorithm to the input.
    
    Implementation borrowed from https://github.com/ratt-ru/PyMORESANE
    """
        
    tmp = filter[2]*C0

    tmp[(2**(scale+1)):,:] += filter[0]*C0[:-(2**(scale+1)),:]
    tmp[:(2**(scale+1)),:] += filter[0]*C0[(2**(scale+1))-1::-1,:]
    
    tmp[(2**scale):,:] += filter[1]*C0[:-(2**scale),:]
    tmp[:(2**scale),:] += filter[1]*C0[(2**scale)-1::-1,:]
        
    tmp[:-(2**scale),:] += filter[3]*C0[(2**scale):,:]
    tmp[-(2**scale):,:] += filter[3]*C0[:-(2**scale)-1:-1,:]
    
    tmp[:-(2**(scale+1)),:] += filter[4]*C0[(2**(scale+1)):,:]
    tmp[-(2**(scale+1)):,:] += filter[4]*C0[:-(2**(scale+1))-1:-1,:]
    
    C1 = filter[2]*tmp
    
    C1[:,(2**(scale+1)):] += filter[0]*tmp[:,:-(2**(scale+1))]
    C1[:,:(2**(scale+1))] += filter[0]*tmp[:,(2**(scale+1))-1::-1]
    
    C1[:,(2**scale):] += filter[1]*tmp[:,:-(2**scale)]
    C1[:,:(2**scale)] += filter[1]*tmp[:,(2**scale)-1::-1]
    
    C1[:,:-(2**scale)] += filter[3]*tmp[:,(2**scale):]
    C1[:,-(2**scale):] += filter[3]*tmp[:,:-(2**scale)-1:-1]
    
    C1[:,:-(2**(scale+1))] += filter[4]*tmp[:,(2**(scale+1)):]
    C1[:,-(2**(scale+1)):] += filter[4]*tmp[:,:-(2**(scale+1))-1:-1]

    return C1

def ser_iuwt_decomposition(
    in1: np.ndarray,
    scale_count: int,
    scale_adjust: int,
    store_smoothed: bool,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Decompose an image with the isotropic undecimated wavelet transform.

    The B3-spline à trous transform, on a single CPU core. This is the
    decomposition the pipeline runs; :func:`atrous` is the general-filter
    version and is slower.

    The wavelet planes are returned in the DAWIS axis order — the two image
    axes unchanged and the scale last — not the scale-first order the
    algorithm computes them in. The coarse plane is **not**: it comes back
    transposed relative to them. Nothing depends on this, because
    :func:`~dawis.wavelets.transforms.bspl_atrous` is the only caller that
    asks for it and it discards it, but a new caller must not assume the two
    outputs share an axis order.

    Parameters
    ----------
    in1 : numpy.ndarray
        The 2D image to decompose.
    scale_count : int
        Number of scales to smooth through. Scales are computed for
        ``range(scale_adjust, scale_count)``, so this is an exclusive upper
        bound, not the index of the coarsest scale.
    scale_adjust : int
        Number of leading scales to smooth through without storing. Their
        coefficients are discarded, but the smoothing still happens, because
        every scale depends on the one before it. The pipeline passes 0.
    store_smoothed : bool
        Whether to return the coarse plane alongside the wavelet planes.

    Returns
    -------
    detail_coeffs : numpy.ndarray
        Wavelet planes, of shape ``in1.shape + (scale_count - scale_adjust,)``.
    C0 : numpy.ndarray
        The coarse plane, of shape ``in1.shape[::-1]`` — transposed relative
        to both the input and the wavelet planes. Returned only when
        ``store_smoothed`` is True.

    Notes
    -----
    Implementation borrowed from https://github.com/ratt-ru/PyMORESANE.
    """

    wavelet_filter = (1./16)*np.array([1,4,6,4,1])      # Filter-bank for use in the a trous algorithm.

    # Initialises an empty array to store the coefficients.

    detail_coeffs = np.empty([scale_count-scale_adjust, in1.shape[0], in1.shape[1]])

    C0 = in1    # Sets the initial value to be the input array.

    # The following loop, which iterates up to scale_adjust, applies the a trous algorithm to the scales which are
    # considered insignificant. This is important as each set of wavelet coefficients depends on the last smoothed
    # version of the input.

    if scale_adjust>0:
        for i in range(0, scale_adjust):
            C0 = ser_a_trous(C0, wavelet_filter, i)

    # The meat of the algorithm - two sequential applications fo the a trous followed by determination and storing of
    # the detail coefficients. C0 is reassigned the value of C on each loop - C0 is always the smoothest version of the
    # input image.

    for i in range(scale_adjust,scale_count):
        C = ser_a_trous(C0, wavelet_filter, i)                                  # Approximation coefficients.
        C1 = ser_a_trous(C, wavelet_filter, i)                                  # Approximation coefficients.
        detail_coeffs[i-scale_adjust,:,:] = C0 - C1                             # Detail coefficients.
        C0 = C

    # DAWIS comp
    detail_coeffs = np.swapaxes(detail_coeffs, 0, 2)
    detail_coeffs = np.swapaxes(detail_coeffs, 0, 1)
    C0 = np.swapaxes(C0, 0, 1)
    
    if store_smoothed:
        return detail_coeffs, C0
    else:
        return detail_coeffs
