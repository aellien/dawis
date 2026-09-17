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


def dilated_offsets(
    filter: np.ndarray,
    level: int,
) -> np.ndarray:
    """Where the à trous filter samples from, at one level.

    Transcribed from :func:`atrous`, which builds its offsets by ``logspace``
    rather than by ``arange`` — so a 5-tap filter reaches ``(-2, -1, 0, 1,
    2)`` and a 3-tap one ``(-1, 0, 1)``, both then dilated by ``2 **
    level``.

    Parameters
    ----------
    filter : numpy.ndarray
        The 1D scaling filter.
    level : int
        The à trous level, counting from zero.

    Returns
    -------
    numpy.ndarray
        Integer offsets, one per filter tap.

    Raises
    ------
    DawisDimensionError
        If the filter is not 1D. :func:`atrous` also takes a 2D separable
        filter and meshes its offsets in both axes; the transpose is written
        for the 1D path the published solve runs, and an unguarded 2D filter
        would either broadcast its rows against the image and return a wrong
        array, or fail far from here.
    DawisValueError
        If the filter has an even number of taps. The construction produces
        ``2 * (size // 2) + 1`` offsets, so an even filter would come back
        with one more offset than it has taps, and the ``zip`` in
        :func:`smooth_adjoint` would silently drop the last one and transpose
        a different operator. :func:`atrous` fails on the same filter,
        loudly, at its ``matmul``; this fails at the same place for the same
        reason.

    See Also
    --------
    smooth_adjoint : The transposed smoothing these offsets scatter into.
    """

    if filter.ndim != 1:
        raise DawisDimensionError('Not a 1D filter -->', filter.ndim, 1)

    if filter.size % 2 == 0:
        raise DawisValueError('The a trous filter has to have a centre -->',
                              filter.size, 'odd number of taps')

    filter_half_size = filter.size // 2
    positive_positions = np.logspace(0, filter_half_size - 1,
                                     num = filter_half_size, base = 2)
    position_array = np.concatenate((- np.flip(positive_positions),
                                     np.zeros(1), positive_positions))

    return (position_array * 2 ** level).astype(int)


def _clamped(offset: int, size: int) -> np.ndarray:
    """The indices a gather at one offset reads, folded back inside.

    :func:`boundary_conditions` with ``'prolongation'`` clamps rather than
    mirrors or wraps, so a sample off the low edge reads index 0 and one off
    the high edge reads ``size - 1``.

    Parameters
    ----------
    offset : int
        The dilated filter offset, from :func:`dilated_offsets`.
    size : int
        The image extent along the axis being folded.

    Returns
    -------
    numpy.ndarray
        Integer indices, of length ``size``.
    """

    return np.clip(np.arange(size) + offset, 0, size - 1)


def smooth_adjoint(
    image: np.ndarray,
    filter: np.ndarray,
    level: int,
) -> np.ndarray:
    """Apply the transpose of one à trous smoothing.

    The forward smoothing is a gather — ``out[x] = sum_k f[k] * in[clamp(x +
    d_k)]`` along each axis in turn — so its transpose is the scatter that
    accumulates ``f[k] * out[x]`` back into ``clamp(x + d_k)``. The two axis
    passes compose in the opposite order to the forward, which applies rows
    then columns.

    Clamping is what makes the smoothing operator **not** symmetric: it
    duplicates the edge samples, so re-running the same smoothing is not the
    transpose of it.

    Parameters
    ----------
    image : numpy.ndarray
        The 2D array to transpose-smooth.
    filter : numpy.ndarray
        The 1D scaling filter.
    level : int
        The à trous level, counting from zero.

    Returns
    -------
    numpy.ndarray
        2D, the shape of ``image``.

    See Also
    --------
    atrous_adjoint : The recursion built from these.
    """

    offsets = dilated_offsets(filter, level)
    rows, columns = image.shape

    along_columns = np.zeros_like(image, dtype = float)
    for tap, offset in zip(filter, offsets):
        np.add.at(along_columns, (slice(None), _clamped(offset, columns)),
                  tap * image)

    along_rows = np.zeros_like(along_columns)
    for tap, offset in zip(filter, offsets):
        np.add.at(along_rows, _clamped(offset, rows), tap * along_columns)

    return along_rows


def atrous_adjoint(
    wavelet_tube: np.ndarray,
    filter: np.ndarray,
    coarse: bool = False,
) -> np.ndarray:
    r"""Apply the transpose of the à trous transform to a tube.

    The adjoint of ``atrous(x, n_levels, filter, 'prolongation')[1]``, the
    wavelet planes alone — unless ``coarse`` says the tube carries the coarse
    term as well, in which case it is the adjoint of the transform that
    returns those planes *and* :math:`c_{L+1}`.

    With :math:`H_{j}` the smoothing at level :math:`j`, the transform a tube
    holds is

    .. math::

        w_{j} = (I - H_{j}) H_{j-1} \cdots H_{0} x .

    Transposing term by term and factoring gives the backwards recursion run
    here, which costs :math:`2n` smoothing transposes for :math:`n` levels
    rather than the :math:`n(n+1)/2` the sum-of-products form would.

    This is a third operator, not a renaming of either published one.
    :meth:`~dawis.trees.InterscaleTree.synthesis_sum` adds the planes, which
    is the starlet's *reconstruction* dual;
    :meth:`~dawis.trees.InterscaleTree.synthesis_adjoint` cascades each plane
    through the low-pass filters, which is the operator the published lineage
    adopted *after* the true adjoint produced ring artefacts ([LU3CW9P8]
    p.272, #76) — a rejection #85 measured on this instrument and did not
    reproduce.

    Parameters
    ----------
    wavelet_tube : numpy.ndarray
        3D, of shape ``(rows, columns, n_levels)``, or of shape
        ``(rows, columns, n_levels + 1)`` when ``coarse`` is True — the
        wavelet planes, then the coarse channel.
    filter : numpy.ndarray
        The 1D scaling filter, normally the B3-spline kernel.
    coarse : bool, optional
        Whether the last channel of the tube is the coarse term (#89).
        Default False, which is the transform every result in #74's Ledger
        was taken on.

    Returns
    -------
    numpy.ndarray
        2D, of shape ``(rows, columns)``.

    Raises
    ------
    DawisValueError
        If the tube carries no wavelet plane at all. With ``coarse`` True a
        one-channel tube leaves no plane behind the coarse channel, and the
        recursion would read that channel twice — once as the coarse term and
        once as a detail plane at level ``-1``, where the dilation ``2 ** -1``
        collapses the stencil onto duplicated taps. It returns an array rather
        than failing, so it is guarded here.

    See Also
    --------
    atrous : The transform this is the transpose of.
    smooth_adjoint : The per-level transpose this recursion is built from.

    Notes
    -----
    The coarse term of the transform is :math:`c_{L+1} = H_{L} \cdots H_{0}
    x`, so its column of :math:`W^{\mathsf{T}}` is :math:`H_{0}^{\mathsf{T}}
    \cdots H_{L}^{\mathsf{T}}`. Applied right to left that is
    :math:`H_{L}^{\mathsf{T}}` first and the rest after, which is exactly
    what the backwards recursion already does to whatever it starts from. So
    the channel costs one extra smoothing transpose rather than :math:`L + 1`
    of them: it is seeded into the accumulator at the top level and rides the
    recursion down.

    The identity that pins this operator is :math:`\langle W x, y \rangle =
    \langle x, W^{\mathsf{T}} y \rangle`, which holds for the adjoint and for
    nothing else. It is asserted on both filters, several depths and
    non-square boxes in ``tests/unit/test_synthesis_transpose.py``.
    """

    n_levels = wavelet_tube.shape[2] - int(coarse)

    if n_levels < 1:
        raise DawisValueError('Tube carries no wavelet plane -->',
                              wavelet_tube.shape[2],
                              'at least one plane, plus one channel when '
                              'coarse is True')

    def detail_adjoint(level):
        plane = wavelet_tube[:, :, level].astype(float)
        return plane - smooth_adjoint(plane, filter, level)

    accumulated = detail_adjoint(n_levels - 1)
    if coarse:
        accumulated = accumulated + smooth_adjoint(
            wavelet_tube[:, :, n_levels].astype(float), filter, n_levels - 1)

    for level in range(n_levels - 2, -1, -1):
        accumulated = (detail_adjoint(level)
                       + smooth_adjoint(accumulated, filter, level))

    return accumulated

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


def ser_iuwt_cascade(
    in1: np.ndarray,
    scale_count: int,
    scale_adjust: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Decompose an image and keep every smooth plane, not only the last.

    :func:`ser_iuwt_decomposition` walks the chain
    :math:`c_0, c_1, \dots, c_{d}` and returns :math:`c_{d}` alone. A tree
    of interscale-maximum level :math:`L` needs :math:`c_{L+1}` as its coarse
    term (#88, #89), and the trees of one iteration sit at many levels, so
    the whole chain is needed at once.

    The planes cost **no extra smoothing**. They are the loop's own running
    value, kept instead of overwritten; the price is one array the size of
    the wavelet planes. The alternative — one decomposition per depth, which
    is what ``restoration_tests/fixtures/mock.py`` pays at dump time — costs
    the smoothing chain :math:`d` times over, and :func:`atrous` is 87% of a
    run's wall time (#5).

    The loop below is written out rather than shared with
    :func:`ser_iuwt_decomposition`, so that the two are independent
    implementations and the test that asserts them equal at every depth can
    disagree with one of them.

    Parameters
    ----------
    in1 : numpy.ndarray
        The 2D image to decompose.
    scale_count : int
        Number of scales to smooth through, an exclusive upper bound as in
        :func:`ser_iuwt_decomposition`.
    scale_adjust : int, optional
        Number of leading scales whose *wavelet* coefficients are discarded.
        The smoothing still happens and their smooth planes are still kept,
        because the cascade is indexed by level and a caller asking for
        :math:`c_{L+1}` does not know which scales the caller of the wavelet
        planes threw away. Default 0, which is what the pipeline passes.

    Returns
    -------
    detail_coeffs : numpy.ndarray
        Wavelet planes, of shape ``in1.shape + (scale_count - scale_adjust,)``
        — bit for bit what :func:`ser_iuwt_decomposition` returns.
    coarse_cascade : numpy.ndarray
        Smooth planes, of shape ``in1.shape + (scale_count + 1,)``. Plane
        ``j`` is the image smoothed ``j`` times, so plane 0 is the image and
        plane ``scale_count`` is the coarse plane
        :func:`ser_iuwt_decomposition` returns.

    Notes
    -----
    Unlike :func:`ser_iuwt_decomposition`'s coarse plane, which comes back
    transposed relative to its own wavelet planes, the cascade is returned in
    the DAWIS axis order — the two image axes unchanged and the scale last,
    the same order as the wavelet planes beside it.

    The chain does **not** telescope onto the wavelet planes: the loop takes
    the detail as :math:`w_j = c_j - H_j^2 c_j` while the chain advances as
    :math:`c_{j+1} = H_j c_j`, one smoothing against two, so
    :math:`c_{L+1} \neq c_{d} + \sum_{j > L} w_j` (#89, measured at 15.8-21.0%
    relative).

    See Also
    --------
    ser_iuwt_decomposition : The pipeline's decomposition, coarse plane
        dropped.
    """

    wavelet_filter = (1. / 16) * np.array([1, 4, 6, 4, 1])

    detail_coeffs = np.empty([scale_count - scale_adjust,
                              in1.shape[0], in1.shape[1]])
    coarse_cascade = np.empty([in1.shape[0], in1.shape[1], scale_count + 1])

    C0 = in1
    coarse_cascade[:, :, 0] = C0

    for i in range(0, scale_adjust):
        C0 = ser_a_trous(C0, wavelet_filter, i)
        coarse_cascade[:, :, i + 1] = C0

    for i in range(scale_adjust, scale_count):
        C = ser_a_trous(C0, wavelet_filter, i)
        C1 = ser_a_trous(C, wavelet_filter, i)
        detail_coeffs[i - scale_adjust, :, :] = C0 - C1
        C0 = C
        coarse_cascade[:, :, i + 1] = C0

    detail_coeffs = np.swapaxes(detail_coeffs, 0, 2)
    detail_coeffs = np.swapaxes(detail_coeffs, 0, 1)

    return detail_coeffs, coarse_cascade
