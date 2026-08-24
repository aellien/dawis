#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This file is part of DAWIS (Detection Algorithm for Intracluster light Studies).
# Author: Amaël Ellien
# Last modification: 20/05/2021
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODULES
import numpy as np
from dawis.exceptions import *
import pdb
#from numba import jit, njit
import logging

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def boundary_conditions(position_array, sizes, conditions = 'prolongation' ):
    '''
    Border conditions, doc to do
    '''

    if np.ndim(position_array) not in [1, 3]:
        raise DawisDimensionError('Wrong dimensions for array -->', np.ndim(position_array), [1, 3])

    if np.ndim(sizes) not in [0, 1]:
        raise DawisDimensionError('Wrong dimensions for array -->', np.ndim(sizes), [0, 1])

    if type(conditions) != str:
        raise DawisWrongType('Wrong type -->', type(conditions), 'str')

    allowed_conditions = ['mirror', 'prolongation', 'loop' ]

    if conditions not in allowed_conditions:
        raise DawisValueError('Not an allowed condition -->', condition, allowed_conditions)

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

def atrous(image, n_levels, filter, conditions = 'loop', verbose = False):
    '''
    A trous algorithm, doc to do
    '''

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

def ser_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed):
    """
    This function calls the a trous algorithm code to decompose the input into its wavelet coefficients. This is
    the isotropic undecimated wavelet transform implemented for a single CPU core.

    INPUTS:
    in1                 (no default):   Array on which the decomposition is to be performed.
    scale_count         (no default):   Maximum scale to be considered.
    scale_adjust        (default=0):    Adjustment to scale value if first scales are of no interest.
    store_smoothed      (default=False):Boolean specifier for whether the smoothed image is stored or not.

    OUTPUTS:
    detail_coeffs                       Array containing the detail coefficients.
    C0                  (optional):     Array containing the smoothest version of the input.
    
    Implementation borrowed from https://github.com/ratt-ru/PyMORESANE
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
