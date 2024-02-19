#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This file is part of DAWIS (Detection Algorithm for Intracluster light Studies).
# Author: Amaël Ellien
# Last modification: 20/05/2021
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODULES
import numpy as np
from dawis.datacube import datacube
from dawis.exceptions import *
import pdb
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.signal import bspline
import matplotlib.pyplot as plt
from numba import jit, njit
from dawis.congrid import congrid
import logging
import multiprocessing as mp
import ctypes

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

def interlaced_atrous_congrid(image, n_levels, n_voices, filter, verbose = False):

    '''
    Interlaced à trous algorithm, doc to do
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
    final_coarse_array = np.zeros((image_x_size, image_y_size, 2 * n_levels  ), dtype = np.float32)
    final_wavelet_array = np.zeros((image_x_size, image_y_size, 2 * n_levels ), dtype = np.float32)

    #===========================================================================
    # Convolution 0

    anx, any = int(image_x_size / ( np.sqrt(2) / 2 )), \
               int(image_y_size / ( np.sqrt(2) / 2 ))
    print(anx, any)
    coarse_array_1 = np.zeros((image_x_size, image_y_size, n_levels + 1 ), dtype = np.float32)
    coarse_array_2 = np.zeros((image_x_size, image_y_size, n_levels ), dtype = np.float32)

    congrid_coarse_array = np.zeros((anx, \
                                     any, \
                                     n_levels + 1 ), dtype = np.float32)

    coarse_array_1[:,:,0] = np.copy(image)
    congrid_coarse_array[:,:,0] = congrid(image, (anx, any), method = 'spline')

    #===========================================================================
    # Convolution 1
    for level in range(0, n_levels ):

        position_array_lvl = np.array(position_array) * np.power(2, level)
        position_array_lvl = position_array_lvl.astype(int)

        if filter.ndim == 2:
            with np.nditer(coarse_array_1[:,:,level + 1], flags = ['multi_index'], op_flags = ['readwrite']) as it:
                for coeff in it:
                    position_array_coeff = np.copy(position_array_lvl)
                    position_array_coeff[0] += it.multi_index[0]
                    position_array_coeff[1] += it.multi_index[1]
                    position_array_coeff = boundary_conditions(position_array_coeff, interp_im.shape, 'loop')

                    coeff += np.sum(coarse_array_1[position_array_coeff[0], position_array_coeff[1], level] * filter)

        elif filter.ndim == 1:

            temporary_copy = np.zeros((image_x_size, image_y_size))
            # Over X axis

            for x in range(0, image_x_size ):

                position_array_coeff = np.copy(position_array_lvl)
                position_array_coeff += x
                position_array_coeff = boundary_conditions(position_array_coeff, image.shape[0], 'loop')

                temporary_copy[x,:] = np.matmul(np.array([filter]), coarse_array_1[position_array_coeff, :, level])

            for y in range(0, image_y_size ):

                position_array_coeff = np.copy(position_array_lvl)
                position_array_coeff += y
                position_array_coeff = boundary_conditions(position_array_coeff, image.shape[1] , 'loop')

                coarse_array_1[:,y,level + 1] = np.matmul(np.array([filter]), temporary_copy[ :, position_array_coeff].T)[0]

    # Convolution 2
    for level in range(0, n_levels ):

        position_array_lvl = np.array(position_array) * np.power(2, level)
        position_array_lvl = position_array_lvl.astype(int)

        if filter.ndim == 2:
            with np.nditer(coarse_array_2[:,:,level + 1], flags = ['multi_index'], op_flags = ['readwrite']) as it:
                for coeff in it:
                    position_array_coeff = np.copy(position_array_lvl)
                    position_array_coeff[0] += it.multi_index[0]
                    position_array_coeff[1] += it.multi_index[1]
                    position_array_coeff = boundary_conditions(position_array_coeff, interp_im.shape, 'loop')

                    coeff += np.sum(coarse_array_2[position_array_coeff[0], position_array_coeff[1], level] * filter)

        elif filter.ndim == 1:

            temporary_copy = np.copy(congrid_coarse_array[:,:,level + 1])

            for x in range(0, anx):

                position_array_coeff = np.copy(position_array_lvl)
                position_array_coeff += x
                position_array_coeff = boundary_conditions(position_array_coeff, anx, 'loop')

                temporary_copy[x,:] = np.matmul(np.array([filter]), congrid_coarse_array[position_array_coeff, :, level])

            for y in range(0, any):

                position_array_coeff = np.copy(position_array_lvl)
                position_array_coeff += y
                position_array_coeff = boundary_conditions(position_array_coeff, any, 'loop')

                congrid_coarse_array[:,y,level + 1] = np.matmul(np.array([filter]), temporary_copy[ :, position_array_coeff].T)[0]

            coarse_array_2[:,:,level] = congrid(congrid_coarse_array[:,:,level + 1], (image_x_size, image_y_size), method = 'spline')

    # Correction for mean bias arror given by interpolation
    for level in range(0, n_levels):

        interpmean = ( np.mean( coarse_array_1[:,:,level + 1] ) + np.mean( coarse_array_1[:,:,level] ) ) / 2.
        norm = interpmean - np.mean( coarse_array_2[:,:,level] )
        coarse_array_2[:,:,level] += norm

    voice = 0
    for level in range(0, n_levels):

        final_wavelet_array[:,:,voice] = coarse_array_1[:,:,level] - coarse_array_2[:,:,level]
        final_coarse_array[:,:,voice] = coarse_array_1[:,:,level]
        #print(level, voice, np.mean(final_wavelet_array[:,:,voice]), np.std(final_wavelet_array[:,:,voice]))
        voice += 1

        final_wavelet_array[:,:,voice] = coarse_array_2[:,:,level] - coarse_array_1[:,:,level+1]
        final_coarse_array[:,:,voice] = coarse_array_2[:,:,level]
        #print(level, voice, np.mean(final_wavelet_array[:,:,voice]), np.std(final_wavelet_array[:,:,voice]))
        voice += 1

    return final_coarse_array, final_wavelet_array

#@jit(nopython=True, cache = True)
def DL_atrou(IMAGE,N_LEVELS=11,WAV_TYPE='BSPL',WAV_COEFF=True,VERBOSE=False, DISPLAY=False):
    '''
    Wavelet convolution of a 2D image. Here we use an interpolation method for B-spline wavelets, in order to gain calculus time. The output data are a 3D table, containing the wavelets coefficients for each wavelet plane.
    '''
    
    IMAGE_SHAPE=np.shape(IMAGE) # Test if data are a 2D image.
            
    X_SIZE=IMAGE_SHAPE[0]
    Y_SIZE=IMAGE_SHAPE[1]
    WAV_PLANES=np.zeros((X_SIZE,Y_SIZE,N_LEVELS),dtype=np.float64)
    WAV_PLANES[:,:,0]=IMAGE # Initialisation of the wavelet planes.
    TEMP_TAB=np.zeros((X_SIZE,Y_SIZE),dtype=np.float64) # Needed to compute
                                                            # the 2D convolution.
                                                            
    for LEVEL in range(0,N_LEVELS-1):
                
        if WAV_TYPE=='BSPL':
            
            # Position vector for B-spline wavelets.                
            POS_VECT=np.array([-2*(2**LEVEL),-2**LEVEL,0,2**LEVEL,2*(2**LEVEL)])
            
            # The actual convolution for the X-axis for B-spline wavelets.
            for X in range(0,X_SIZE):
                X_VECT=POS_VECT+X
                X_VECT[np.where(X_VECT<0)]=0
                X_VECT[np.where(X_VECT>(X_SIZE-1))]=X_SIZE-1
                TEMP_TAB[X,:]=(WAV_PLANES[X_VECT[0],:,LEVEL]\
                                  +WAV_PLANES[X_VECT[4],:,LEVEL])*0.0625\
                                  +(WAV_PLANES[X_VECT[1],:,LEVEL]\
                                  +WAV_PLANES[X_VECT[3],:,LEVEL])*0.2500\
                                  +WAV_PLANES[X_VECT[2],:,LEVEL]*0.3750
                  
            # The actual convolution for the Y-axis for B-spline wavelets. 
            for Y in range(0,Y_SIZE):
                Y_VECT=POS_VECT+Y
                Y_VECT[np.where(Y_VECT<0)]=0
                Y_VECT[np.where(Y_VECT>(Y_SIZE-1))]=Y_SIZE-1
                WAV_PLANES[:,Y,LEVEL+1]=(TEMP_TAB[:,Y_VECT[0]]\
                                            +TEMP_TAB[:,Y_VECT[4]])*0.0625\
                                            +(TEMP_TAB[:,Y_VECT[1]]\
                                            +TEMP_TAB[:,Y_VECT[3]])*0.2500\
                                            +TEMP_TAB[:,Y_VECT[2]]*0.3750

    if WAV_COEFF==True:
        
        for LEVEL in range(0,N_LEVELS-1): # Create wavelet coefficients.
            
            WAV_PLANES[:,:,LEVEL]=WAV_PLANES[:,:,LEVEL]\
                                        -WAV_PLANES[:,:,LEVEL+1]
            

    return WAV_PLANES

#@jit(cache = True)
def ser_a_trous(C0, filter, scale):
    """
    The following is a serial implementation of the a trous algorithm. Accepts the following parameters:

    INPUTS:
    filter      (no default):   The filter-bank which is applied to the components of the transform.
    C0          (no default):   The current array on which filtering is to be performed.
    scale       (no default):   The scale for which the decomposition is being carried out.

    OUTPUTS:
    C1                          The result of applying the a trous algorithm to the input.
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

    if store_smoothed:
        return detail_coeffs, C0
    else:
        return detail_coeffs

def mp_a_trous(C0, wavelet_filter, scale, core_count):
    """
    This is a reimplementation of the a trous filter which makes use of multiprocessing. In particular,
    it divides the input array of dimensions NxN into M smaller arrays of dimensions (N/M)xN, where M is the
    number of cores which are to be used.

    INPUTS:
    C0              (no default):   The current array which is to be decomposed.
    wavelet_filter  (no default):   The filter-bank which is applied to the components of the transform.
    scale           (no default):   The scale at which decomposition is to be carried out.
    core_count      (no default):   The number of CPU cores over which the task should be divided.

    OUTPUTS:
    shared_array                    The decomposed array.
    """

    # Creates an array which may be accessed by multiple processes.

    shared_array_base = mp.Array(ctypes.c_float, C0.shape[0]**2, lock=False)
    shared_array = np.frombuffer(shared_array_base, dtype=ctypes.c_float)
    shared_array = shared_array.reshape(C0.shape)
    shared_array[:,:] = C0

    # Division of the problem and allocation of processes to cores.

    processes = []

    for i in range(core_count):
        process = mp.Process(target = mp_a_trous_kernel, args = (shared_array, wavelet_filter, scale, i,
                                                     C0.shape[0]//core_count, 'row',))
        process.start()
        processes.append(process)

    for i in processes:
        i.join()

    processes = []

    for i in range(core_count):
        process = mp.Process(target = mp_a_trous_kernel, args = (shared_array, wavelet_filter, scale, i,
                                                     C0.shape[1]//core_count, 'col',))
        process.start()
        processes.append(process)

    for i in processes:
        i.join()

    return shared_array

def mp_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed, core_count):
    """
    This function calls the a trous algorithm code to decompose the input into its wavelet coefficients. This is
    the isotropic undecimated wavelet transform implemented for multiple CPU cores. NOTE: Python is not well suited
    to multiprocessing - this may not improve execution speed.

    INPUTS:
    in1                 (no default):   Array on which the decomposition is to be performed.
    scale_count         (no default):   Maximum scale to be considered.
    scale_adjust        (default=0):    Adjustment to scale value if first scales are of no interest.
    store_smoothed      (default=False):Boolean specifier for whether the smoothed image is stored or not.
    core_count          (no default):   Indicates the number of cores to be used.

    OUTPUTS:
    detail_coeffs                       Array containing the detail coefficients.
    C0                  (optional):     Array containing the smoothest version of the input.
    """

    wavelet_filter = (1./16)*np.array([1,4,6,4,1])      # Filter-bank for use in the a trous algorithm.

    C0 = in1                                            # Sets the initial value to be the input array.

    # Initialises a zero array to store the coefficients.

    detail_coeffs = np.empty([scale_count-scale_adjust, in1.shape[0], in1.shape[1]])

    # The following loop, which iterates up to scale_adjust, applies the a trous algorithm to the scales which are
    # considered insignificant. This is important as each set of wavelet coefficients depends on the last smoothed
    # version of the input.

    if scale_adjust>0:
        for i in range(0, scale_adjust):
            C0 = mp_a_trous(C0, wavelet_filter, i, core_count)

    # The meat of the algorithm - two sequential applications fo the a trous followed by determination and storing of
    # the detail coefficients. C0 is reassigned the value of C on each loop - C0 is always the smoothest version of the
    # input image.

    for i in range(scale_adjust,scale_count):
        C = mp_a_trous(C0, wavelet_filter, i, core_count)                   # Approximation coefficients.
        C1 = mp_a_trous(C, wavelet_filter, i, core_count)                   # Approximation coefficients.
        detail_coeffs[i-scale_adjust,:,:] = C0 - C1                         # Detail coefficients.
        C0 = C

    if store_smoothed:
        return detail_coeffs, C0
    else:
        return detail_coeffs


def mp_a_trous_kernel(C0, wavelet_filter, scale, slice_ind, slice_width, r_or_c="row"):
    """
    This is the convolution step of the a trous algorithm.

    INPUTS:
    C0              (no default):       The current array which is to be decomposed.
    wavelet_filter  (no default):       The filter-bank which is applied to the elements of the transform.
    scale           (no default):       The scale at which decomposition is to be carried out.
    slice_ind       (no default):       The index of the particular slice in which the decomposition is being performed.
    slice_width     (no default):       The number of elements of the shorter dimension of the array for decomposition.
    r_or_c          (default = "row"):  Indicates whether strips are rows or columns.

    OUTPUTS:
    NONE - C0 is a mutable array which this function alters. No value is returned.

    """

    lower_bound = slice_ind*slice_width
    upper_bound = (slice_ind+1)*slice_width

    if r_or_c == "row":
        row_conv = wavelet_filter[2]*C0[:,lower_bound:upper_bound]

        row_conv[(2**(scale+1)):,:] += wavelet_filter[0]*C0[:-(2**(scale+1)),lower_bound:upper_bound]
        row_conv[:(2**(scale+1)),:] += wavelet_filter[0]*C0[(2**(scale+1))-1::-1,lower_bound:upper_bound]

        row_conv[(2**scale):,:] += wavelet_filter[1]*C0[:-(2**scale),lower_bound:upper_bound]
        row_conv[:(2**scale),:] += wavelet_filter[1]*C0[(2**scale)-1::-1,lower_bound:upper_bound]

        row_conv[:-(2**scale),:] += wavelet_filter[3]*C0[(2**scale):,lower_bound:upper_bound]
        row_conv[-(2**scale):,:] += wavelet_filter[3]*C0[:-(2**scale)-1:-1,lower_bound:upper_bound]

        row_conv[:-(2**(scale+1)),:] += wavelet_filter[4]*C0[(2**(scale+1)):,lower_bound:upper_bound]
        row_conv[-(2**(scale+1)):,:] += wavelet_filter[4]*C0[:-(2**(scale+1))-1:-1,lower_bound:upper_bound]

        C0[:,lower_bound:upper_bound] = row_conv

    elif r_or_c == "col":
        col_conv = wavelet_filter[2]*C0[lower_bound:upper_bound,:]

        col_conv[:,(2**(scale+1)):] += wavelet_filter[0]*C0[lower_bound:upper_bound,:-(2**(scale+1))]
        col_conv[:,:(2**(scale+1))] += wavelet_filter[0]*C0[lower_bound:upper_bound,(2**(scale+1))-1::-1]

        col_conv[:,(2**scale):] += wavelet_filter[1]*C0[lower_bound:upper_bound,:-(2**scale)]
        col_conv[:,:(2**scale)] += wavelet_filter[1]*C0[lower_bound:upper_bound,(2**scale)-1::-1]

        col_conv[:,:-(2**scale)] += wavelet_filter[3]*C0[lower_bound:upper_bound,(2**scale):]
        col_conv[:,-(2**scale):] += wavelet_filter[3]*C0[lower_bound:upper_bound,:-(2**scale)-1:-1]

        col_conv[:,:-(2**(scale+1))] += wavelet_filter[4]*C0[lower_bound:upper_bound,(2**(scale+1)):]
        col_conv[:,-(2**(scale+1)):] += wavelet_filter[4]*C0[lower_bound:upper_bound,:-(2**(scale+1))-1:-1]

        C0[lower_bound:upper_bound,:] = col_conv

def interlaced_atrous_zeros(image, n_levels, n_voices, filter, verbose = False):

    '''
    Interlaced à trous algorithm, doc to do
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
    final_coarse_array = np.zeros((image_x_size, image_y_size, 2 * n_levels), dtype = np.float32)
    final_wavelet_array = np.zeros((image_x_size, image_y_size, 2 * n_levels), dtype = np.float32)

    coarse_array_1 = np.zeros((image_x_size, image_y_size, n_levels), dtype = np.float32)
    coarse_array_2 = np.zeros((image_x_size, image_y_size, n_levels), dtype = np.float32)

    coarse_array_1[:,:,0] = np.copy(image)
    coarse_array_2[:,:,0] = np.copy(image)

    #===========================================================================
    # Convolution 1
    for level in range(0, n_levels - 1):

        position_array_lvl = np.array(position_array) * np.power(2, level)
        position_array_lvl = position_array_lvl.astype(int)

        if filter.ndim == 2:
            with np.nditer(coarse_array_1[:,:,level + 1], flags = ['multi_index'], op_flags = ['readwrite']) as it:
                for coeff in it:
                    position_array_coeff = np.copy(position_array_lvl)
                    position_array_coeff[0] += it.multi_index[0]
                    position_array_coeff[1] += it.multi_index[1]
                    position_array_coeff = boundary_conditions(position_array_coeff, interp_im.shape, 'loop')

                    coeff += np.sum(coarse_array_1[position_array_coeff[0], position_array_coeff[1], level] * filter)

        elif filter.ndim == 1:

            temporary_copy = np.copy(coarse_array_1[:,:,level + 1])
            # Over X axis

            for x in range(0, image_x_size ):

                position_array_coeff = np.copy(position_array_lvl)
                position_array_coeff += x
                position_array_coeff = boundary_conditions(position_array_coeff, image.shape[0], 'loop')

                temporary_copy[x,:] = np.matmul(np.array([filter]), coarse_array_1[position_array_coeff, :, level])

            for y in range(0, image_y_size ):

                position_array_coeff = np.copy(position_array_lvl)
                position_array_coeff += y
                position_array_coeff = boundary_conditions(position_array_coeff, image.shape[1] , 'loop')

                coarse_array_1[:,y,level + 1] = np.matmul(np.array([filter]), temporary_copy[ :, position_array_coeff].T)[0]

    # Convolution 2
    for level in range(0, n_levels - 1):

        position_array_lvl = np.array(position_array) * np.power(2, level + 0.5)
        position_array_lvl_ceil = np.ceil(position_array_lvl).astype(int)
        position_array_lvl_floor = np.floor(position_array_lvl).astype(int)

        if filter.ndim == 2:
            with np.nditer(coarse_array_2[:,:,level + 1], flags = ['multi_index'], op_flags = ['readwrite']) as it:
                for coeff in it:
                    position_array_coeff = np.copy(position_array_lvl)
                    position_array_coeff[0] += it.multi_index[0]
                    position_array_coeff[1] += it.multi_index[1]
                    position_array_coeff = boundary_conditions(position_array_coeff, interp_im.shape, 'loop')

                    coeff += np.sum(coarse_array_2[position_array_coeff[0], position_array_coeff[1], level] * filter)

        elif filter.ndim == 1:

            temporary_copy = np.copy(coarse_array_2[:,:,level])

            for x in range(0, image_x_size):

                position_array_ceil_coeff = np.copy(position_array_lvl_ceil)
                position_array_ceil_coeff += x
                position_array_ceil_coeff = boundary_conditions(position_array_ceil_coeff, image_x_size, 'loop')

                position_array_floor_coeff = np.copy(position_array_lvl_floor)
                position_array_floor_coeff += x
                position_array_floor_coeff = boundary_conditions(position_array_floor_coeff, image_x_size, 'loop')

                temporary_copy[x,:] = np.matmul(np.array([filter]),\
                            0.5 * (coarse_array_2[position_array_ceil_coeff, :, level]\
                            + coarse_array_2[position_array_floor_coeff, :, level]))

            for y in range(0, image_y_size):

                position_array_ceil_coeff = np.copy(position_array_lvl_ceil)
                position_array_ceil_coeff += y
                position_array_ceil_coeff = boundary_conditions(position_array_ceil_coeff, image_y_size, 'loop')

                position_array_floor_coeff = np.copy(position_array_lvl_floor)
                position_array_floor_coeff += y
                position_array_floor_coeff = boundary_conditions(position_array_floor_coeff, image_y_size, 'loop')

                #pdb.set_trace()

                coarse_array_2[:,y,level + 1] = np.matmul(np.array([filter]), \
                            0.5 * (temporary_copy[ :, position_array_ceil_coeff] \
                            + temporary_copy[ :, position_array_floor_coeff]).T)[0]

    voice = 0
    stds = []
    for level in range(0, n_levels - 1):

        final_wavelet_array[:,:,voice] = coarse_array_2[:,:,level] - coarse_array_1[:,:,level+1]
        final_coarse_array[:,:,voice] = coarse_array_2[:,:,level]
        print(level, voice, np.mean(final_wavelet_array[:,:,voice]), np.std(final_wavelet_array[:,:,voice]))
        stds.append(np.std(final_wavelet_array[:,:,voice]))
        voice += 1

        final_wavelet_array[:,:,voice] = coarse_array_1[:,:,level+1] - coarse_array_2[:,:,level+1]
        final_coarse_array[:,:,voice] = coarse_array_1[:,:,level+1]
        print(level, voice, np.mean(final_wavelet_array[:,:,voice]), np.std(final_wavelet_array[:,:,voice]))
        stds.append(np.std(final_wavelet_array[:,:,voice]))
        voice += 1

    final_wavelet_array[:,:,-1] = np.copy(coarse_array_2[:,:,-1])

    return final_coarse_array, final_wavelet_array, stds

def interlaced_atrous_filter(image, n_levels, n_voices, filter, verbose = False):

    '''
    Interlaced à trous algorithm, doc to do
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
    final_coarse_array = np.zeros((image_x_size, image_y_size, 2 * n_levels), dtype = np.float32)
    final_wavelet_array = np.zeros((image_x_size, image_y_size, 2 * n_levels), dtype = np.float32)

    coarse_array_1 = np.zeros((image_x_size, image_y_size, n_levels), dtype = np.float32)
    coarse_array_2 = np.zeros((image_x_size, image_y_size, n_levels), dtype = np.float32)

    coarse_array_1[:,:,0] = np.copy(image)
    coarse_array_2[:,:,0] = np.copy(image)

    filter_test = np.array([ 0.0, 1/6, 2/3, 1/6, 0.0 ]) # from scipy.signal.bslpine with scale np.sqrt(2)

    #===========================================================================
    # Convolution 1
    for level in range(0, n_levels - 1):

        position_array_lvl = np.array(position_array) * np.power(2, level)
        position_array_lvl = position_array_lvl.astype(int)

        if filter.ndim == 2:
            with np.nditer(coarse_array_1[:,:,level + 1], flags = ['multi_index'], op_flags = ['readwrite']) as it:
                for coeff in it:
                    position_array_coeff = np.copy(position_array_lvl)
                    position_array_coeff[0] += it.multi_index[0]
                    position_array_coeff[1] += it.multi_index[1]
                    position_array_coeff = boundary_conditions(position_array_coeff, interp_im.shape, 'loop')

                    coeff += np.sum(coarse_array_1[position_array_coeff[0], position_array_coeff[1], level] * filter)

        elif filter.ndim == 1:

            temporary_copy = np.copy(coarse_array_1[:,:,level + 1])
            # Over X axis

            for x in range(0, image_x_size ):

                position_array_coeff = np.copy(position_array_lvl)
                position_array_coeff += x
                position_array_coeff = boundary_conditions(position_array_coeff, image.shape[0], 'loop')

                temporary_copy[x,:] = np.matmul(np.array([filter]), coarse_array_1[position_array_coeff, :, level])

            for y in range(0, image_y_size ):

                position_array_coeff = np.copy(position_array_lvl)
                position_array_coeff += y
                position_array_coeff = boundary_conditions(position_array_coeff, image.shape[1] , 'loop')

                coarse_array_1[:,y,level + 1] = np.matmul(np.array([filter]), temporary_copy[ :, position_array_coeff].T)[0]

    # Convolution 2
    for level in range(0, n_levels - 1):

        position_array_lvl = np.array(position_array) * np.power(2, level)
        position_array_lvl_ceil = np.ceil(position_array_lvl).astype(int)
        position_array_lvl_floor = np.floor(position_array_lvl).astype(int)

        if filter.ndim == 2:
            with np.nditer(coarse_array_2[:,:,level + 1], flags = ['multi_index'], op_flags = ['readwrite']) as it:
                for coeff in it:
                    position_array_coeff = np.copy(position_array_lvl)
                    position_array_coeff[0] += it.multi_index[0]
                    position_array_coeff[1] += it.multi_index[1]
                    position_array_coeff = boundary_conditions(position_array_coeff, interp_im.shape, 'loop')

                    coeff += np.sum(coarse_array_2[position_array_coeff[0], position_array_coeff[1], level] * filter_test)

        elif filter.ndim == 1:

            temporary_copy = np.copy(coarse_array_2[:,:,level])

            for x in range(0, image_x_size):

                position_array_ceil_coeff = np.copy(position_array_lvl_ceil)
                position_array_ceil_coeff += x
                position_array_ceil_coeff = boundary_conditions(position_array_ceil_coeff, image_x_size, 'loop')

                position_array_floor_coeff = np.copy(position_array_lvl_floor)
                position_array_floor_coeff += x
                position_array_floor_coeff = boundary_conditions(position_array_floor_coeff, image_x_size, 'loop')

                temporary_copy[x,:] = np.matmul(np.array([filter_test]),\
                            0.5 * (coarse_array_2[position_array_ceil_coeff, :, level]\
                            + coarse_array_2[position_array_floor_coeff, :, level]))

            for y in range(0, image_y_size):

                position_array_ceil_coeff = np.copy(position_array_lvl_ceil)
                position_array_ceil_coeff += y
                position_array_ceil_coeff = boundary_conditions(position_array_ceil_coeff, image_y_size, 'loop')

                position_array_floor_coeff = np.copy(position_array_lvl_floor)
                position_array_floor_coeff += y
                position_array_floor_coeff = boundary_conditions(position_array_floor_coeff, image_y_size, 'loop')

                #pdb.set_trace()

                coarse_array_2[:,y,level + 1] = np.matmul(np.array([filter_test]), \
                            0.5 * (temporary_copy[ :, position_array_ceil_coeff] \
                            + temporary_copy[ :, position_array_floor_coeff]).T)[0]

    voice = 0
    stds = []
    for level in range(0, n_levels - 1):

        final_wavelet_array[:,:,voice] = coarse_array_1[:,:,level] - coarse_array_2[:,:,level+1]
        final_coarse_array[:,:,voice] = coarse_array_1[:,:,level]
        print(level, voice, np.mean(final_wavelet_array[:,:,voice]), np.std(final_wavelet_array[:,:,voice]))
        stds.append(np.std(final_coarse_array[:,:,voice]))
        voice += 1

        final_wavelet_array[:,:,voice] = coarse_array_2[:,:,level+1] - coarse_array_1[:,:,level+1]
        final_coarse_array[:,:,voice] = coarse_array_2[:,:,level+1]
        print(level, voice, np.mean(final_wavelet_array[:,:,voice]), np.std(final_wavelet_array[:,:,voice]))
        stds.append(np.std(final_coarse_array[:,:,voice]))
        voice += 1

    final_wavelet_array[:,:,-1] = np.copy(coarse_array_2[:,:,-1])

    return final_coarse_array, final_wavelet_array, stds

def interlaced_gaus(image, n_levels, n_voices, verbose = False):
    '''
    Interlaced à trous algorithm, doc to do
    '''
    #===========================================================================
    # Exceptions
    if image.ndim != 2:
        raise DawisDimensionError('Not an image -->', image.ndim, 2)
    image_x_size = image.shape[0]
    image_y_size = image.shape[1]

    #===========================================================================
    # Arrays
    final_coarse_array = np.zeros((image_x_size, image_y_size, 2 * n_levels), dtype = np.float32)
    final_wavelet_array = np.zeros((image_x_size, image_y_size, 2 * n_levels), dtype = np.float32)

    #===========================================================================
    # Convolution 0

    coarse_array_1 = np.zeros((image_x_size, image_y_size, n_levels), dtype = np.float32)
    coarse_array_2 = np.zeros((image_x_size, image_y_size, n_levels), dtype = np.float32)

    coarse_array_1[:,:,0] = np.copy(image)

    coarse_array_2[:,:,0] = gaussian_filter(image, sigma = 0.5 * np.sqrt(2))
    #coarse_array_2[:,:,0] = interp_im


    #===========================================================================
    # Convolution
    for level in range(0, n_levels - 1):
        coarse_array_1[:,:,level+1] = gaussian_filter(coarse_array_1[:,:,0], sigma = 2**(level+1))
        coarse_array_2[:,:,level+1] = gaussian_filter(coarse_array_2[:,:,0], sigma = 2**(level+1) * np.sqrt(2))


    voice = 0
    stds = []
    for level in range(0, n_levels - 1):

        final_wavelet_array[:,:,voice] = coarse_array_1[:,:,level] - coarse_array_2[:,:,level]
        final_coarse_array[:,:,voice] = coarse_array_1[:,:,level]
        print(level, voice, np.mean(final_wavelet_array[:,:,voice]), np.std(final_wavelet_array[:,:,voice]))
        stds.append(np.std(final_wavelet_array[:,:,voice]))
        voice += 1

        final_wavelet_array[:,:,voice] = coarse_array_2[:,:,level] - coarse_array_1[:,:,level+1]
        final_coarse_array[:,:,voice] = coarse_array_2[:,:,level]
        print(level, voice, np.mean(final_wavelet_array[:,:,voice]), np.std(final_wavelet_array[:,:,voice]))
        stds.append(np.std(final_wavelet_array[:,:,voice]))
        voice += 1

    final_wavelet_array[:,:,-1] = np.copy(coarse_array_2[:,:,-1])

    return final_coarse_array, final_wavelet_array, stds

def interlaced_spline(image, n_levels, n_voices, verbose = False):

    '''
    Interlaced à trous algorithm, doc to do
    '''
    #===========================================================================
    # Exceptions
    if image.ndim != 2:
        raise DawisDimensionError('Not an image -->', image.ndim, 2)
    image_x_size = image.shape[0]
    image_y_size = image.shape[1]

    #===========================================================================
    # Arrays
    final_coarse_array = np.zeros((image_x_size, image_y_size, 2 * n_levels), dtype = np.float32)
    final_wavelet_array = np.zeros((image_x_size, image_y_size, 2 * n_levels), dtype = np.float32)

    final_coarse_array[:,:,0] = np.copy(image)

    #===========================================================================
    # Convolution
    z = 0
    for level in np.arange(0, n_levels - 1, 0.5):

        x = np.arange( -1 * 2**level, 0.5 + 2**level, 1)

        bspl = bspline(x / 2**level, 3) / 2**level
        n = x.size
        bspl2d = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                bspl2d[i,j] = bspl[i] * bspl[j]

        #pdb.set_trace()
        final_coarse_array[:,:,z + 1] = convolve2d(final_coarse_array[:,:,z], bspl2d, 'same', 'wrap')

        z += 1

    #===========================================================================
    stds = []
    for level in range(0, 2 * n_levels - 1):

        final_wavelet_array[:,:,level] = final_coarse_array[:,:,level] - final_coarse_array[:,:,level+1]
        stds.append(np.std(final_wavelet_array[:,:,level]))

    final_wavelet_array[:,:,-1] = np.copy(final_coarse_array[:,:,-1])

    return final_coarse_array, final_wavelet_array, stds

def interlaced_mix_atrous_spline(image, n_levels, n_voices, verbose = False):

    '''
    Interlaced à trous algorithm, doc to do
    '''
    #===========================================================================
    # Exceptions
    if image.ndim != 2:
        raise DawisDimensionError('Not an image -->', image.ndim, 2)
    image_x_size = image.shape[0]
    image_y_size = image.shape[1]

    #===========================================================================
    # Arrays
    final_coarse_array = np.zeros((image_x_size, image_y_size, 2 * n_levels), dtype = np.float32)
    final_wavelet_array = np.zeros((image_x_size, image_y_size, 2 * n_levels), dtype = np.float32)

    final_coarse_array[:,:,0] = np.copy(image)

    #===========================================================================
    # Convolution
    z = 0
    for level in np.arange(0, n_levels - 1, 0.5):

        print(z)
        x = np.arange( -1 * 2**level, 0.5 + 2**level, 1)

        bspl = bspline(x / 2**level, 3) / 2**level
        n = x.size
        bspl2d = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                bspl2d[i,j] = bspl[i] * bspl[j]

        #pdb.set_trace()
        final_coarse_array[:,:,z + 1] = convolve2d(final_coarse_array[:,:,z], bspl2d, 'same', 'wrap')

        z += 1

    #===========================================================================
    stds = []
    for level in range(0, 2 * n_levels - 1):

        final_wavelet_array[:,:,level] = final_coarse_array[:,:,level] - final_coarse_array[:,:,level+1]
        print(level, np.mean(final_wavelet_array[:,:,level]), np.std(final_wavelet_array[:,:,level]))
        stds.append(np.std(final_wavelet_array[:,:,level]))

    final_wavelet_array[:,:,-1] = np.copy(final_coarse_array[:,:,-1])

    return final_coarse_array, final_wavelet_array, stds

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if __name__ == '__main__':

    from numpy.random import normal
    from astropy.io import fits
    import sys

    im = normal(0, 1,(7500, 7500))
    #im = np.zeros((1000, 1000))
    #im[500,500] = 1

    #im = fits.getdata('/home/ellien/wavelets/MW_star_acs.fits')
    #im = fits.getdata('/home/ellien/wavelets/A1365.rebin.fits')
    #im = fits.getdata('/home/ellien/devd/tests/cat_gal_004_z_0.1_Ficl_0.4_Re_050_noise_megacam.fits')
    #nx, ny = np.shape(im)
    #anx, any = int(nx * np.sqrt(2) / 2), int(ny * np.sqrt(2) / 2)
    #im = congrid(im, (1024, 1024), method = 'spline')
    n_levels = 5
    '''
    # spline lissés
    startTime = datetime.now()
    bspl = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
    cbspl_new, wbspl_new = interlaced_atrous_congrid(image = im, n_levels = n_levels, filter = bspl, n_voices = 2)
    cbspl_new = np.delete(cbspl_new, obj = 1, axis = 2 ) # delete coarse scale 2^0.5
    wbspl_new = np.delete(wbspl_new, obj = 0, axis = 2 ) # delete coarse scale 2^0.5
    wbspl_new[:,:,0] = cbspl_new[:,:,0] - cbspl_new[:,:,1]
    print(datetime.now() - startTime)

    dcc_new = datacube(cbspl_new)
    #dcc_new.waveplot(name = 'coarse', ncol = 7)
    dcw_new = datacube(wbspl_new)
    #dcw_new.waveplot(name = 'wav', ncol = 7, cmap = 'hot' )
    '''
    # a trous
    bspl = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
    startTime = datetime.now()
    cbspl_old, wbspl_old = atrous(image = im, n_levels = n_levels, filter = bspl)
    stds_old = np.std(wbspl_old)
    print('a trous', datetime.now() - startTime)
    '''
    #ovwav
    startTime = datetime.now()
    cbspl_ov = DL_atrou(im, n_levels, WAV_COEFF = False)
    wbspl_ov = DL_atrou(im, n_levels, WAV_COEFF = True)
    print('ov_wav', datetime.now() - startTime)
    
    #ovwav
    startTime = datetime.now()
    cbspl_ov = DL_atrou(im, n_levels, WAV_COEFF = False)
    wbspl_ov = DL_atrou(im, n_levels, WAV_COEFF = True)
    print('ov_wav', datetime.now() - startTime)
    '''
    # pyMOresane
    startTime = datetime.now()
    wbspl_ov, cbspl_mor = ser_iuwt_decomposition(im, scale_count = n_levels, scale_adjust = 0, store_smoothed = True)
    print('ser pyMOresane', datetime.now() - startTime)

    # pyMOresane
    startTime = datetime.now()
    wbspl_ov, cbspl_mor = mp_iuwt_decomposition(im, scale_count = n_levels, scale_adjust = 0, store_smoothed = True, core_count = 8)
    print('mp pyMOresane', datetime.now() - startTime)
    

    '''
    dcc_old = datacube(cbspl_old)
    #dcc_old.waveplot(name = 'coarse', ncol = 7, norm=matplotlib.colors.SymLogNorm(10**-5))
    dcw_old = datacube(wbspl_old)
    #dcw_old.waveplot(name = 'wav', ncol = 7, cmap = 'hot', norm=matplotlib.colors.SymLogNorm(10**-5) )

    # plots stds
    x_old = np.logspace(start = 0, stop = n_levels, num = n_levels + 1, base = 2)
    x_new = np.logspace(start = 0, stop = n_levels, num = 2 * n_levels + 1, base = 2)
    x_new = np.delete(x_new, 1)

    plt.figure()
    plt.title('stds of coarse planes')
    plt.plot(x_new[:-1], np.std(cbspl_new, axis = (0,1)), 'ro', linestyle = '-', label = 'interlaced à trous')
    plt.plot(x_old, np.std(cbspl_old, axis = (0,1)), 'bo', linestyle = '-', label = 'regular à trous')
    #plt.plot(x_old[:-1], np.std(cbspl_ov, axis = (0,1)), 'go', linestyle = '-')
    plt.xscale('log', base = 2)
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('means of coarse planes')
    plt.plot(x_new[:-1], np.mean(cbspl_new, axis = (0,1)), 'ro', linestyle = '-', label = 'interlaced à trous')
    plt.plot(x_old, np.mean(cbspl_old, axis = (0,1)), 'bo', linestyle = '-', label = 'regular à trous')
    #plt.plot(x_old[:-1], np.mean(cbspl_ov, axis = (0,1)), 'go', linestyle = '-')
    plt.xscale('log', base = 2)
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('stds of wavelet planes')
    plt.plot(x_new[:-1], np.std(wbspl_new, axis = (0,1)), 'ro', linestyle = '-', label = 'interlaced à trous')
    plt.plot(x_old[:-1], np.std(wbspl_old, axis = (0,1)), 'bo', linestyle = '-', label = 'regular à trous')
    #plt.plot(x_old[:-1], np.std(wbspl_ov, axis = (0,1)), 'go', linestyle = '-')
    plt.xscale('log', base = 2)
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('means of wavelet planes')
    plt.plot(x_new[:-1], np.mean(wbspl_new, axis = (0,1)), 'ro', linestyle = '-', label = 'interlaced à trous')
    plt.plot(x_old[:-1], np.mean(wbspl_old, axis = (0,1)), 'bo', linestyle = '-', label = 'regular à trous')
    #plt.plot(x_old[:-1], np.mean(wbspl_ov, axis = (0,1)), 'go', linestyle = '-')
    plt.xscale('log', base = 2)
    plt.legend()
    plt.show()

    print(np.std(wbspl_new, axis = (0,1)))

    #bspl = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
    #startTime = datetime.now()
    #cbspl, wbspl = atrous(image = im, n_levels = 6, filter = bspl)

    #wp_new_check = np.zeros(wp_old_check.shape)
    #for level in range(0, n_levels - 1):
    #   wp_new_check[:,:,level] = dcc.array[:,:,2*level] - dcc.array[:,:,2*(level+1)]

    #plt.figure()
    #plt.title('diff new/old a trous')
    #plt.imshow( 100 * (wp_new_check[:,:,0] - wp_old_check[:,:,0]) / wp_new_check[:,:,0], vmax = 1, vmin = -1 )
    #plt.show()

    # print(im.shape)
    # starlet = 1 / 256. * np.array([ [1,  4,  6,  4, 1],
    #                                 [4, 16, 24, 16, 4],
    #                                 [6, 24, 36, 24, 6],
    #                                 [4, 16, 24, 16, 4],
    #                                 [1,  4,  6,  4, 1] ])
    # startTime = datetime.now()
    # cstar, wstar = atrous(image = im, n_levels = 6, filter = starlet)
    # print(datetime.now() - startTime)

    #bspl = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
    #startTime = datetime.now()
    #cbspl, wbspl = atrous(image = im, n_levels = 6, filter = bspl)
    #print(datetime.now() - startTime)

    # haar = np.array([ 0.5, 0, 0.5 ])
    # startTime = datetime.now()
    # chaar, whaar = atrous(image = im, n_levels = 6, filter = haar)
    # print(datetime.now() - startTime)

    # diff_bspl_starlet = datacube(wstar - wbspl)
    # diff_bspl_starlet.waveplot(name = 'Diff btw starlet and bspl', save_path = '/home/ellien/devd/tests/diff_bspl_starlet.pdf')

    # diff_bspl_haar = datacube(whaar - wbspl)
    # diff_bspl_haar.waveplot(name = 'Diff btw haar and bspl', save_path = '/home/ellien/devd/tests/diff_bspl_haar.pdf')
    '''