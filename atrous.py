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
from dawis.congrid import congrid
import logging

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def boundary_conditions(position_array, sizes, condition = 'prolongation' ):
    '''
    Border conditions, doc to do
    '''

    if np.ndim(position_array) not in [1, 3]:
        raise DawisDimensionError('Wrong dimensions for array -->', np.ndim(position_array), [1, 3])

    if np.ndim(sizes) not in [0, 1]:
        raise DawisDimensionError('Wrong dimensions for array -->', np.ndim(sizes), [0, 1])

    if type(condition) != str:
        raise DawisWrongType('Wrong type -->', type(condition), 'str')

    allowed_conditions = ['mirror', 'prolongation', 'loop' ]

    if condition not in allowed_conditions:
        raise DawisValueError('Not an allowed condition -->', condition, allowed_conditions)

    if condition == 'prolongation':

        position_array[np.where(position_array < 0)] = 0

        if ( np.ndim(position_array) == 3 ) & ( np.ndim(sizes) == 1 ):
            position_array[0, np.where(position_array[0,:] > sizes[0] - 1)] = sizes[0] - 1
            position_array[1, np.where(position_array[1,:] > sizes[1] - 1)] = sizes[1] - 1
        else:
            position_array[np.where(position_array > sizes - 1)] = sizes - 1

    if condition == 'mirror':

        position_array[np.where(position_array < 0)] -= 2 * position_array[np.where(position_array < 0)]

        if ( np.ndim(position_array) == 3 ) & ( np.ndim(sizes) == 1 ):
            position_array[0][np.where(position_array[0,:] > sizes[0] - 1)] -= 2 * ( position_array[0][np.where(position_array[0,:] > sizes[0] - 1)] - sizes[0] )
            position_array[1][np.where(position_array[1,:] > sizes[1] - 1)] -= 2 * ( position_array[1][np.where(position_array[1,:] > sizes[1] - 1)] - sizes[1] )
        else:
            position_array[np.where(position_array > sizes - 1)] -= sizes - ( position_array[np.where(position_array > sizes - 1 )] - sizes )

    if condition == 'loop':

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
        position_array_lvl = position_array_lvl.astype(np.int)

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
    final_coarse_array = np.zeros((image_x_size, image_y_size, 2 * n_levels), dtype = np.float32)
    final_wavelet_array = np.zeros((image_x_size, image_y_size, 2 * n_levels), dtype = np.float32)

    #===========================================================================
    # Convolution 0

    anx, any = np.int(image_x_size * np.sqrt(2) / 2), \
               np.int(image_y_size * np.sqrt(2) / 2)

    coarse_array_1 = np.zeros((image_x_size, image_y_size, n_levels), dtype = np.float32)

    coarse_array_2 = np.zeros((image_x_size, image_y_size, n_levels), dtype = np.float32)
    congrid_coarse_array = np.zeros((anx, \
                                     anx, \
                                     n_levels), dtype = np.float32)

    coarse_array_1[:,:,0] = np.copy(image)

    congrid_coarse_array[:,:,0] = congrid(image, (anx, any), method = 'spline')
    #coarse_array_2[:,:,0] = gaussian_filter(image, sigma = 0.5 * np.sqrt(2))
    coarse_array_2[:,:,0] = congrid(congrid_coarse_array[:,:,0], (image_x_size, image_y_size), method = 'spline')

    #===========================================================================
    # Convolution 1
    for level in range(0, n_levels - 1):

        position_array_lvl = np.array(position_array) * np.power(2, level)
        position_array_lvl = position_array_lvl.astype(np.int)

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
        position_array_lvl = position_array_lvl.astype(np.int)
        #print(position_array * np.power(2, level).astype(np.int) )

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

            coarse_array_2[:,:,level+1] = congrid(congrid_coarse_array[:,:,level + 1], (image_x_size, image_y_size), method = 'spline')

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
        position_array_lvl = position_array_lvl.astype(np.int)

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
        position_array_lvl_ceil = np.ceil(position_array_lvl).astype(np.int)
        position_array_lvl_floor = np.floor(position_array_lvl).astype(np.int)

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
        position_array_lvl = position_array_lvl.astype(np.int)

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
        position_array_lvl_ceil = np.ceil(position_array_lvl).astype(np.int)
        position_array_lvl_floor = np.floor(position_array_lvl).astype(np.int)

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
    from DL_reconstruct import DL_atrou

    im = normal(0, 1,(1024, 1024))
    #im = np.zeros((1000, 1000))
    #im[500,500] = 1

    #im = fits.getdata('/home/ellien/wavelets/MW_star_acs.fits')
    #im = fits.getdata('/home/ellien/wavelets/A1365.rebin.fits')
    #im = fits.getdata('/home/ellien/devd/tests/cat_gal_004_z_0.1_Ficl_0.4_Re_050_noise_megacam.fits')
    #nx, ny = np.shape(im)
    #anx, any = np.int(nx * np.sqrt(2) / 2), np.int(ny * np.sqrt(2) / 2)
    #im = congrid(im, (1024, 1024), method = 'spline')
    n_levels = 10

    # spline lissés
    startTime = datetime.now()
    cbspl_new, wbspl_new, stds_new = interlaced_spline(image = im, n_levels = n_levels, n_voices = 2)
    print(datetime.now() - startTime)

    dcc_new = datacube(cbspl_new)
    #dcc_new.waveplot(name = 'coarse', ncol = 7)
    dcw_new = datacube(wbspl_new)
    #dcw_new.waveplot(name = 'wav', ncol = 7, cmap = 'hot' )

    # a trous
    bspl = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
    startTime = datetime.now()
    cbspl_old, wbspl_old, stds_old = atrous(image = im, n_levels = n_levels, filter = bspl)
    print(datetime.now() - startTime)

    dcc_old = datacube(cbspl_old)
    dcc_old.waveplot(name = 'coarse', ncol = 7, norm=matplotlib.colors.SymLogNorm(10**-5))
    dcw_old = datacube(wbspl_old)
    dcw_old.waveplot(name = 'wav', ncol = 7, cmap = 'hot', norm=matplotlib.colors.SymLogNorm(10**-5) )

    #ovwav
    startTime = datetime.now()
    cbspl_ov = DL_atrou(im, n_levels, WAV_COEFF = False)
    wbspl_ov = DL_atrou(im, n_levels, WAV_COEFF = True)
    print(datetime.now() - startTime)

    # plots stds
    x_old = np.logspace(start = 0, stop = n_levels, num = n_levels + 1, base = 2)
    x_new = np.logspace(start = 0, stop = n_levels, num = 2 * n_levels + 1, base = 2)

    plt.figure()
    plt.title('stds of coarse planes')
    plt.plot(x_old[:-1], np.std(cbspl_old, axis = (0,1)), 'bo', linestyle = '-')
    plt.plot(x_new[:-1], np.std(cbspl_new, axis = (0,1)), 'ro', linestyle = '-')
    plt.plot(x_old[:-1], np.std(cbspl_ov, axis = (0,1)), 'go', linestyle = '-')
    plt.xscale('log', basex = 2)
    plt.show()

    plt.figure()
    plt.title('stds of coarse planes')
    plt.plot(x_old[:-2], np.std(wbspl_old, axis = (0,1)), 'bo', linestyle = '-')
    plt.plot(x_new[:-2], np.std(wbspl_new, axis = (0,1)), 'ro', linestyle = '-')
    plt.plot(x_old[:-2], np.std(wbspl_ov, axis = (0,1)), 'go', linestyle = '-')
    plt.xscale('log', basex = 2)
    plt.show()

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
