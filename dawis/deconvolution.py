#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# This file is part of DAWIS.
#
# NAME : make_interscale_trees.py
# AUTHOR : Amael Ellien
# LAST MODIFICATION : 07/2021
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import logging
import math
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift

def pad(img, paddedsize, mode):
    """ pad image to paddedsize

    Args:
        img ([type]): image to pad 
        paddedsize ([type]): size to pad to 
        mode ([type]): one of the np.pad modes

    Returns:
        [nd array]: padded image
    """
    padding = tuple(map(lambda i,j: ( math.ceil((i-j)/2), math.floor((i-j)/2) ),paddedsize,img.shape))
    return np.pad(img, padding,mode), padding


def unpad(padded, imgsize):
    """Crop padded back to imgsize.

    Args:
        padded (numpy.ndarray or cupy.ndarray): The padded array.
        imgsize (tuple): The target size of the unpadded array.

    Returns:
        numpy.ndarray or cupy.ndarray: The unpadded array.
    """
    padding = tuple(map(lambda i,j: (math.ceil((i-j)/2), math.floor((i-j)/2)), padded.shape, imgsize))
    slices = tuple(slice(p[0], p[0]+s) for p, s in zip(padding, imgsize))

    return padded[slices]

def richardson_lucy_np(image, psf, num_iters, noncirc=False, mask=None):
    """ Deconvolves an image using the Richardson-Lucy algorithm with non-circulant option and option to mask bad pixels, uses numpy

    Note: NumPy FFT functions always cast 32 bit arrays to float64, so passing in 32 bit arrays to save memory will not work. 

    Args:
        image [numpy float array]: the image to be deconvolved 
        psf [numpy float array]: the point spread function
        num_iters (int): the number of iterations to perform
        noncirc (bool, optional): If true use non-circulant edge handling. Defaults to False.
        mask (numpy float array, optional): If not None, use this mask to mask image pixels that should not be considered in the deconvolution. Defaults to None.
            'bad' pixels will be zeroed during the deconvolution and then replaced with the original value after the deconvolution.

    Returns:
        [numpy float array]: the deconvolved image
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

def B2spline(xin, order = 0):
    x = xin / 2**order
    return ( abs(x-2)**3 - 4*abs(x-1)**3 + 6 * abs(x)**3 - 4 * abs(x+1)**3 + abs(x+2)**3 ) / 16.

def twod_B2spline(xsize, ysize, order = 0):
    
    xc = int(xsize / 2.)
    yc = int(ysize / 2.)
    
    xa = np.linspace(- xc, xc, xsize )
    ya = np.linspace(- yc, yc, ysize )
    
    b2s_x = B2spline(xa, order = order)
    b2s_y = B2spline(ya, order = order)
    
    b2s_2d = np.outer(b2s_x, b2s_y)
    
    return b2s_2d
    