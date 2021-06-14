#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# This file is part of DAWIS.
#
# NAME : gif.py
# AUTHOR : Amael Ellien
# LAST MODIFICATION : 07/2021
# ffmpeg -framerate 5 -i img-%02d.png video.avi

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from matplotlib.colors import LogNorm
from subprocess import run
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import numpy as np
from astropy.visualization import LinearStretch, LogStretch
from astropy.visualization import ZScaleInterval, MinMaxInterval
from astropy.visualization import ImageNormalize
from dawis.restore_objects import *
from astropy.io import fits

def plot_frame( level, it, nobj, original_image, restored_image, residuals, atom, outpath = '' ):

    '''doc to do'''

    fig = plt.figure(dpi = 500)
    ax = fig.subplots(2, 2, sharex = True, sharey = True, \
                                gridspec_kw = { 'hspace': 0.1, 'wspace': 0.1 })
    fig.suptitle('Level %d\nIteration %d\n%d objects' %(level, it, nobj))

    axim = ax[0][0].imshow(original_image,  norm = ImageNormalize( original_image, \
                                            interval = ZScaleInterval(), \
                                            stretch = LinearStretch()), \
                                            origin = 'lower', \
                                            cmap = 'binary' )
    cax = plt.colorbar(axim, ax = ax[0][0], aspect = 10, pad = 0, \
                                    orientation = 'vertical')
    cax.ax.tick_params(labelsize = 5)
    ax[0][0].set_title('Original')

    axim = ax[0][1].imshow(residuals, norm = ImageNormalize( residuals, \
                                      interval = ZScaleInterval(),
                                      stretch = LinearStretch()), \
                                      origin = 'lower', \
                                      cmap = 'binary' )

    cax = plt.colorbar(axim, ax = ax[0][1], aspect = 10, pad = 0, \
                                    orientation = 'vertical')
    cax.ax.tick_params(labelsize = 5)
    ax[0][1].set_title('Residuals')

    axim = ax[1][0].imshow(restored_image,  norm = ImageNormalize( original_image, \
                                            interval = ZScaleInterval(),
                                            stretch = LinearStretch()), \
                                            origin = 'lower', \
                                            cmap = 'binary' )

    cax = plt.colorbar(axim, ax = ax[1][0], aspect = 10, pad = 0, \
                                    orientation = 'vertical')
    cax.ax.tick_params(labelsize = 5)
    ax[1][0].set_title('Restored')

    axim = ax[1][1].imshow(atom, norm = ImageNormalize( atom, \
                                 interval = MinMaxInterval(),
                                 stretch = LogStretch()), \
                                 origin = 'lower', \
                                  cmap = 'binary' )

    cax = plt.colorbar(axim, ax = ax[1][1], aspect = 10, pad = 0, \
                                    orientation = 'vertical')
    cax.ax.tick_params(labelsize = 5)
    ax[1][1].set_title('Atom')

    plt.savefig(''.join(( outpath, '.frame.it%03d.png' %(it) )), format = 'png' )
    plt.close()

def make_gif( framerate, outpath ):

    '''doc to do'''

    args_ffmpeg = [ 'ffmpeg', '-framerate', str(framerate), '-i', \
                ''.join(( outpath, '.frame.it%03d.png' )), \
                ''.join(( outpath, '.run.avi' )) ]

    run( args_ffmpeg )

    args_ffmpeg = [ 'ffmpeg', '-framerate', str(framerate), '-i', \
                ''.join(( outpath, '.hist.it%03d.png' )), \
                ''.join(( outpath, '.hist.avi' )) ]

    run( args_ffmpeg )

    #args_rm = [ 'rm', ''.join(( outpath, '.frame*.png' )) ]
    #run( args_rm )

def remake_frames( imagepath, outpath, gamma, n_it ):

    #outpath = '/home/ellien/devd/tests/full/tidal_group_f814w_sci.rl.it'


    image = fits.getdata(imagepath)
    res = np.copy(image)
    rec = np.zeros(image.shape)
    for it in range(1, n_it + 1):
        atom = np.zeros(image.shape)
        ol = read_objects_from_pickle(outpath+'.frame.it%03d'%(it)+'.pkl')
        for object in ol:
            x_min, y_min, x_max, y_max = object.bbox
            atom[ x_min : x_max, y_min : y_max ] += object.image * gamma
        res -= atom
        rec += atom
        level = np.max( [ x.level for x in ol ] )
        plot_frame( level = level, it = it, nobj = len(ol), original_image = image,  \
                                    restored_image = rec, residuals = res, \
                                    atom = atom, outpath = outpath )

def plot_convergence( cparl, outpath, **kwargs ):
    '''doc to do'''

    level = []
    it = []
    avnormeps = []
    normeps = []
    eps = []
    flux_rec = []
    flux_res = []
    nobj = []
    time = []

    for cpar in cparl:

        level.append(int(cpar[0]))
        it.append(int(cpar[1]))
        avnormeps.append(float(cpar[2]))
        normeps.append(float(cpar[3]))
        eps.append(float(cpar[4]))
        flux_rec.append(float(cpar[5]))
        flux_res.append(float(cpar[6]))
        nobj.append(int(cpar[7]))
        time.append(cpar[8])

    fig = plt.figure(figsize = (9, 6), dpi = 500)
    plt.plot(it, avnormeps, label = 'window normalized eps', **kwargs)
    plt.plot(it, normeps, label = 'normalized eps', **kwargs)
    plt.plot(it, eps, label = 'eps', **kwargs)
    plt.plot(it, flux_rec, label = 'flux restored fraction', **kwargs)
    plt.plot(it, flux_res, label = 'flux residuals fraction', **kwargs)
    plt.grid(linestyle='-', linewidth=2, alpha=0.2)

    plt.legend()
    plt.yscale('log')
    plt.savefig(''.join(( outpath, '.conv.pdf' %(it) )), format = 'pdf' )
    plt.close()
