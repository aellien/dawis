#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# This file is part of DAWIS.
#
# NAME : synthesis_by_analysis.py
# AUTHOR : Amael Ellien
# LAST MODIFICATION : 07/2021
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import os
import numpy as np
import logging
from astropy.io import fits
from datetime import datetime
import sys as sys
import pdb
from dawis.make_regions import *
from dawis.datacube import *
from dawis.wavelet_transforms import bspl_atrous
from dawis.hard_threshold import hard_threshold
from dawis.label_regions import label_regions
from dawis.anscombe_transforms import *
from dawis.make_regions import *
from dawis.make_interscale_trees import *
from dawis.restore_objects import *
from dawis.gif import *

def synthesis_by_analysis(indir, infile, outdir, n_cpus = 3, starting_level = 2, tau = 0.8, n_levels = None,\
                                gamma = 0.2, min_span = 2, max_span = 3, lvl_sep_big = 6, \
                                extent_sep = 0.1, lvl_sep_lin = 2, ceps = 1E-3, \
                                max_iter = 500, data_dump = True, gif = True):

    #===========================================================================
    # Check infile extension
    if infile[-5:] != '.fits':
        raise DawisWrongType('Incorrect file extension -->', infile[:-5], '.fits')

    # Logs
    outpath = os.path.join( outdir, infile[:-5] )
    namelog = ''.join(( ( outpath, '.log.txt') ))
    file_handler = logging.FileHandler( filename = namelog, mode = 'w' )
    stdout_handler = logging.StreamHandler( sys.stdout )
    logging.basicConfig( level = logging.DEBUG, handlers = [ file_handler, stdout_handler ])
    logging.info(datetime.now())
    StartTimeAlgo = datetime.now()
    logging.info('DAWIS\nInfile = %s\nInput Directory = %s\nOutput Directory = %s\nN_cpus = %d\nTau = %f\nGamma = %f\nmin_span = %d\nmax_span = %d\nConvergence Epsilon = %f'\
                                    %(infile, indir, outdir, n_cpus, tau, gamma, min_span, max_span, ceps))

    #  FITS header & image properties
    hdu = fits.open(os.path.join(indir, infile))
    im  = hdu[0].data
    header = hdu[0].header
    if not n_levels:
        n_levels = np.int(np.min(np.floor(np.log2(im.shape))))
    logging.info('Image of size %d x %d\n--> n_levels = %d, corresponding to maximum size %d\n' \
                                %(im.shape[0], im.shape[1], n_levels, 2**n_levels) )

    # Noise properties
    sigma, mean, gain = pg_noise_bissection(im, max_err = 1E-3, n_sigmas = 3)
    logging.info('Noise properties: sigma = %.6f, mean = %.6f, gain = %.6f\n' %(sigma, mean, gain))

    #===========================================================================

    res = np.copy(im)
    rec = np.zeros(im.shape)
    rec_lvl = np.zeros((im.shape[0], im.shape[1], n_levels))
    cparl = []
    it = 1
    std = 1.

    for level in range( starting_level, n_levels ):

        normeps = 1.
        avnormeps = 1.
        window = []
        while avnormeps > ceps :

            start_time_it = datetime.now()
            logging.info('\n\n[ %s ] Level = %d Iteration = %d' %(datetime.now(), level, it))
            res[ res < 0. ] = 0.

            # Anscombe transform & thresholding
            aim = anscombe_transform(res, sigma, mean, gain)
            acdc, awdc = bspl_atrous(aim, level, header)
            sdc = hard_threshold(awdc, n_sigmas = 5)

            # Labels & true wavelet coefficients
            ldc = label_regions(sdc)
            cdc, wdc = bspl_atrous(res, level, header)

            # Regions of significance
            rl = make_regions_full_props(wdc, ldc, verbose = True)
            if not rl:
                break

            # Interscale trees
            itl = make_interscale_trees(rl, wdc, ldc, tau = tau, \
                                                      min_span = min_span, \
                                                      max_span = max_span, \
                                                      lvl_sep_big = lvl_sep_big, \
                                                      verbose = True)
            if not itl:
                break

            # Restoration of detected objects
            ol = restore_objects_default(itl, wdc,ldc, size_patch_small = 50, \
                                               extent_sep = extent_sep, \
                                               lvl_sep_lin = lvl_sep_lin, \
                                               lvl_sep_big = lvl_sep_big, \
                                               size_patch_big = 5, \
                                               size_big_objects = 512, \
                                               n_cpus = n_cpus )

            # Atom
            atom = np.zeros(res.shape)
            for object in ol:
                x_min, y_min, x_max, y_max = object.bbox
                atom[ x_min : x_max, y_min : y_max ] += object.image * gamma
                rec_lvl[ x_min : x_max, y_min : y_max, object.level ] += object.image * gamma

            # Update Residuals
            res -= atom
            rec += atom

            # Convergence
            flux_res = 1 - ( np.abs(np.sum(atom) - np.sum(np.sqrt(res**2)))) / np.sum(np.sqrt(res**2))
            flux_rec = 1 - ( np.abs(np.sum(atom) - np.sum(np.sqrt(rec**2)))) / np.sum(np.sqrt(rec**2))
            logging.info('Convergence : \nres = %f rec = %f' %( flux_res, flux_rec ))
            old_std = std
            std = np.std(res)
            eps = np.sqrt( ( old_std - std )**2 ) / np.sqrt( old_std**2 )
            normeps = eps *  level / len(ol)
            window.append(normeps)
            if len(window) > 5:window.pop(0)
            avnormeps = np.mean(window)
            cparl.append([ str(level), str(it), str(avnormeps), str(normeps), str(eps), str(flux_rec), str(flux_res), str(len(ol)), str( datetime.now() - start_time_it ) ])
            logging.info('Number Objects = %d, Normalized Epsilon = %f, Window Normalized Epsilon = %f' %( len(ol), normeps, avnormeps ))

            # Data Dump
            if data_dump:
                logging.info('Dumping data in %s' %(outdir) )
                wdc.to_fits( ''.join(( outpath, '.wdc.it%03d.fits' %(it)) ), overwrite = True)
                ldc.to_fits( ''.join(( outpath, '.ldc.it%03d.fits' %(it) )), overwrite = True)
                write_regions_to_pickle( rl, ''.join(( outpath, '.rl.it%03d.pkl' %(it) )), overwrite = True)
                write_interscale_trees_to_pickle( itl, ''.join(( outpath, '.itl.it%03d.pkl' %(it) )), overwrite = True)
                write_objects_to_pickle( ol, ''.join(( outpath, '.ol.it%03d.pkl' %(it) )), overwrite = True)

            if gif:
                sdc.histogram_noise( name = 'Noise histogram (lvl = 0)\nIteration %d'%(it), show = False, save_path = ''.join(( outpath, '.hist.it%03d.png' %(it))))
                plot_frame( level = level, it = it, nobj = len(ol), \
                                                    original_image = im, \
                                                    restored_image = rec, \
                                                    residuals = res, \
                                                    atom = atom, \
                                                    outpath = outpath )

            it += 1
            if it > max_iter:
                break

        if it > max_iter:
            break

    #===========================================================================
    # Write results to disk
    logging.info('Finished iterating.\nWriting results to disk in %s' %(outdir) )
    hdu_res = fits.PrimaryHDU( res, header = header )
    hdu_res.writeto( ''.join(( outpath, '.residuals.fits' )), overwrite = True )
    hdu_rec = fits.PrimaryHDU( rec, header = header )
    hdu_rec.writeto( ''.join(( outpath, '.restored.fits' )), overwrite = True )
    hdu_rec_lvl = fits.PrimaryHDU( rec_lvl, header = header )
    hdu_rec_lvl.writeto( ''.join(( outpath, '.sclrestored.fits' )), overwrite = True )

    with open( ''.join(( outpath, '.cpar.txt' )), 'w+') as conv:
        conv.write('# level, it, window_normeps, normeps, eps, n_obj, time_it\n')
        for l in cparl:
            conv.write('%s\n' %(' '.join(l)))

    plot_convergence( cparl = cparl, outpath = outpath, marker = 'o', linewidth = 2 )

    if gif:
        make_gif( framerate = 3, outpath = outpath )

    logging.info( datetime.now() - StartTimeAlgo )


def load_iteration( it, outpath ):
    '''doc todo'''
    wdc = wavelet_datacube.from_fits(''.join(( outpath, '.wdc.it%03d.fits' %(it)) ))
    ldc = label_datacube.from_fits( ''.join(( outpath, '.ldc.it%03d.fits' %(it) )))
    rl = read_regions_from_pickle( ''.join(( outpath, '.rl.it%03d.pkl' %(it) )))
    itl = read_interscale_trees_from_pickle( ''.join(( outpath, '.itl.it%03d.pkl' %(it) )))
    ol = read_objects_from_pickle( ''.join(( outpath, '.ol.it%03d.pkl' %(it) )))
    return wdc, ldc, rl, itl, ol

if __name__=="__main__":

    if len(sys.argv) < 4:
        raise SyntaxError("Insufficient arguments.")

    main(sys.argv[1], sys.argv[2], sys.argv[3], *sys.argv[4:])
