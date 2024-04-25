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
import glob as glob
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
from dawis.inpainting import sample_noise, inpaint_with_gaussian_noise
from dawis.detect_and_deblend import ms_detect_and_deblend

@ray.remote
def synthesis_by_analysis(indir, infile, outdir, n_cpus = 1, starting_level = 2, tau = 0.8, n_levels = None, n_sigmas = 5, deblend_contrast = 0.1,\
                                gamma = 0.2, min_span = 2, max_span = 3, lvl_sep_big = 6, rm_gamma_for_big = False, lvl_deblend = 3, \
                                extent_sep = 0.1, ecc_sep = 0.95, lvl_sep_lin = 2, lvl_sep_op = 3, ceps = 1E-3, scale_lvl_eps = 1, conditions = 'loop', deconv = False,\
                                max_iter = 500, size_patch = 100, inpaint_res = True, data_dump = True, gif = True, iptd_sigma = 3, resume = True):

    # Setup Ray for parallelisation
    # If there is no head ray cluster setup, create one internal to dawis run
    intern_ray_flag = False
    if (n_cpus > 1) & (ray.is_initialized == False):
        ray.init(num_cpus = n_cpus)
        inter_ray_flag = True
    
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
        n_levels = int(np.min(np.floor(np.log2(im.shape))))
    logging.info('Image of size %d x %d\n--> n_levels = %d, corresponding to maximum size %d pixels\n' \
                                %(im.shape[0], im.shape[1], n_levels, 2**n_levels) )

    # Noise properties
    noise_pixels, valmax = sample_noise(im, n_sigmas = 3, bins = 200)
    mean, sigma = np.mean(noise_pixels), np.std(noise_pixels)
    logging.info('Noise properties for inpainting: sigma = %1.3e, mean = %1.3e\n' %(sigma, mean))
    im -= mean #remove background

    # Anscombe transform parameters
    ans_im = np.copy(im)
    C = np.nanmin(ans_im)
    ans_im -= C # Set minimum pixel value to 0 for Anscombe transform
    sigma_ans, mean_ans, gain_ans = pg_noise_bissection(ans_im, max_err = 1E-4, n_sigmas = 3)
    logging.info('Noise parametric values for Anscombe transform: sigma = %1.3e, mean = %1.3e, gain = %1.3e\n' %(sigma_ans, mean_ans, gain_ans))
    
    #===========================================================================

    res = np.copy(im)
    dei = np.zeros(im.shape)
    rec = np.zeros(im.shape)
    rec_lvl = np.zeros((im.shape[0], im.shape[1], n_levels))
    cparl = []
    it = 1
    std = 1
    
    for level in range( starting_level, n_levels ):

        normeps = 1.
        avnormeps = 1.
        window = []
        while avnormeps > ceps :

            start_time_it = datetime.now()
            logging.info('[ %s ] Level = %d Iteration = %d' %(datetime.now(), level, it))

            # inpaint bad reconstruction pixels and NaN pixels with noise draws
            if inpaint_res == True:
                res = inpaint_with_gaussian_noise(res, mean = mean, sigma = sigma, iptd_sigma = iptd_sigma)
            
            if ( os.path.exists(''.join(( outpath, '.ol.it%03d.pkl' %(it)))) ) & resume == True:

                logging.info('\n\nFound %s --> resuming iteration' %(''.join(( outpath, '.ol.it%03d.pkl' %(it)))))
                ol = read_objects_from_pickle(''.join(( outpath, '.ol.it%03d.pkl' %(it))))

            else:

                # Anscombe transform & thresholding
                logging.info('[ %s ] Start wavelet transform and detection'%datetime.now())
                ares = np.copy(res)
                ares -= C
                aim = anscombe_transform(ares, sigma_ans, mean_ans, gain_ans)
                awdc = bspl_atrous(aim, level, header, conditions)
                
                # Labels & true wavelet coefficients
                ldc, dea = ms_detect_and_deblend(awdc, n_sigmas = n_sigmas, lvl_deblend = lvl_deblend, deblend_contrast = deblend_contrast, verbose = True)
                wdc = bspl_atrous(res, level, header, conditions)

                # Regions of significance
                rl = make_regions_full_props(wdc, ldc, verbose = True)
                if not rl:
                    break

                # Interscale trees
                logging.info('[ %s ] Start interscale trees'%datetime.now())
                itl = make_interscale_trees(rl, wdc, ldc, dea, tau = tau, \
                                                    min_span = min_span, \
                                                    max_span = max_span, \
                                                    lvl_sep_big = lvl_sep_big, \
                                                    n_cpus = n_cpus, \
                                                    size_patch = size_patch, \
                                                    verbose = True)
                if not itl:
                    break

                # Restoration of detected objects
                logging.info('[ %s ] Start object restoration'%datetime.now())
                ol = restore_objects_default(itl, res, 0.01, 15, wdc,ldc, size_patch_small = 1, \
                                                   gamma = gamma, \
                                                   extent_sep = extent_sep, \
                                                   ecc_sep = ecc_sep, \
                                                   lvl_sep_lin = lvl_sep_lin, \
                                                   lvl_sep_big = lvl_sep_big, \
                                                   lvl_sep_op = lvl_sep_op, \
                                                   rm_gamma_for_big = rm_gamma_for_big, \
                                                   size_patch_big = 1, \
                                                   size_big_objects = 512, \
                                                   deconv = deconv, \
                                                   n_cpus = n_cpus ) # objects in original image pixel value range

            # Atom
            logging.info('[ %s ] Add atoms to restored images.'%datetime.now())
            atom = np.zeros(res.shape)
            
            for o in ol:
                x_min, y_min, x_max, y_max = o.bbox
                atom[ x_min : x_max, y_min : y_max ] += o.image
                rec_lvl[ x_min : x_max, y_min : y_max, o.level ] += o.image
                dei[x_min : x_max, y_min : y_max] = dei[x_min : x_max, y_min : y_max] + o.det_err_image
            
            # Update Residuals
            res -= atom
            rec += atom

            # Convergence
            flux_res = 1 - ( np.abs(np.sum(atom) - np.sum(np.sqrt(res**2)))) / np.sum(np.sqrt(res**2))
            flux_rec = 1 - ( np.abs(np.sum(atom) - np.sum(np.sqrt(rec**2)))) / np.sum(np.sqrt(rec**2))
            logging.info('[ %s ] Convergence : res = %f rec = %f' %( datetime.now(), flux_res, flux_rec ))
            old_std = std
            std = np.std(res)
            eps = np.sqrt( ( old_std - std )**2 ) / np.sqrt( old_std**2 )
            normeps = scale_lvl_eps * eps * level / len(ol)
            window.append(normeps)
            if len(window) > 5:window.pop(0)
            avnormeps = np.mean(window)
            cparl.append([ str(level), str(it), str(avnormeps), str(normeps), str(eps), str(flux_rec), str(flux_res), str(len(ol)), str( datetime.now() - start_time_it ) ])
            logging.info('[ %s ] Number Objects = %d, Normalized Epsilon = %f, Window Normalized Epsilon = %f' %( datetime.now(), len(ol), normeps, avnormeps ))

            # Data Dump
            if ( os.path.exists(''.join(( outpath, '.ol.it%03d.pkl' %(it)))) ) & resume == True:
                it += 1
                continue
            else:
                write_interscale_trees_to_pickle( itl, ''.join(( outpath, '.itl.it%03d.pkl' %(it) )), overwrite = True)
                write_objects_to_pickle( ol, ''.join(( outpath, '.ol.it%03d.pkl' %(it) )), overwrite = True)
                if data_dump:
                    logging.info('[ %s ] Dumping data in %s' %(datetime.now(), outdir) )
                    write_regions_to_pickle( rl, ''.join(( outpath, '.rl.it%03d.pkl' %(it) )), overwrite = True)
                    
                    hdu = fits.PrimaryHDU(res)
                    hdu.writeto(''.join(( outpath, '.res.it%03d.fits' %(it))), overwrite = True)
                    
                    hdu = fits.PrimaryHDU(rec)
                    hdu.writeto(''.join(( outpath, '.rec.it%03d.fits' %(it))), overwrite = True)

                    hdu = fits.PrimaryHDU(dei)
                    hdu.writeto(''.join(( outpath, '.dei.it%03d.fits' %(it))), overwrite = True)

                if gif:
                    awdc.waveplot( name = 'Anscombe modified\nWavelet Planes\nIteration %d'%(it), show = False, save_path = ''.join(( outpath, '.awdc.it%03d.png' %(it))), origin = 'lower')
                    wdc.waveplot( name = 'Wavelet Planes\nIteration %d'%(it), show = False, save_path = ''.join(( outpath, '.wdc.it%03d.png' %(it))), origin = 'lower')
                    ldc.waveplot( name = 'Multiscale label\nIteration %d'%(it), show = False, save_path = ''.join(( outpath, '.ldc.it%03d.png' %(it))), origin = 'lower')
                    #sdc.histogram_noise( name = 'Noise histogram (lvl = 0)\nIteration %d'%(it), show = False, save_path = ''.join(( outpath, '.hist.it%03d.png' %(it))))
                    plot_frame( level = level, it = it, nobj = len(ol), \
                                                        original_image = im, \
                                                        restored_image = rec, \
                                                        residuals = res, \
                                                        atom = atom, \
                                                        outpath = outpath )
            logging.info('[ %s ]End of iteration (time: %s).\n\n'%(datetime.now(), datetime.now() - start_time_it ))
            it += 1
            if it > max_iter:
                break

        if it > max_iter:
            break
    
    # If intern Ray cluster, shut it down after when run is over
    if inter_ray_flag:
        ray.shutdown()
    #===========================================================================
    # Write results to disk
    logging.info('Finished iterating.\nWriting results to disk in %s' %(outdir) )
    hdu_res = fits.PrimaryHDU( res, header = header )
    hdu_res.writeto( ''.join(( outpath, '.residuals.fits' )), overwrite = True )
    hdu_rec = fits.PrimaryHDU( rec, header = header )
    hdu_rec.writeto( ''.join(( outpath, '.restored.fits' )), overwrite = True )
    hdu_dei = fits.PrimaryHDU( dei, header = header )
    hdu_rec.writeto( ''.join(( outpath, '.deterror.fits' )), overwrite = True )
    hdu_rec_lvl = fits.PrimaryHDU( rec_lvl, header = header )
    hdu_rec_lvl.writeto( ''.join(( outpath, '.scl.restored.fits' )), overwrite = True )

    with open( ''.join(( outpath, '.cpar.txt' )), 'w+') as conv:
        conv.write('# level, it, window_normeps, normeps, eps, n_obj, time_it\n')
        for l in cparl:
            conv.write('%s\n' %(' '.join(l)))

    plot_convergence( cparl = cparl, outpath = outpath, marker = 'o', linewidth = 2 )

    if gif:
        make_gif( framerate = 3, outpath = outpath )

    logging.info( datetime.now() - StartTimeAlgo )


def load_iteration( it, outpath ):
    '''deprecated'''

    wdc = wavelet_datacube.from_fits(''.join(( outpath, '.wdc.it%03d.fits' %(it)) ))
    ldc = label_datacube.from_fits( ''.join(( outpath, '.ldc.it%03d.fits' %(it) )))
    rl = read_regions_from_pickle( ''.join(( outpath, '.rl.it%03d.pkl' %(it) )))
    itl = read_interscale_trees_from_pickle( ''.join(( outpath, '.itl.it%03d.pkl' %(it) )))
    ol = read_objects_from_pickle( ''.join(( outpath, '.ol.it%03d.pkl' %(it) )))
    return wdc, ldc, rl, itl, ol

def read_image_atoms( nfp, filter_it = None, verbose = False ):

    # Object lists
    if filter_it == None:
        opath = nfp + '*ol.it*.pkl'
        itpath = nfp + '*itl.it*.pkl'
    else:
        opath = nfp + '*ol.it' + filter_it  + '.pkl'
        itpath = nfp + '*itl.it' + filter_it + '.pkl'

    opathl = glob.glob(opath)
    opathl.sort()

    # Interscale tree lists

    itpathl = glob.glob(itpath)
    itpathl.sort()

    tol = []
    titl = []

    if verbose:
        print('Reading %s.'%(opath))
        print('Reading %s.'%(itpath))

    for i, ( op, itlp ) in enumerate( zip( opathl, itpathl )):

        if verbose :
            print('Iteration %d' %(i), end ='\r')

        ol = read_objects_from_pickle( op )
        itl = read_interscale_trees_from_pickle( itlp )

        for j, o in enumerate(ol):

            tol.append(o)
            titl.append(itl[j])

    return tol, titl

if __name__=="__main__":

    if len(sys.argv) < 4:
        raise SyntaxError("Insufficient arguments.")

    main(sys.argv[1], sys.argv[2], sys.argv[3], *sys.argv[4:])
