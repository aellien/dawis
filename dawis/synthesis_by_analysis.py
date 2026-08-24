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
from dawis.cubes import *
from dawis.detection import (
    hard_threshold,
    label_regions,
    make_regions_full_props,
    ms_detect_and_deblend,
    region,
)
from dawis.wavelet_transforms import bspl_atrous
from dawis.anscombe_transforms import *
from dawis.make_interscale_trees import *
from dawis.restore_objects import *
from dawis.restoration import sample_noise, inpaint_with_gaussian_noise
from dawis.parallel import parallel_backend
from dawis.io import (
    read_image,
    read_regions_from_pickle,
    read_ol_from_hdf5,
    read_ol_from_pickle,
    write_itl_to_hdf5,
    write_itl_to_pickle,
    write_ol_to_hdf5,
    write_ol_to_pickle,
    write_regions_to_pickle,
)

def synthesis_by_analysis(indir, infile, outdir, n_cpus = 1, starting_level = 2, tau = 0.8, n_levels = None, n_sigmas = 5, deblend_contrast = 0.1,\
                                gamma = 0.2, min_span = 2, max_span = 3, lvl_sep_big = 6, rm_gamma_for_big = False, lvl_deblend = 3, outfile_format = 'pickle', \
                                extent_sep = 0.1, ecc_sep = 0.95, lvl_sep_lin = 2, lvl_sep_op = 3, ceps = 1E-3, scale_lvl_eps = 1, conditions = 'loop', deconv = False,\
                                max_iter = 500, size_patch = 100, inpaint_res = True, data_dump = True, iptd_sigma = 3, resume = True):

    # Check infile extension
    if infile[-5:] != '.fits':
        raise DawisWrongType('Incorrect file extension -->', infile[:-5], '.fits')

    # What a worker is, for the length of this run. One worker selects the
    # serial backend, more than one the default parallel backend; the adapter
    # attaches to a cluster that is already up and starts one only if none is,
    # and stops only what it started. `dawis.parallel` owns all of that now --
    # this file names no backend and imports nothing optional.
    #
    # `try`/`finally` rather than a `with` block only because the run below is
    # the whole of this function and the indentation would swallow it twice
    # over; #49 carves the pipeline out and the call site becomes a `with`.
    # Either way the cluster is released on the way out, including out through
    # an exception, which the pre-#47 code never managed: its shutdown sat on
    # a misspelt flag and was unreachable.
    parallelism = parallel_backend(n_workers = n_cpus)
    parallelism.start()
    try:
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
        im, header = read_image(os.path.join(indir, infile))
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
                logging.info('[ %s ] Level = %d Iteration = %d ---------------' %(datetime.now(), level, it))

                # inpaint bad reconstruction pixels and NaN pixels with noise draws
                if inpaint_res == True:
                    res = inpaint_with_gaussian_noise(res, mean = mean, sigma = sigma, iptd_sigma = iptd_sigma)
            
                # Check if resume run
                if ( os.path.exists(''.join(( outpath, '.ol.it%03d.pkl' %(it)))) ) & resume == True:

                    logging.info('Found %s.\nResuming iteration' %(''.join(( outpath, '.ol.it%03d.pkl' %(it)))))
                    ol = read_ol_from_pickle(''.join(( outpath, '.ol.it%03d.pkl' %(it))))
                
                elif ( os.path.exists(''.join(( outpath, '.ol.it%03d.hdf5' %(it)))) ) & resume == True:

                    logging.info('Found %s --> resuming iteration' %(''.join(( outpath, '.ol.it%03d.hdf5' %(it)))))
                    ol = read_ol_from_hdf5(''.join(( outpath, '.ol.it%03d.hdf5' %(it))))

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
                logging.info('Convergence : res = %f rec = %f' %(flux_res, flux_rec ))
                old_std = std
                std = np.std(res)
                eps = np.sqrt( ( old_std - std )**2 ) / np.sqrt( old_std**2 )
                normeps = scale_lvl_eps * eps * level / len(ol)
                window.append(normeps)
                if len(window) > 5:window.pop(0)
                avnormeps = np.mean(window)
                cparl.append([ str(level), str(it), str(avnormeps), str(normeps), str(eps), str(flux_rec), str(flux_res), str(len(ol)), str( datetime.now() - start_time_it ) ])
                logging.info('Number Objects = %d, Normalized Epsilon = %f, Window Normalized Epsilon = %f' %( len(ol), normeps, avnormeps ))

                # Data Dump
                if ( os.path.exists(''.join(( outpath, '.ol.it%03d.hdf5' %(it)))) ) & resume == True:
                    it += 1
                    logging.info('[ %s ]End of iteration (time: %s) \n\n'%(datetime.now(), datetime.now() - start_time_it ))
                    continue
            
                elif ( os.path.exists(''.join(( outpath, '.ol.it%03d.pkl' %(it)))) ) & resume == True:
                    logging.info('[ %s ]End of iteration (time: %s) \n\n'%(datetime.now(), datetime.now() - start_time_it ))
                    it += 1
                    continue
            
                else:
                    if outfile_format == 'hdf5':
                        write_itl_to_hdf5( itl, ''.join(( outpath, '.itl.it%03d.hdf5' %(it) )))
                        write_ol_to_hdf5( ol, ''.join(( outpath, '.ol.it%03d.hdf5' %(it) )))
                    elif outfile_format == 'pickle':
                        write_itl_to_pickle( itl, ''.join(( outpath, '.itl.it%03d.pkl' %(it) )))
                        write_ol_to_pickle( ol, ''.join(( outpath, '.ol.it%03d.pkl' %(it) )))
                    if data_dump:
                        logging.info('[ %s ] Dumping data in %s' %(datetime.now(), outdir) )
                        write_regions_to_pickle( rl, ''.join(( outpath, '.rl.it%03d.pkl' %(it) )), overwrite = True)
                    
                        hdu = fits.PrimaryHDU(res)
                        hdu.writeto(''.join(( outpath, '.res.it%03d.fits' %(it))), overwrite = True)
                    
                        hdu = fits.PrimaryHDU(rec)
                        hdu.writeto(''.join(( outpath, '.rec.it%03d.fits' %(it))), overwrite = True)

                        hdu = fits.PrimaryHDU(dei)
                        hdu.writeto(''.join(( outpath, '.dei.it%03d.fits' %(it))), overwrite = True)

                logging.info('[ %s ]End of iteration (time: %s)\n\n'%(datetime.now(), datetime.now() - start_time_it ))
                it += 1
                if it > max_iter:
                    break

            if it > max_iter:
                break
    
    finally:
        # Release the cluster, if this run is the one that stood it up.
        parallelism.stop()
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

    # The convergence plot and the animation are drawn from these files by
    # `dawis.viz`, after the run rather than during it (#46).
    logging.info('Draw this run with:\n'
                 '    from dawis.viz import rebuild_animation, rebuild_convergence_plot\n'
                 "    rebuild_convergence_plot('%s')\n"
                 "    rebuild_animation('%s')" %(outpath, outpath))

    logging.info( datetime.now() - StartTimeAlgo )


def load_iteration( it, outpath ):
    '''deprecated'''

    wdc = wavelet_datacube.from_fits(''.join(( outpath, '.wdc.it%03d.fits' %(it)) ))
    ldc = label_datacube.from_fits( ''.join(( outpath, '.ldc.it%03d.fits' %(it) )))
    rl = read_regions_from_pickle( ''.join(( outpath, '.rl.it%03d.pkl' %(it) )))
    itl = read_interscale_trees_from_pickle( ''.join(( outpath, '.itl.it%03d.pkl' %(it) )))
    ol = read_ol_from_pickle( ''.join(( outpath, '.ol.it%03d.pkl' %(it) )))
    return wdc, ldc, rl, itl, ol

if __name__=="__main__":

    if len(sys.argv) < 4:
        raise SyntaxError("Insufficient arguments.")

    main(sys.argv[1], sys.argv[2], sys.argv[3], *sys.argv[4:])
