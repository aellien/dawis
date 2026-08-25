"""Synthesis by analysis: the greedy loop, and nothing else.

The flagship pipeline of DAWIS, and the composition the other blocks exist to
be composed by. It analyses an image in wavelet space, detects the structure
that stands above the noise, restores that structure as atoms, subtracts them
from a running residual, and does it again on what is left — coarsening the
scale it works at each time a scale stops paying.

What is here is the *order* of those steps, the conditions under which the loop
stops, and the files a run leaves behind. What is not here is any of the
mathematics: every number this file produces is produced by a call into a
block. That is the shape #49 asked for, and it is checkable: this package
imports no numerical library, which ``tests/unit/test_pipelines_layout.py``
asserts. The only arithmetic left is on the loop counter and on the log lines.

The dependency rule the layout rests on runs one way. This package may import
any block; no block may import this package. A block that needed the pipeline
would be a block that cannot be used on its own, which is the opposite of what
a toolbox is for.

One iteration, in the order it runs::

    inpaint the residual        dawis.restoration
    stabilise its variance      dawis.wavelets
    decompose it                dawis.wavelets
    detect and deblend          dawis.detection
    measure the regions         dawis.detection
    group them into trees       dawis.trees
    restore the trees as atoms  dawis.restoration
    subtract, and measure       dawis.restoration, dawis.convergence

Two things a run does are deliberately *not* done here. It does not draw: the
convergence plot and the animation are rebuilt afterwards by :mod:`dawis.viz`
from the files below, which took the frame writing off the loop (#46). And it
does not choose a parallel backend: it says how many workers a worker count
means and :mod:`dawis.parallel` decides the rest (#47).

Files a run writes, all hanging off ``<outdir>/<infile without .fits>``:

=========================== ================================================
``.log.txt``                the run log
``.ol.itNNN.pkl``/``.hdf5`` the atoms of one iteration
``.itl.itNNN.pkl``/``.hdf5`` the interscale trees they were restored from
``.rl.itNNN.pkl``           the regions detected, under ``data_dump``
``.res``/``.rec``/``.dei.itNNN.fits`` the running images, under ``data_dump``
``.residuals.fits``         what the loop could not explain
``.restored.fits``          the sum of every atom
``.deterror.fits``          the accumulated detection error
``.scl.restored.fits``      the same sum, split by wavelet scale
``.cpar.txt``               the convergence trace
=========================== ================================================

The atom dumps are also how a run resumes: an iteration whose atoms are
already on disk is read back rather than recomputed. That is resume by file
existence, which is what DAWIS has always done and what Phase B replaces with
a real object store.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

from astropy.io import fits

from dawis.convergence import CONVERGENCE_HEADER, ConvergenceMonitor
from dawis.detection import make_regions_full_props, ms_detect_and_deblend
from dawis.exceptions import DawisWrongType
from dawis.io import (
    read_image,
    read_ol_from_hdf5,
    read_ol_from_pickle,
    write_itl_to_hdf5,
    write_itl_to_pickle,
    write_ol_to_hdf5,
    write_ol_to_pickle,
    write_regions_to_pickle,
)
from dawis.parallel import parallel_backend
from dawis.restoration import (
    Reconstruction,
    restore_objects_default,
    subtract_background,
)
from dawis.trees import make_interscale_trees
from dawis.wavelets import AnscombeStabilisation, bspl_atrous, max_levels

__all__ = ['synthesis_by_analysis']


def _dump_path(outpath: str, kind: str, it: int, extension: str) -> str:
    """Path of one iteration's dump, in the one place the naming is decided."""

    return '%s.%s.it%03d.%s' % (outpath, kind, it, extension)


def _write_fits(path: str, image, header = None) -> None:
    """Write one image, overwriting whatever a previous run left there.

    The pipeline still writes its own images rather than going through
    :mod:`dawis.io`, which is read-only by decision: the writing side moves
    when the object store is rebuilt in Phase B.
    """

    fits.PrimaryHDU(image, header = header).writeto(path, overwrite = True)


def _start_logging(outpath: str) -> None:
    """Send this run's log to ``<outpath>.log.txt`` and to stdout."""

    file_handler = logging.FileHandler( filename = ''.join(( outpath, '.log.txt')), mode = 'w' )
    stdout_handler = logging.StreamHandler( sys.stdout )
    logging.basicConfig( level = logging.DEBUG, handlers = [ file_handler, stdout_handler ])


def synthesis_by_analysis(indir, infile, outdir, n_cpus = 1, starting_level = 2, tau = 0.8, n_levels = None, n_sigmas = 5, deblend_contrast = 0.1,\
                                gamma = 0.2, min_span = 2, max_span = 3, lvl_sep_big = 6, rm_gamma_for_big = False, lvl_deblend = 3, outfile_format = 'pickle', \
                                extent_sep = 0.1, ecc_sep = 0.95, lvl_sep_lin = 2, lvl_sep_op = 3, ceps = 1E-3, scale_lvl_eps = 1, conditions = 'loop', deconv = False,\
                                max_iter = 500, size_patch = 100, inpaint_res = True, data_dump = True, iptd_sigma = 3, resume = True):
    """Run the DAWIS synthesis-by-analysis pipeline on a FITS image.

    Parameters
    ----------
    indir, infile, outdir : str
        Directory the image is read from, its filename — which must end in
        ``.fits`` — and the directory every output is written to.
    n_cpus : int, optional
        Workers to spread the tree building and the restoration over. One
        runs in this process; more asks :mod:`dawis.parallel` for a backend.
        Default 1.
    starting_level : int, optional
        Coarsest wavelet scale the loop starts at. Default 2.
    n_levels : int or None, optional
        Wavelet planes to decompose onto. Default None takes the deepest the
        image supports, from :func:`~dawis.wavelets.max_levels`.
    tau, min_span, max_span, lvl_sep_big, size_patch : optional
        Handed to :func:`~dawis.trees.make_interscale_trees`, which decides
        which regions belong to the same structure.
    n_sigmas, deblend_contrast, lvl_deblend : optional
        Handed to :func:`~dawis.detection.ms_detect_and_deblend`, which decides
        what counts as a detection and where blended sources split.
    gamma, rm_gamma_for_big, extent_sep, ecc_sep, lvl_sep_lin, lvl_sep_op, deconv : optional
        Handed to :func:`~dawis.restoration.restore_objects_default`, which
        decides how each tree is synthesised back into an image.
    conditions : {'loop', 'prolongation', 'mirror'}, optional
        Boundary condition of the wavelet transform. Default ``'loop'``.
    ceps : float, optional
        Convergence threshold. A scale stops when its window-averaged
        normalised epsilon falls below this. Default 1e-3.
    scale_lvl_eps : float, optional
        Factor on that epsilon, for raising or lowering the threshold without
        touching ``ceps``. Default 1.
    max_iter : int, optional
        Hard cap on iterations, counted across every scale. Default 500.
    inpaint_res : bool, optional
        Refill the holes subtraction digs in the residual before each
        iteration. Default True.
    iptd_sigma : float, optional
        Depth, in sigmas of the noise, below which a residual pixel is
        inpainted. Default 3.
    outfile_format : {'pickle', 'hdf5'}, optional
        Format the per-iteration atoms and trees are written in. Default
        ``'pickle'``.
    data_dump : bool, optional
        Also write the regions and the running images of every iteration.
        Default True.
    resume : bool, optional
        Read an iteration back from disk when its atoms are already there,
        instead of recomputing it. Default True.

    Raises
    ------
    DawisWrongType
        If ``infile`` does not end in ``.fits``.

    Notes
    -----
    Nothing is returned. A run is its output directory — see the file table in
    the module docstring — and :func:`dawis.viz.rebuild_convergence_plot` and
    :func:`dawis.viz.rebuild_animation` draw it from there.
    """

    # Check infile extension
    if infile[-5:] != '.fits':
        raise DawisWrongType('Incorrect file extension -->', infile[:-5], '.fits')

    # What a worker is, for the length of this run. One worker selects the
    # serial backend, more than one the default parallel backend; the adapter
    # attaches to a cluster that is already up and starts one only if none is,
    # and stops only what it started. `dawis.parallel` owns all of that -- this
    # file names no backend and imports nothing optional. The cluster is
    # released on the way out, including out through an exception, which the
    # pre-#47 code never managed: its shutdown sat on a misspelt flag and was
    # unreachable.
    with parallel_backend(n_workers = n_cpus):

        # Logs
        outpath = os.path.join( outdir, infile[:-5] )
        _start_logging(outpath)
        logging.info(datetime.now())
        start_time_algo = datetime.now()
        logging.info('DAWIS\nInfile = %s\nInput Directory = %s\nOutput Directory = %s\nN_cpus = %d\nTau = %f\nGamma = %f\nmin_span = %d\nmax_span = %d\nConvergence Epsilon = %f'\
                                        %(infile, indir, outdir, n_cpus, tau, gamma, min_span, max_span, ceps))

        #  FITS header & image properties
        im, header = read_image(os.path.join(indir, infile))
        if not n_levels:
            n_levels = max_levels(im.shape)
        logging.info('Image of size %d x %d\n--> n_levels = %d, corresponding to maximum size %d pixels\n' \
                                    %(im.shape[0], im.shape[1], n_levels, 2**n_levels) )

        # Noise properties. The background goes out of the image here, so that
        # zero is the centre of the noise everywhere below.
        im, mean, sigma = subtract_background(im, n_sigmas = 3, bins = 200)
        logging.info('Noise properties for inpainting: sigma = %1.3e, mean = %1.3e\n' %(sigma, mean))

        # Anscombe transform parameters, measured once against the input image
        # and applied to every residual the loop produces.
        stabilisation = AnscombeStabilisation.fit(im, max_err = 1E-4, n_sigmas = 3)
        logging.info('Noise parametric values for Anscombe transform: sigma = %1.3e, mean = %1.3e, gain = %1.3e\n' \
                                    %(stabilisation.sigma, stabilisation.mean, stabilisation.gain))

        #===========================================================================

        run = Reconstruction(im, n_levels)
        monitor = ConvergenceMonitor(scale_lvl_eps = scale_lvl_eps)
        it = 1

        for level in range( starting_level, n_levels ):

            monitor.begin_level()
            while monitor.avnormeps > ceps :

                start_time_it = datetime.now()
                logging.info('[ %s ] Level = %d Iteration = %d ---------------' %(datetime.now(), level, it))

                # inpaint bad reconstruction pixels and NaN pixels with noise draws
                if inpaint_res:
                    run.inpaint(mean = mean, sigma = sigma, iptd_sigma = iptd_sigma)

                # Check if resume run
                pickle_dump = _dump_path(outpath, 'ol', it, 'pkl')
                hdf5_dump = _dump_path(outpath, 'ol', it, 'hdf5')
                resumed = True

                if resume and os.path.exists(pickle_dump):

                    logging.info('Found %s.\nResuming iteration' %(pickle_dump))
                    ol = read_ol_from_pickle(pickle_dump)

                elif resume and os.path.exists(hdf5_dump):

                    logging.info('Found %s --> resuming iteration' %(hdf5_dump))
                    ol = read_ol_from_hdf5(hdf5_dump)

                else:

                    resumed = False

                    # Anscombe transform & thresholding
                    logging.info('[ %s ] Start wavelet transform and detection'%datetime.now())
                    aim = stabilisation.transform(run.residuals)
                    awdc = bspl_atrous(aim, level, header, conditions)

                    # Labels & true wavelet coefficients
                    ldc, dea = ms_detect_and_deblend(awdc, n_sigmas = n_sigmas, lvl_deblend = lvl_deblend, deblend_contrast = deblend_contrast, verbose = True)
                    wdc = bspl_atrous(run.residuals, level, header, conditions)

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
                    ol = restore_objects_default(itl, run.residuals, 0.01, 15, wdc, ldc, size_patch_small = 1, \
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
                atom = run.add(ol)

                # Convergence
                step = monitor.update(level, it, atom, run.residuals, run.restored,
                                      len(ol), datetime.now() - start_time_it)
                logging.info('Convergence : res = %f rec = %f' %(step.flux_res, step.flux_rec ))
                logging.info('Number Objects = %d, Normalized Epsilon = %f, Window Normalized Epsilon = %f' \
                                            %( step.n_atoms, step.normeps, step.avnormeps ))

                # Data Dump. An iteration that was read back off disk is
                # already dumped, by the run that computed it.
                if not resumed:

                    if outfile_format == 'hdf5':
                        write_itl_to_hdf5( itl, _dump_path(outpath, 'itl', it, 'hdf5') )
                        write_ol_to_hdf5( ol, hdf5_dump )
                    elif outfile_format == 'pickle':
                        write_itl_to_pickle( itl, _dump_path(outpath, 'itl', it, 'pkl') )
                        write_ol_to_pickle( ol, pickle_dump )

                    if data_dump:
                        logging.info('[ %s ] Dumping data in %s' %(datetime.now(), outdir) )
                        write_regions_to_pickle( rl, _dump_path(outpath, 'rl', it, 'pkl'), overwrite = True)
                        _write_fits( _dump_path(outpath, 'res', it, 'fits'), run.residuals )
                        _write_fits( _dump_path(outpath, 'rec', it, 'fits'), run.restored )
                        _write_fits( _dump_path(outpath, 'dei', it, 'fits'), run.det_err )

                logging.info('[ %s ]End of iteration (time: %s)\n\n'%(datetime.now(), datetime.now() - start_time_it ))
                it += 1
                # The cap applies to a resumed iteration too. It did not
                # before: the resume branch had its own `continue`, which
                # jumped over this check, so `max_iter` was not a cap on a
                # resumed run at all.
                if it > max_iter:
                    break

            if it > max_iter:
                break

    #===========================================================================
    # Write results to disk
    logging.info('Finished iterating.\nWriting results to disk in %s' %(outdir) )
    _write_fits( ''.join(( outpath, '.residuals.fits' )), run.residuals, header )
    _write_fits( ''.join(( outpath, '.restored.fits' )), run.restored, header )
    # The detection error, not a second copy of the restored image: the old
    # code built the right HDU and then wrote the wrong one (#49).
    _write_fits( ''.join(( outpath, '.deterror.fits' )), run.det_err, header )
    _write_fits( ''.join(( outpath, '.scl.restored.fits' )), run.restored_per_level, header )

    with open( ''.join(( outpath, '.cpar.txt' )), 'w+') as conv:
        conv.write('%s\n' %(CONVERGENCE_HEADER))
        for row in monitor.rows:
            conv.write('%s\n' %(' '.join(row)))

    # The convergence plot and the animation are drawn from these files by
    # `dawis.viz`, after the run rather than during it (#46).
    logging.info('Draw this run with:\n'
                 '    from dawis.viz import rebuild_animation, rebuild_convergence_plot\n'
                 "    rebuild_convergence_plot('%s')\n"
                 "    rebuild_animation('%s')" %(outpath, outpath))

    logging.info( datetime.now() - start_time_algo )
