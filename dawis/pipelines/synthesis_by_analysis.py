"""Synthesis by analysis: the greedy loop, and nothing else.

The flagship pipeline of DAWIS, and the composition the other blocks exist to
be composed by. It analyses an image in wavelet space, detects the structure
that stands above the noise, restores that structure as atoms, subtracts them
from a running residual, and does it again on what is left — coarsening the
scale it works at each time a scale stops paying.

:func:`run` is how it is called, and it takes **an array and a**
:class:`~dawis.config.Config` — never a path (#54). That is what makes the
whole toolbox testable: a block below the pipeline never opens a file, so the
regression harness can drive a run on a synthetic array and a caller can work
entirely in memory. FITS is a format the image may happen to arrive in, and
:func:`run_fits` is the one place on the run path that opens one, reading the
file and delegating immediately.

A run answers with a :class:`~dawis.results.RunResult`. It still writes its
outputs — Phase A does not change what lands on disk — but nothing has to be
read back to see what happened.

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

Before the first of those, the input is measured and anything concerning about
it is warned about — zeros, NaNs, a saturated plateau, an integer dtype. It is
recorded on the result and it never stops the run (#4); see
:mod:`dawis.results`.

Two things a run does are deliberately *not* done here. It does not draw: the
convergence plot and the animation are rebuilt afterwards by :mod:`dawis.viz`
from the files below, which took the frame writing off the loop (#46). And it
does not choose a parallel backend: it says how many workers a worker count
means and :mod:`dawis.parallel` decides the rest (#47).

One thing a run does that no block may: it installs logging handlers, for the
length of the run and no longer. That is what gives the run directory its
``.log.txt``, and what makes a default run visible in a notebook without any
setup — and it is confined to this one entry point precisely so that
importing a block configures nothing. See :mod:`dawis.logs` (#55).

Files a run writes, all hanging off ``<outdir>/<name>``:

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

``name`` is what that stem is called. :func:`run_fits` takes it from the input
filename, so a run driven from ``ngc1275.fits`` writes ``ngc1275.restored.fits``
exactly as it always did.

The atom dumps are also how a run resumes: an iteration whose atoms are
already on disk is read back rather than recomputed. That is resume by file
existence, which is what DAWIS has always done and what Phase B replaces with
a real object store.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from datetime import datetime

from astropy.io import fits

from dawis.config import Config, make_rng, resolve_seed
from dawis.convergence import CONVERGENCE_HEADER, ConvergenceMonitor
from dawis.detection import make_regions_full_props, ms_detect_and_deblend
from dawis.exceptions import ConfigError, DawisDimensionError
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
from dawis.logs import run_logging
from dawis.parallel import parallel_backend
from dawis.restoration import (
    Reconstruction,
    restore_objects_default,
    subtract_background,
)
from dawis.results import RunResult, inspect_image
from dawis.trees import make_interscale_trees
from dawis.wavelets import AnscombeStabilisation, bspl_atrous, max_levels

__all__ = ['DEFAULT_RUN_NAME', 'FITS_EXTENSION', 'run', 'run_fits']

log = logging.getLogger(__name__)

#: Stem every output of an array-in run hangs off, when the caller names none.
#:
#: :func:`run_fits` has a better answer — the input filename — but an array
#: carries no name, and a run has to call its files something.
DEFAULT_RUN_NAME = 'dawis'

#: Extension :func:`run_fits` accepts, and the length of it that is stripped to
#: make the run's name.
FITS_EXTENSION = '.fits'


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


def run_fits(path: str, config: Config | None = None, *,
             outdir: str = '.', name: str | None = None) -> RunResult:
    """Run the pipeline on a FITS image. The thin wrapper, and the only one.

    DAWIS works on arrays; FITS is how an image usually reaches it. This is
    the only place in the package that opens one — it reads the file, keeps
    the header so the outputs carry the astrometry of the input, and hands the
    array to :func:`run`. Nothing below this function ever sees a path (#54).

    Parameters
    ----------
    path : str
        The FITS file to run on. Its primary HDU is the image, and its name
        must end in ``.fits``.
    config : Config, optional
        The knobs of the run. Default builds a :class:`~dawis.config.Config`
        of published defaults.
    outdir : str, optional
        Directory every output is written into. Default the working directory.
    name : str, optional
        Stem the outputs are named after. Default is the input filename
        without its ``.fits``, which is what DAWIS has always written.

    Returns
    -------
    RunResult
        What the run produced — see :func:`run`.

    Raises
    ------
    ConfigError
        If ``path`` does not end in ``.fits``. Checked before the file is
        opened, so a mistyped extension is a configuration error and not a
        confusing read failure halfway down astropy.

    Examples
    --------
    >>> from dawis import Config, run_fits            # doctest: +SKIP
    >>> result = run_fits('ngc1275.fits', Config(n_cpus = 4),
    ...                   outdir = 'out')             # doctest: +SKIP
    """

    if not path.endswith(FITS_EXTENSION):
        raise ConfigError('Incorrect file extension --> %s does not end in %s'
                          % (path, FITS_EXTENSION))

    image, header = read_image(path)

    if name is None:
        name = os.path.basename(path)[:-len(FITS_EXTENSION)]

    return run(image, config, outdir = outdir, name = name, header = header)


def run(image, config: Config | None = None, *, outdir: str = '.',
        name: str = DEFAULT_RUN_NAME,
        header: fits.Header | None = None) -> RunResult:
    """Run the DAWIS synthesis-by-analysis pipeline on an image array.

    The front door of the package. It takes what a run *works on* — an array —
    and what a run *does* — a :class:`~dawis.config.Config` — and keeps them
    apart from where it reads and writes, because the same configuration is
    meant to be run over many images.

    Every knob lives on the configuration and is checked when that is built,
    which is why this signature is three arguments and not thirty. The one
    rule that cannot be checked there is checked here: ``n_levels`` defaults
    to ``None``, meaning the image decides the depth, so the configuration is
    rebuilt the moment there is an image to ask — still before the first
    wavelet transform.

    Parameters
    ----------
    image : numpy.ndarray
        The 2D image to run on. Not modified: the sky background comes out of
        a copy.
    config : Config, optional
        The knobs of the run. Default builds a :class:`~dawis.config.Config`
        of published defaults.
    outdir : str, optional
        Directory every output is written into. Default the working directory.
    name : str, optional
        Stem the outputs are named after — see the file table in the module
        docstring. Default ``'dawis'``; :func:`run_fits` passes the input
        filename instead.
    header : astropy.io.fits.Header, optional
        Header copied onto the images the run writes, so they carry the
        astrometry of the image they came from. Default None, which writes
        them with no astrometry.

    Returns
    -------
    RunResult
        The images the run produced, the configuration it used with
        ``n_levels`` resolved, the seed it drew, and the diagnostics measured
        on ``image``. The same information is on disk under ``outdir``.

    Raises
    ------
    DawisDimensionError
        If ``image`` is not 2D.
    ConfigError
        If a configuration rule that needed the image's depth is broken — see
        :class:`dawis.config.Config`.

    Notes
    -----
    Anything concerning about ``image`` is warned about and never fatal (#4):
    the numbers land on :attr:`~dawis.results.RunResult.diagnostics` and the
    concerning ones go to the log. See :mod:`dawis.results`.

    :func:`dawis.viz.rebuild_convergence_plot` and
    :func:`dawis.viz.rebuild_animation` draw a finished run from
    :attr:`~dawis.results.RunResult.outpath`.

    Examples
    --------
    >>> from dawis import Config, run                 # doctest: +SKIP
    >>> result = run(image, Config(seed = 1), outdir = 'out')  # doctest: +SKIP
    >>> result.n_atoms                                # doctest: +SKIP
    463
    """

    if config is None:
        config = Config()

    # The run's one generator, made once here and threaded to every consumer
    # that draws: the Anscombe fit, the Anscombe transform of each residual,
    # and the inpainting. They used to be independent unseeded calls into the
    # global `numpy.random` state, which is why two runs of the same input
    # took different trajectories (#53). A fourth consumer, the bootstrap
    # noise cube inside the interscale trees, was deleted with the rest of the
    # unreachable synthesis block in #38 and has no site left to thread.
    run_seed = resolve_seed(config.seed)
    rng = make_rng(run_seed)

    # Where the run's output lands, and what its files are called. Named
    # before anything is set up, because the log file hangs off it too.
    outpath = os.path.join( outdir, name )

    # The run's own voice, for the length of the run and no longer. This is
    # the one place in DAWIS that installs a handler: it gives the run
    # directory its log file, and adds a stdout handler only when the caller
    # configured no logging at all, so a default run is not twenty silent
    # minutes and a configured one is not doubled. See :mod:`dawis.logs`.
    with run_logging(outpath):

        # What a worker is, for the length of this run. One worker selects the
        # serial backend, more than one the default parallel backend; the adapter
        # attaches to a cluster that is already up and starts one only if none is,
        # and stops only what it started. `dawis.parallel` owns all of that -- this
        # file names no backend and imports nothing optional. The cluster is
        # released on the way out, including out through an exception, which the
        # pre-#47 code never managed: its shutdown sat on a misspelt flag and was
        # unreachable.
        with parallel_backend(n_workers = config.n_cpus):

            log.info(datetime.now())
            start_time_algo = datetime.now()
            log.info('DAWIS\nRun = %s\nOutput Directory = %s\nN_cpus = %d\nTau = %f\nGamma = %f\nmin_span = %d\nmax_span = %d\nConvergence Epsilon = %f'\
                                            %(name, outdir, config.n_cpus, config.tau, config.gamma, config.min_span, config.max_span, config.ceps))
            # The seed goes in the log whether it was asked for or not. An
            # unseeded run is the science default, and this line is the whole
            # reason that default costs nothing: it is what makes the run
            # repeatable afterwards.
            #
            # WARNING, alone among the banner lines, and not because anything
            # is wrong. A caller who configured logging keeps their own level
            # (#55), and a level of WARNING -- what `logging.basicConfig()`
            # gives by default -- would otherwise leave the run's `.log.txt`
            # with no seed in it. That is the one line whose loss cannot be
            # repaired afterwards: the run is then unrepeatable, and nothing
            # in the outputs says what it was.
            log.warning('Random seed = %d%s\n'
                        %(run_seed, '' if config.seed is not None else ' (generated)'))

            # What the run was handed. Warn, never block (#4): the astronomer
            # usually knows their image has a blank border, and a pipeline that
            # refuses one is a pipeline nobody can use.
            diagnostics = inspect_image(image)
            log.info('Input image: %s\n' %(diagnostics.summary()))
            for line in diagnostics.warnings:
                log.warning('/!\\ Input /!\\ %s', line)

            # Not one of those warnings, and deliberately fatal: a cube is not an
            # image the run could do anything with, so there is nothing to warn
            # about and proceed to. Every diagnostic above is still logged first,
            # because they say what was handed over.
            if diagnostics.ndim != 2:
                raise DawisDimensionError(
                    'DAWIS runs on one image at a time -->', diagnostics.ndim, 2)

            if not config.n_levels:
                # Re-checked with the number the image supplied, so the rules that
                # compare a scale index against the depth of the decomposition are
                # not skipped just because the depth was left to the image.
                config = replace(config, n_levels = max_levels(diagnostics.shape))
            n_levels = config.n_levels
            log.info('Image of size %d x %d\n--> n_levels = %d, corresponding to maximum size %d pixels\n' \
                                        %(diagnostics.shape[0], diagnostics.shape[1], n_levels, 2**n_levels) )

            # Noise properties. The background goes out of the image here, so that
            # zero is the centre of the noise everywhere below. Out of a copy: the
            # array belongs to the caller.
            im, mean, sigma = subtract_background(image, n_sigmas = 3, bins = 200)
            log.info('Noise properties for inpainting: sigma = %1.3e, mean = %1.3e\n' %(sigma, mean))

            # Anscombe transform parameters, measured once against the input image
            # and applied to every residual the loop produces. The fit draws too --
            # every bisection step stabilises the image and fills its NaNs -- so it
            # takes the run's generator like everything else.
            stabilisation = AnscombeStabilisation.fit(im, max_err = 1E-4, n_sigmas = 3,
                                                      rng = rng)
            log.info('Noise parametric values for Anscombe transform: sigma = %1.3e, mean = %1.3e, gain = %1.3e\n' \
                                        %(stabilisation.sigma, stabilisation.mean, stabilisation.gain))

            #===========================================================================

            reconstruction = Reconstruction(im, n_levels)
            monitor = ConvergenceMonitor(scale_lvl_eps = config.scale_lvl_eps)
            it = 1
            n_atoms = 0

            for level in range( config.starting_level, n_levels ):

                monitor.begin_level()
                while monitor.avnormeps > config.ceps :

                    start_time_it = datetime.now()
                    log.info('[ %s ] Level = %d Iteration = %d ---------------' %(datetime.now(), level, it))

                    # inpaint bad reconstruction pixels and NaN pixels with noise draws
                    if config.inpaint_res:
                        reconstruction.inpaint(mean = mean, sigma = sigma,
                                               iptd_sigma = config.iptd_sigma, rng = rng)

                    # Check if resume run
                    pickle_dump = _dump_path(outpath, 'ol', it, 'pkl')
                    hdf5_dump = _dump_path(outpath, 'ol', it, 'hdf5')
                    resumed = True

                    if config.resume and os.path.exists(pickle_dump):

                        log.info('Found %s.\nResuming iteration' %(pickle_dump))
                        ol = read_ol_from_pickle(pickle_dump)

                    elif config.resume and os.path.exists(hdf5_dump):

                        log.info('Found %s --> resuming iteration' %(hdf5_dump))
                        ol = read_ol_from_hdf5(hdf5_dump)

                    else:

                        resumed = False

                        # Anscombe transform & thresholding
                        log.info('[ %s ] Start wavelet transform and detection'%datetime.now())
                        aim = stabilisation.transform(reconstruction.residuals, rng = rng)
                        awdc = bspl_atrous(aim, level, header, config.conditions)

                        # Labels & true wavelet coefficients
                        ldc, dea = ms_detect_and_deblend(awdc, n_sigmas = config.n_sigmas, lvl_deblend = config.lvl_deblend, deblend_contrast = config.deblend_contrast, verbose = True)
                        wdc = bspl_atrous(reconstruction.residuals, level, header, config.conditions)

                        # Regions of significance
                        rl = make_regions_full_props(wdc, ldc, verbose = True)
                        if not rl:
                            break

                        # Interscale trees
                        log.info('[ %s ] Start interscale trees'%datetime.now())
                        itl = make_interscale_trees(rl, wdc, ldc, dea, tau = config.tau, \
                                                            min_span = config.min_span, \
                                                            max_span = config.max_span, \
                                                            lvl_sep_big = config.lvl_sep_big, \
                                                            n_cpus = config.n_cpus, \
                                                            size_patch = config.size_patch, \
                                                            verbose = True)
                        if not itl:
                            break

                        # Restoration of detected objects
                        log.info('[ %s ] Start object restoration'%datetime.now())
                        ol = restore_objects_default(itl, reconstruction.residuals, 0.01, 15, wdc, ldc, size_patch_small = 1, \
                                                           gamma = config.gamma, \
                                                           extent_sep = config.extent_sep, \
                                                           ecc_sep = config.ecc_sep, \
                                                           lvl_sep_lin = config.lvl_sep_lin, \
                                                           lvl_sep_big = config.lvl_sep_big, \
                                                           lvl_sep_op = config.lvl_sep_op, \
                                                           rm_gamma_for_big = config.rm_gamma_for_big, \
                                                           size_patch_big = 1, \
                                                           size_big_objects = 512, \
                                                           deconv = config.deconv, \
                                                           cg_max_iter = config.cg_max_iter, \
                                                           n_cpus = config.n_cpus ) # objects in original image pixel value range

                    # Atom
                    log.info('[ %s ] Add atoms to restored images.'%datetime.now())
                    atom = reconstruction.add(ol)
                    n_atoms += len(ol)

                    # Convergence
                    step = monitor.update(level, it, atom, reconstruction.residuals,
                                          reconstruction.restored, len(ol),
                                          datetime.now() - start_time_it)
                    log.info('Convergence : res = %f rec = %f' %(step.flux_res, step.flux_rec ))
                    log.info('Number Objects = %d, Normalized Epsilon = %f, Window Normalized Epsilon = %f' \
                                                %( step.n_atoms, step.normeps, step.avnormeps ))

                    # Data Dump. An iteration that was read back off disk is
                    # already dumped, by the run that computed it.
                    if not resumed:

                        if config.outfile_format == 'hdf5':
                            write_itl_to_hdf5( itl, _dump_path(outpath, 'itl', it, 'hdf5') )
                            write_ol_to_hdf5( ol, hdf5_dump )
                        elif config.outfile_format == 'pickle':
                            write_itl_to_pickle( itl, _dump_path(outpath, 'itl', it, 'pkl') )
                            write_ol_to_pickle( ol, pickle_dump )

                        if config.data_dump:
                            log.info('[ %s ] Dumping data in %s' %(datetime.now(), outdir) )
                            write_regions_to_pickle( rl, _dump_path(outpath, 'rl', it, 'pkl'), overwrite = True)
                            _write_fits( _dump_path(outpath, 'res', it, 'fits'), reconstruction.residuals )
                            _write_fits( _dump_path(outpath, 'rec', it, 'fits'), reconstruction.restored )
                            _write_fits( _dump_path(outpath, 'dei', it, 'fits'), reconstruction.det_err )

                    log.info('[ %s ]End of iteration (time: %s)\n\n'%(datetime.now(), datetime.now() - start_time_it ))
                    it += 1
                    # The cap applies to a resumed iteration too. It did not
                    # before: the resume branch had its own `continue`, which
                    # jumped over this check, so `max_iter` was not a cap on a
                    # resumed run at all.
                    if it > config.max_iter:
                        break

                if it > config.max_iter:
                    break

        #===========================================================================
        # Write results to disk
        log.info('Finished iterating.\nWriting results to disk in %s' %(outdir) )
        _write_fits( ''.join(( outpath, '.residuals.fits' )), reconstruction.residuals, header )
        _write_fits( ''.join(( outpath, '.restored.fits' )), reconstruction.restored, header )
        # The detection error, not a second copy of the restored image: the old
        # code built the right HDU and then wrote the wrong one (#49).
        _write_fits( ''.join(( outpath, '.deterror.fits' )), reconstruction.det_err, header )
        _write_fits( ''.join(( outpath, '.scl.restored.fits' )), reconstruction.restored_per_level, header )

        with open( ''.join(( outpath, '.cpar.txt' )), 'w+') as conv:
            conv.write('%s\n' %(CONVERGENCE_HEADER))
            for row in monitor.rows:
                conv.write('%s\n' %(' '.join(row)))

        # The convergence plot and the animation are drawn from these files by
        # `dawis.viz`, after the run rather than during it (#46).
        log.info('Draw this run with:\n'
                 '    from dawis.viz import rebuild_animation, rebuild_convergence_plot\n'
                 "    rebuild_convergence_plot('%s')\n"
                 "    rebuild_animation('%s')" %(outpath, outpath))

        log.info( datetime.now() - start_time_algo )

        return RunResult(
            config = config,
            seed = run_seed,
            diagnostics = diagnostics,
            outpath = outpath,
            # The trace, not the counter: an iteration that found no region breaks
            # out before it is measured, so `it` can stand one ahead of what the
            # run actually did.
            n_iterations = len(monitor.rows),
            n_atoms = n_atoms,
            restored = reconstruction.restored,
            residuals = reconstruction.residuals,
            det_err = reconstruction.det_err,
            restored_per_level = reconstruction.restored_per_level,
        )
