"""The run animation, rebuilt after the run instead of during it.

DAWIS is a greedy loop, so the interesting thing about a run is the *order* in
which it takes structure out of the image: one frame per iteration, showing
the original, what has been restored, what is left, and the atom that
iteration removed. That animation is the clearest picture of the algorithm
there is, which is why it exists.

It used to be written from inside the loop, one frame per iteration, and #5
measured that at 52% of the loop's time — the pipeline spent more time drawing
than restoring. The frames carry no information the run does not already
store: every atom of every iteration is written to disk anyway, and the sum of
the final restored and residual images is the input the loop worked on. So
:func:`rebuild_animation` replays a finished run from those files and draws
the frames afterwards.

That makes the animation a documentation and gallery tool rather than a
runtime cost: a normal run never draws, and any run — including one recorded
long ago, by an older version, as long as its atom dumps are next to it — can
be animated later.

:func:`plot_frame` and :func:`make_gif` are the two halves of the old
``dawis.gif``: one draws a frame, the other calls ``ffmpeg`` to assemble the
frames into an ``.mp4``. ``ffmpeg`` must be on ``PATH``; when it is not, the
frames are still on disk and only the assembly is skipped.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from subprocess import CalledProcessError, run

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import (
    ImageNormalize,
    LinearStretch,
    LogStretch,
    MinMaxInterval,
    ZScaleInterval,
)

from dawis.io import read_ol_from_hdf5, read_ol_from_pickle

__all__ = ['iteration_dumps', 'make_gif', 'plot_frame', 'rebuild_animation']

#: Iteration dumps a run leaves behind, in the two formats the pipeline
#: writes, mapped to the reader that opens them.
_DUMP_READERS = {'.pkl': read_ol_from_pickle, '.hdf5': read_ol_from_hdf5}

_ITERATION = re.compile(r'\.ol\.it(\d+)\.(pkl|hdf5)$')


def plot_frame(
    level: int,
    it: int,
    nobj: int,
    original_image: np.ndarray,
    restored_image: np.ndarray,
    residuals: np.ndarray,
    atom: np.ndarray,
    outpath: str = '',
) -> None:

    '''Draw one iteration of a run as a four-panel frame.

    Parameters
    ----------
    level : int
        Wavelet scale the iteration worked at.
    it : int
        Iteration number, which also numbers the file written.
    nobj : int
        Number of atoms the iteration restored.
    original_image : numpy.ndarray
        The image the run started from.
    restored_image : numpy.ndarray
        Sum of every atom restored up to and including this iteration.
    residuals : numpy.ndarray
        What is left of the image after that subtraction.
    atom : numpy.ndarray
        Sum of the atoms of this iteration alone.
    outpath : str, optional
        Path stem of the run. The frame is written to
        ``<outpath>.frame.it<NNN>.png``.
    '''

    fig = plt.figure(dpi = 500)
    ax = fig.subplots(2, 2, sharex = True, sharey = True, \
                                gridspec_kw = { 'hspace': 0.1, 'wspace': 0.1 })
    fig.suptitle('Level %d\nIteration %d\n%d objects' %(level, it, nobj), fontsize = 10)

    axim = ax[0][0].imshow(original_image,  norm = ImageNormalize( original_image, \
                                            interval = ZScaleInterval(), \
                                            stretch = LinearStretch()), \
                                            origin = 'lower', \
                                            cmap = 'binary' )
    cax = plt.colorbar(axim, ax = ax[0][0], aspect = 10, pad = 0, \
                                    orientation = 'vertical')
    cax.ax.tick_params(labelsize = 5)
    ax[0][0].set_title('Original', fontsize = 8)

    axim = ax[0][1].imshow(residuals, norm = ImageNormalize( original_image, \
                                      interval = ZScaleInterval(),
                                      stretch = LinearStretch()), \
                                      origin = 'lower', \
                                      cmap = 'binary' )

    cax = plt.colorbar(axim, ax = ax[0][1], aspect = 10, pad = 0, \
                                    orientation = 'vertical')
    cax.ax.tick_params(labelsize = 5)
    ax[0][1].set_title('Residuals', fontsize = 8)

    axim = ax[1][0].imshow(restored_image,  norm = ImageNormalize( original_image, \
                                            interval = ZScaleInterval(),
                                            stretch = LinearStretch()), \
                                            origin = 'lower', \
                                            cmap = 'binary' )

    cax = plt.colorbar(axim, ax = ax[1][0], aspect = 10, pad = 0, \
                                    orientation = 'vertical')
    cax.ax.tick_params(labelsize = 5)
    ax[1][0].set_title('Restored', fontsize = 8)

    axim = ax[1][1].imshow(atom, norm = ImageNormalize( atom, \
                                 interval = MinMaxInterval(),
                                 stretch = LogStretch()), \
                                 origin = 'lower', \
                                  cmap = 'binary' )

    cax = plt.colorbar(axim, ax = ax[1][1], aspect = 10, pad = 0, \
                                    orientation = 'vertical')
    cax.ax.tick_params(labelsize = 5)
    ax[1][1].set_title('Atom', fontsize = 8)

    plt.savefig(''.join(( outpath, '.frame.it%03d.png' %(it) )), format = 'png' )
    plt.close()


def make_gif( framerate: int, outpath: str ) -> None:

    '''Assemble the frames of a run into an ``.mp4`` with ``ffmpeg``.

    Parameters
    ----------
    framerate : int
        Frames per second of the animation.
    outpath : str
        Path stem of the run. Reads ``<outpath>.frame.it<NNN>.png`` and
        writes ``<outpath>.run.mp4``.

    Notes
    -----
    Failure is logged, not raised, and this is the one place in the package
    where that is the right answer: the frames are the expensive part and
    they are already on disk, so a missing or failing ``ffmpeg`` costs only
    the assembly. It is logged with the error that caused it, and ``check``
    is set so that a non-zero exit is one of the failures that reaches the
    log rather than passing unnoticed.
    '''

    try:
        args_ffmpeg = [ 'ffmpeg', '-framerate', str(framerate), '-y', '-i', \
                ''.join(( outpath, '.frame.it%03d.png' )), \
                ''.join(( outpath, '.run.mp4' )) ]

        run( args_ffmpeg, check = True )
    except (CalledProcessError, OSError) as error:
        logging.info('Could not assemble the frames into a video: %s', error)


def iteration_dumps(outpath: str) -> list[tuple[int, str]]:
    """List the atom dumps of a finished run, in iteration order.

    Parameters
    ----------
    outpath : str
        Path stem of the run, as handed to the pipeline —
        ``<outdir>/<infile without .fits>``.

    Returns
    -------
    list of (int, str)
        Iteration number and path, one per dump, sorted by iteration. A run
        that was resumed can hold both formats; a pickle and an HDF5 file of
        the same iteration are the same atoms, so only one is returned — the
        first in name order, which makes the choice stable rather than a
        matter of what the filesystem lists first.
    """

    found = {}
    for path in sorted(glob.glob(''.join(( outpath, '.ol.it*' )))):
        match = _ITERATION.search(path)
        if match:
            found.setdefault(int(match.group(1)), path)

    return sorted(found.items())


def rebuild_animation(outpath: str, framerate: int = 3, show_progress: bool = True) -> int:
    """Rebuild a finished run's animation from what the run stored.

    Replays the run: the atoms of each iteration are read back, summed into
    the restored image and taken out of the residuals exactly as the loop did,
    and one frame is drawn per iteration. The frames are then assembled into
    ``<outpath>.run.mp4``.

    Parameters
    ----------
    outpath : str
        Path stem of the run, as handed to the pipeline —
        ``<outdir>/<infile without .fits>``. Every file read and written
        hangs off it.
    framerate : int, optional
        Frames per second of the animation. Default is 3.
    show_progress : bool, optional
        Log one line per frame. Default is True.

    Returns
    -------
    int
        Number of frames drawn, one per iteration of the run.

    Raises
    ------
    FileNotFoundError
        If the run left no atom dumps, or no restored and residual images, at
        ``outpath``.

    Notes
    -----
    The image the loop worked on is recovered as ``restored + residuals``
    rather than re-read from the input FITS file, because the pipeline
    subtracts the background before iterating and inpaints the residuals as it
    goes. The sum of what a run wrote is what it actually iterated on, and it
    needs no knowledge of where the input came from.

    Frames are drawn from the atoms alone, so a run made with ``data_dump``
    off can still be animated.
    """

    log = logging.getLogger(__name__)

    dumps = iteration_dumps(outpath)
    if not dumps:
        raise FileNotFoundError(
            "no atom dumps found for run '%s': expected '%s.ol.itNNN.pkl' or "
            "'.hdf5'" % (outpath, outpath)
        )

    restored_path = ''.join(( outpath, '.restored.fits' ))
    residuals_path = ''.join(( outpath, '.residuals.fits' ))
    for path in (restored_path, residuals_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "run '%s' is missing '%s': an animation is rebuilt from a "
                "finished run" % (outpath, path)
            )

    image = np.asarray(fits.getdata(restored_path), dtype = float) \
                + np.asarray(fits.getdata(residuals_path), dtype = float)

    res = np.copy(image)
    rec = np.zeros(image.shape)

    for it, path in dumps:

        ol = _DUMP_READERS[os.path.splitext(path)[1]](path)

        atom = np.zeros(image.shape)
        for o in ol:
            x_min, y_min, x_max, y_max = o.bbox
            atom[ x_min : x_max, y_min : y_max ] += o.image

        res -= atom
        rec += atom
        level = np.max([ o.level for o in ol ]) if ol else 0

        if show_progress:
            log.info('Drawing frame %03d of %03d (%d atoms)'
                     %(it, dumps[-1][0], len(ol)))

        plot_frame( level = level, it = it, nobj = len(ol), \
                                  original_image = image, \
                                  restored_image = rec, \
                                  residuals = res, \
                                  atom = atom, \
                                  outpath = outpath )

    make_gif( framerate = framerate, outpath = outpath )

    return len(dumps)
