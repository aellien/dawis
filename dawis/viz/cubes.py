"""Drawing cubes.

The four display methods that used to hang off the cube classes live here as
functions, so that :mod:`dawis.cubes` — shared vocabulary imported by every
block — carries no matplotlib.

:func:`plot_cube` is the grid of planes, and it is one name rather than two.
The classes carried two versions of it: the plain one on
:class:`~dawis.cubes.Cube`, and an override on
:class:`~dawis.cubes.WaveletCube` that puts a symmetric-log norm on
coefficients which straddle zero over several decades. Method overriding is
how the class tree spelled that choice; :func:`functools.singledispatch`
spells it here, on the same types and with the same result, so a caller
holding any cube calls one name and no longer has to know which class it
holds.

The two bodies stay as separate as they were: the branches share most of
their grid make-up, and they shared it as two overriding methods before this
move too. Folding them into one body is a rewrite of drawing code, not a
carve, so it is left for whoever next has a reason to touch it.

The other three are per-type by nature — standard deviation against scale
belongs to a wavelet cube, the noise histogram to a support cube, the label
counts to a label cube — so they stay separate functions rather than becoming
dispatch branches that raise on three types out of four.

The plotting bodies are the ones the methods had.
"""

from __future__ import annotations

from functools import singledispatch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

from dawis.cubes import (
    Cube,
    LabelCube,
    SupportCube,
    WaveletCube,
)

__all__ = [
    'plot_cube',
    'plot_label_counts',
    'plot_noise_histogram',
    'plot_wavelet_stds',
]


@singledispatch
def plot_cube(cube: Cube, ncol: int = 5, name: str | None = None,
              show: bool = True, save_path: str | None = None, **kwargs) -> None:
    """Draw every plane of a cube as a grid of images.

    Dispatches on the type of ``cube``: a
    :class:`~dawis.cubes.WaveletCube` gets the symmetric-log treatment
    its coefficients need, every other cube gets the plain grid below.

    Parameters
    ----------
    cube : dawis.cubes.Cube
        The cube to draw.
    ncol : int, optional
        Number of columns in the grid. Default is 5.
    name : str, optional
        Figure title. Default is None, which titles it by cube type.
    show : bool, optional
        Display the figure and block. Default is True. False closes it
        instead, which is what saving in a loop wants.
    save_path : str, optional
        Write the figure there as PNG. Default is None, which writes nothing.
    **kwargs
        Passed to :func:`matplotlib.pyplot.imshow`.

    Examples
    --------
    >>> from numpy.random import normal
    >>> import dawis
    >>> import dawis.viz as viz
    >>> dc = dawis.Cube(normal(0, 1, (100, 100, 10)), dctype = 'NONE')
    >>> viz.plot_cube(dc, name = 'Cube of random values')
    >>> viz.plot_cube(dc, ncol = 2, cmap = 'hot', interpolation = 'bicubic')
    """

    # Plot make-up.
    if cube.z_size > ncol:
        nline = int( np.ceil(cube.z_size / ncol) )
    else:
        nline = 1

    fig, grid = plt.subplots(nline, ncol, sharex = False, sharey = False, \
                              gridspec_kw = {'hspace': 0.2, 'wspace': 0.2})

    if name == None:
        fig.suptitle('Cube of type %s'%(cube.dctype) )
    else:
        fig.suptitle(name)

    lvl = 0
    for i in range(nline):
        for j in range(ncol):
            lvl += 1
            if lvl > cube.z_size:
                if nline == 1:
                    grid[j].set_frame_on(False)
                    grid[j].get_xaxis().set_ticks([])
                    grid[j].get_yaxis().set_ticks([])
                else:
                    grid[i][j].set_frame_on(False)
                    grid[i][j].get_xaxis().set_ticks([])
                    grid[i][j].get_yaxis().set_ticks([])
            else:
                if nline == 1:
                    ax = grid[j]
                else:
                    ax = grid[i][j]
                if lvl == 1:
                    norm = ax.imshow(cube.array[:, :, lvl - 1], **kwargs)
                else:
                    im = ax.imshow(cube.array[:, :, lvl - 1], **kwargs)
                cax = plt.colorbar(norm, ax = ax, aspect = 10, pad = 0, \
                                                orientation = 'horizontal')
                ax.set_title('z = %d' %(lvl))
                ax.xaxis.tick_top()
                if j != 0:
                    ax.get_yaxis().set_ticks([])
                if i != 0:
                    ax.get_xaxis().set_ticks([])

    if save_path != None:
        plt.savefig(save_path, format = 'png')

    plt.tight_layout()
    if show == True:
        plt.show()
    else:
        plt.close()


@plot_cube.register
def _plot_wavelet_cube(cube: WaveletCube, ncol: int = 5,
                       name: str | None = None, show: bool = True,
                       save_path: str | None = None, **kwargs) -> None:
    """Draw a wavelet cube's planes, on a symmetric-log colour scale.

    The branch :meth:`dawis.cubes.WaveletCube.waveplot` was before #46.
    Same arguments as :func:`plot_cube`; a ``norm`` passed in ``kwargs`` is
    honoured and suppresses the symmetric-log default.
    """

    # Plot make-up.
    if cube.z_size > ncol:
        nline = int( np.ceil(cube.z_size / ncol) )
    else:
        nline = 1
        ncol = cube.z_size

    fig, grid = plt.subplots(nline, ncol, sharex = False, sharey = False, \
                              gridspec_kw = {'hspace': 0.2, 'wspace': 0.2})

    if name == None:
        fig.suptitle('Cube of type %s'%(cube.dctype) )
    else:
        fig.suptitle(name)

    lvl = 0
    for i in range(nline):
        for j in range(ncol):
            lvl += 1
            if lvl > cube.z_size:
                if nline == 1:
                    grid[j].set_frame_on(False)
                    grid[j].get_xaxis().set_ticks([])
                    grid[j].get_yaxis().set_ticks([])
                else:
                    grid[i][j].set_frame_on(False)
                    grid[i][j].get_xaxis().set_ticks([])
                    grid[i][j].get_yaxis().set_ticks([])
            else:
                if nline == 1:
                    ax = grid[j]
                else:
                    ax = grid[i][j]

                try:
                    norm = kwargs.pop('norm')
                    if lvl == 1:
                        im_norm = ax.imshow(cube.array[:, :, lvl - 1], **kwargs)
                    else:
                        im = ax.imshow(cube.array[:, :, lvl - 1], **kwargs)
                    cax = plt.colorbar(im_norm, ax = ax, aspect = 10, pad = 0, \
                                                    orientation = 'horizontal')
                except:
                    logthresh = 1E-3
                    norm = SymLogNorm(logthresh)
                    if lvl == 1:
                        im_norm = ax.imshow(cube.array[:, :, lvl - 1], norm = norm, **kwargs)
                        maxlog = np.log10( np.ceil( np.max( cube.array[:, :, lvl - 1] )))
                        minlog = np.log10( np.ceil( - np.min( cube.array[:, :, lvl - 1] )))
                    else:
                        im = ax.imshow(cube.array[:, :, lvl - 1], norm = norm, **kwargs)

                    tick_locations = (list(- np.logspace(1, minlog, num = 2, base = 10)) \
                                        + [0] \
                                        + list(np.logspace(1, maxlog, num = 2, base = 10)))
                    cax = plt.colorbar(im_norm, ax = ax, aspect = 10, pad = 0, \
                                                orientation = 'horizontal', \
                                                ticks = tick_locations)
                ax.set_title('z = %d' %(lvl))
                ax.xaxis.tick_top()
                if j != 0:
                    ax.get_yaxis().set_ticks([])
                if i != 0:
                    ax.get_xaxis().set_ticks([])

    if save_path != None:
        plt.savefig(save_path, format = 'png')

    #plt.tight_layout()
    if show == True:
        plt.show()
    else:
        plt.close()


def plot_wavelet_stds(cube: WaveletCube, name: str | None = None, show: bool = True,
                      save_path: str | None = None, **kwargs) -> None:
    """Plot the standard deviation of a wavelet cube against scale.

    Parameters
    ----------
    cube : dawis.cubes.WaveletCube
        The cube to measure.
    name : str, optional
        Figure title. Default is None, which titles it generically.
    show : bool, optional
        Display the figure and block. Default is True. False closes it.
    save_path : str, optional
        Write the figure there as PDF. Default is None, which writes nothing.
    **kwargs
        Passed to :func:`matplotlib.pyplot.plot`.

    Notes
    -----
    Carried through unchanged from ``WaveletCube.std_plot``, defect
    included: the log scale is set with ``basey``, an argument matplotlib
    removed in 3.3, so this raises ``TypeError`` on any current install. It is
    a one-word fix — ``base`` — but a drawing change rather than a move, so it
    is left for whoever next has a reason to draw this.
    """

    fig = plt.figure()

    if name == None:
        fig.suptitle('Standard deviation against scale')
    else:
        fig.suptitle(name)

    plt.plot(np.std(cube.array, axis = (0,1)), **kwargs)
    plt.yscale('log', basey = 10)

    if save_path != None:
        plt.savefig(save_path, format = 'pdf')

    #plt.tight_layout()
    if show == True:
        plt.show()
    else:
        plt.close()


def plot_noise_histogram(cube: SupportCube, name: str | None = None, show: bool = True,
                         save_path: str | None = None, **kwargs) -> None:
    """Histogram the noise pixels a support cube kept from thresholding.

    Parameters
    ----------
    cube : dawis.cubes.SupportCube
        The support cube, carrying the sigma-clipped noise pixels.
    name : str, optional
        Figure title. Default is None, which titles it by cube type.
    show : bool, optional
        Display the figure and block. Default is True. False closes it.
    save_path : str, optional
        Write the figure there as PNG. Default is None, which writes nothing.
    **kwargs
        Passed to :func:`matplotlib.pyplot.hist`.
    """

    fig = plt.figure()

    if name == None:
        fig.suptitle('Cube of type %s'%(cube.dctype) )
    else:
        fig.suptitle(name)

    plt.hist(cube.noise_pixels.flatten(),  bins = 'auto',  \
                                           histtype = 'bar', \
                                           rwidth = 10. , \
                                           align='left', \
                                           **kwargs)

    if save_path != None:
        plt.savefig(save_path, format = 'png')

    #plt.tight_layout()
    if show == True:
        plt.show()
    else:
        plt.close()


def plot_label_counts(cube: LabelCube, name: str | None = None, show: bool = True,
                      save_path: str | None = None, **kwargs) -> None:
    """Plot how many labelled regions a label cube holds at each scale.

    Parameters
    ----------
    cube : dawis.cubes.LabelCube
        The cube to read the per-scale counts from.
    name : str, optional
        Figure title. Default is None, which titles it generically.
    show : bool, optional
        Display the figure and block. Default is True. False closes it.
    save_path : str, optional
        Write the figure there as PDF. Default is None, which writes nothing.
    **kwargs
        Passed to :func:`matplotlib.pyplot.plot`.
    """

    fig = plt.figure()

    if name == None:
        fig.suptitle('Label counts against scale')
    else:
        fig.suptitle(name)

    plt.plot(cube.lab_counts, **kwargs)

    if save_path != None:
        plt.savefig(save_path, format = 'pdf')

    #plt.tight_layout()
    if show == True:
        plt.show()
    else:
        plt.close()
