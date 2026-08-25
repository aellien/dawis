"""Drawing one detection.

:func:`plot_region` is ``region.plot`` before #46: the region's cutout of its
wavelet plane, its peak marked, and its outline drawn from the label cube. It
is the picture to reach for when asking why a detection has the shape it has.

It lives here rather than on the class so that :mod:`dawis.detection` — which
every run imports, headless or not — carries no matplotlib.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

from dawis.cubes import LabelCube, WaveletCube
from dawis.detection import region as region_class

__all__ = ['plot_region']

log = logging.getLogger(__name__)


def contourable(clip: np.ndarray) -> bool:
    """Whether an outline can be drawn over this cutout.

    :func:`matplotlib.pyplot.contour` needs at least a 2x2 grid to place a
    line between samples, and raises :exc:`TypeError` below that. A region
    one pixel wide is a normal detection at the finest scale, not a fault,
    so it is tested for rather than caught: catching would swallow every
    other :exc:`TypeError` the drawing can raise along with it.
    """
    return clip.ndim == 2 and min(clip.shape) >= 2


def plot_region(region: region_class, wavelet_cube: WaveletCube,
                label_cube: LabelCube, name: str | None = None,
                show: bool = True, save_path: str | None = None,
                **kwargs) -> None:
    """Show a region's cutout of its wavelet plane, with its outline.

    Parameters
    ----------
    region : dawis.detection.regions.region
        The region to draw.
    wavelet_cube : dawis.cubes.WaveletCube
        The cube the region was detected on, supplying the pixels.
    label_cube : dawis.cubes.LabelCube
        The label cube, supplying the outline.
    name : str, optional
        Figure title. Default is None, which titles it by label.
    show : bool, optional
        Display the figure. Default is True; False closes it instead.
    save_path : str, optional
        Write the figure as PDF. Default is None, which writes nothing.
    **kwargs
        Passed through to :func:`matplotlib.pyplot.imshow`.

    Notes
    -----
    A region with no ``bbox`` gets one computed here, by scanning the whole
    label cube for its label, and the result is cached on the instance.

    Carried through unchanged from ``region.plot``, defect included: that test
    is written ``region.bbox == False``, and a region built with a ``bbox``
    stores it as a :class:`numpy.ndarray`, whose comparison is an array — so
    the ``if`` raises ``ValueError`` and only a region *without* a bbox can be
    drawn. Fixing it means deciding what a region's empty property should be,
    which belongs with the region class, not with a move.
    """

    fig = plt.figure()

    if name == None:
        fig.suptitle('Region n°%d' %(region.label) )
    else:
        fig.suptitle(name)

    if region.bbox == False:
        coo_region = np.where( label_cube.array == region.label )
        min_x_axis = np.min(coo_region[0])
        max_x_axis = np.max(coo_region[0])
        min_y_axis = np.min(coo_region[1])
        max_y_axis = np.max(coo_region[1])
        region.bbox = (min_x_axis, min_y_axis, max_x_axis, max_y_axis)

    plt.imshow( wavelet_cube.array[ region.bbox[0]:region.bbox[2], \
                                        region.bbox[1]:region.bbox[3], \
                                        int(region.level) ], \
                extent = ( region.bbox[1], region.bbox[3], \
                           region.bbox[0], region.bbox[2] ), \
                origin = 'lower', \
                **kwargs )

    plt.plot( region.y_max, region.x_max, '+', color = 'black' )

    plt.gca().add_artist(AnchoredText( r'level = %d'%(region.level), \
                                            loc = 'upper right', \
                                            frameon=True, \
                                            pad=0.4, \
                                            borderpad=0.1, \
                                            prop=dict(size=10)))

    clip = label_cube.array[ region.bbox[0]:region.bbox[2], \
                                   region.bbox[1]:region.bbox[3], \
                                   int(region.level) ]
    clip[np.where(clip != region.label)] = 0.
    if contourable(clip):
        plt.contour( clip, \
                region.label, \
                extent = ( region.bbox[1], region.bbox[3], \
                           region.bbox[0], region.bbox[2] ), \
                levels = 0, color = 'black', linewidth = 3 )
    else:
        log.debug('Region too small for contours.')

    if save_path != None:
        plt.savefig(save_path, format = 'pdf')

    #plt.tight_layout()
    if show == True:
        plt.show()
    else:
        plt.close()
