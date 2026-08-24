"""Drawing one detection.

:func:`plot_region` is ``region.plot`` before #46: the region's cutout of its
wavelet plane, its peak marked, and its outline drawn from the label cube. It
is the picture to reach for when asking why a detection has the shape it has.

It lives here rather than on the class so that :mod:`dawis.detection` — which
every run imports, headless or not — carries no matplotlib.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

from dawis.cubes import label_datacube, wavelet_datacube
from dawis.detection import region as region_class

__all__ = ['plot_region']


def plot_region(region: region_class, wavelet_datacube: wavelet_datacube,
                label_datacube: label_datacube, name: str | None = None,
                show: bool = True, save_path: str | None = None,
                **kwargs) -> None:
    """Show a region's cutout of its wavelet plane, with its outline.

    Parameters
    ----------
    region : dawis.detection.regions.region
        The region to draw.
    wavelet_datacube : dawis.cubes.wavelet_datacube
        The cube the region was detected on, supplying the pixels.
    label_datacube : dawis.cubes.label_datacube
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
        coo_region = np.where( label_datacube.array == region.label )
        min_x_axis = np.min(coo_region[0])
        max_x_axis = np.max(coo_region[0])
        min_y_axis = np.min(coo_region[1])
        max_y_axis = np.max(coo_region[1])
        region.bbox = (min_x_axis, min_y_axis, max_x_axis, max_y_axis)

    plt.imshow( wavelet_datacube.array[ region.bbox[0]:region.bbox[2], \
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

    try:
        clip = label_datacube.array[ region.bbox[0]:region.bbox[2], \
                                       region.bbox[1]:region.bbox[3], \
                                       int(region.level) ]
        clip[np.where(clip != region.label)] = 0.
        plt.contour( clip, \
                region.label, \
                extent = ( region.bbox[1], region.bbox[3], \
                           region.bbox[0], region.bbox[2] ), \
                levels = 0, color = 'black', linewidth = 3 )
    except:
        print('Region too small for countours.')

    if save_path != None:
        plt.savefig(save_path, format = 'pdf')

    #plt.tight_layout()
    if show == True:
        plt.show()
    else:
        plt.close()
