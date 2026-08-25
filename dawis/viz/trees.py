"""Drawing one interscale tree.

:func:`plot_interscale_tree` is ``interscale_tree.plot`` before #46: the
tree's summed wavelet coefficients on the left, the regions it groups on the
right. It is the picture to reach for when asking whether the deblending put
the right regions in the same tree.

It lives here rather than on the class so that :mod:`dawis.trees` — the block
the restoration calls once per atom — carries no matplotlib.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

from dawis.cubes import LabelCube, WaveletCube
from dawis.trees import interscale_tree
from dawis.viz.regions import contourable

__all__ = ['plot_interscale_tree']


def plot_interscale_tree(tree: interscale_tree,
                         wavelet_cube: WaveletCube,
                         label_cube: LabelCube,
                         name: str | None = None, show: bool = True,
                         save_path: str | None = None, **kwargs) -> None:
    """Draw a tree: its summed wavelet coefficients beside its regions.

    Parameters
    ----------
    tree : dawis.trees.construction.interscale_tree
        The tree to draw.
    wavelet_cube : dawis.cubes.WaveletCube
        Wavelet planes the tree was built on.
    label_cube : dawis.cubes.LabelCube
        Label cube the tree was built on.
    name : str, optional
        Figure title. Default is the label of the interscale maximum.
    show : bool, optional
        Display the figure and block. Default is True. When False the figure
        is closed instead, which is what saving in a loop wants.
    save_path : str, optional
        Write the figure there as PNG. Default is not to save.
    **kwargs
        Passed to :func:`matplotlib.pyplot.imshow`. A ``norm`` entry is
        honoured; without one a symmetric-log norm is used.
    """

    fig, ax = plt.subplots(1, 2, sharex = True, sharey = True, \
                              gridspec_kw = {'hspace': 0.5, 'wspace': 0.5})

    if name == None:
        fig.suptitle('Interscale tree of region n°%d' %(tree.interscale_maximum.label) )
    else:
        fig.suptitle(name)

    wavelet_image = np.sum(wavelet_cube.array[ tree.bbox[0]:tree.bbox[2], \
                                                   tree.bbox[1]:tree.bbox[3], \
                                                   0:tree.interscale_maximum.level + 1 ], axis = 2)

    norm = kwargs.pop('norm', None)
    if norm is not None:
        im = ax[0].imshow( wavelet_image, \
                extent = ( tree.bbox[1], tree.bbox[3], \
                           tree.bbox[0], tree.bbox[2] ), \
                origin = 'lower', \
                **kwargs )
        plt.colorbar(im, ax = ax[0], orientation = 'vertical')
    else:
        logthresh = 1E-3
        norm = SymLogNorm(logthresh)
        im = ax[0].imshow( wavelet_image, extent = ( tree.bbox[1], tree.bbox[3], \
                                                     tree.bbox[0], tree.bbox[2] ), \
                                          origin = 'lower', \
                                          norm = norm, \
                                          **kwargs )

        maxlog = np.log10( np.ceil( np.max( wavelet_image )))
        minlog = np.log10( np.ceil( - np.min( wavelet_image )))
        tick_locations = (list(- np.logspace(1, minlog, num = 2, base = 10)) \
                            + [0] \
                            + list(np.logspace(1, maxlog, num = 2, base = 10)))
        plt.colorbar(im, ax = ax[0], orientation = 'vertical', \
                                     ticks = tick_locations,
                                     aspect = 10,
                                     shrink = 0.5)


    label_image = np.zeros((tree.bbox[2] - tree.bbox[0], \
                            tree.bbox[3] - tree.bbox[1]))

    for i, region in enumerate(tree.region_list):

        ax[0].plot( region.y_max, region.x_max, '+' )

        clip = np.copy(label_cube.array[ tree.bbox[0]:tree.bbox[2], \
                                             tree.bbox[1]:tree.bbox[3], \
                                             int(region.level) ])

        clip[np.where(clip != region.label)] = 0.

        if contourable(clip):
            ax[0].contour( clip, region.label,
                                 extent = ( tree.bbox[1], tree.bbox[3], \
                                            tree.bbox[0], tree.bbox[2] ), \
                                 levels = 0, \
                                 color = 'cyan')
        else:
            print('Region too small for contours.')

        clip[np.where(clip == region.label)] = 1
        label_image += clip
        ax[1].plot( region.y_max, region.x_max, '+' )

    ax[1].imshow(label_image, extent = ( tree.bbox[1], tree.bbox[3], \
                                         tree.bbox[0], tree.bbox[2] ), \
                              origin = 'lower',
                              cmap = 'binary'  )

    if save_path != None:
        plt.savefig(save_path, format = 'png')

    if show == True:
        plt.show()
    else:
        plt.close()
