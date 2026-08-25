"""Labelling of the connected regions of the multiscale support.

Second detection step, and the counterpart of :mod:`dawis.detection.thresholding`:
the support says *which* pixels are significant, labelling says which of them
belong to the same object. Each connected patch of significant pixels becomes
a *region* and receives an integer label.

Labels are unique across the whole wavelet space, not per plane. The offset
applied at each scale is what lets the tree builder name a region by its label
alone, without carrying the scale alongside it.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.measure import label

from dawis.cubes import LabelCube, SupportCube


def label_regions(
    support_cube: SupportCube,
    verbose: bool = False,
) -> LabelCube:
    """Label the connected regions of a multiscale support.

    Each plane of the support is labelled independently with 4-connectivity,
    then the labels of plane ``l`` are offset by the total number of regions
    found on all planes below it, so no label is reused anywhere in the cube.

    Parameters
    ----------
    support_cube : dawis.cubes.SupportCube
        Binary significance cube, as returned by
        :func:`~dawis.detection.thresholding.hard_threshold`.
    verbose : bool, optional
        Log the region count and label range of each scale. Default is False.

    Returns
    -------
    dawis.cubes.LabelCube
        Integer cube of labels, 0 outside any region, carrying the per-scale
        region counts.

    Notes
    -----
    Connectivity is 1 — pixels touching edge to edge belong to the same
    region, pixels touching only at a corner do not.

    A cube with a single plane raises ``NameError``: the function frees its
    loop temporaries at the end, and those are only bound from the second
    plane onwards. The pipeline always decomposes on several scales.
    """

    label_array = np.zeros(support_cube.shape, order = 'f')
    lab_counts = np.zeros(support_cube.z_size)

    if verbose == True:
        log = logging.getLogger(__name__)
        log.info('\nLabelling regions')

    for level in range(0, support_cube.z_size):
        label_array[:, :, level], lab_counts[level] = label(support_cube.array[:,:,level], \
                                                      connectivity = 1, \
                                                      return_num = True)
        if level != 0:
            reg = np.where( label_array[:, :, level] != 0 )               # here we make
            temp = label_array[:, :, level]                           # sure that the
            temp[reg] = temp[reg] + np.sum( lab_counts[0:level] ) # labels are
            label_array[:, :, level] = temp                           # unique through
                                                            # the whole
                                                            # wavelet space.
        if verbose == True:
            log.info('level %02d nregions %d labels %d - %d'\
                    %(level+1,lab_counts[level],\
                    np.max(label_array[:,:,level])-lab_counts[level]+1,\
                    np.max(label_array[:,:,level])))

    del temp, reg

    label_dc = LabelCube(label_array, lab_counts, support_cube.fheader)

    return label_dc
