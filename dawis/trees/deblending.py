"""Deblending: deciding where one interscale tree stops and the next begins.

Detection returns a flat list of regions, hundreds of them, spread over every
scale. This module turns that list into trees, and the whole difficulty is
that a region can plausibly belong to more than one structure — a faint
neighbour sitting inside the halo of a bright galaxy shares its pixels at the
coarse scales. Deciding which root claims it *is* the deblending.

The decision is made in two steps.

**Choosing the roots.** The brightest region of the list, measured in
noise-normalised intensity, sets a threshold: every region within a factor
``tau`` of it, on that same scale, becomes an interscale maximum. Only one
scale produces roots per call, and the finest and coarsest planes are never
eligible.

**Attaching the regions.** For each root, :func:`interscale_connectivity_serial`
walks the label cube through the footprint of the root and keeps a region only
if its own peak falls inside that footprint. That single rule is what splits
blended structures: two peaks under one coarse region become two trees, each
keeping the fine-scale regions that peak inside it. Two spans bound the
result — a tree must reach at least ``min_span`` scales to be kept at all, and
may not reach further than ``max_span`` below its root. Above ``lvl_sep_big``
both spans collapse to one, so a very coarse root stands alone rather than
swallowing everything beneath it.

:func:`interscale_connectivity_para` is the same function under
``@ray.remote``, run on patches of roots when there are enough to be worth
distributing. It moves behind the ``dawis.parallel`` seam later.
"""

from __future__ import annotations

import logging

import numpy as np

from dawis.cubes import label_datacube, wavelet_datacube
from dawis.detection import region
from dawis.trees.construction import interscale_tree

try:
    import ray
except ImportError:  # ray is the optional [ray] extra
    from dawis._optional import MissingRay
    ray = MissingRay()


@ray.remote
def interscale_connectivity_para(
    interscale_maximum_list: list[region],
    region_list: list[region],
    wavelet_datacube: wavelet_datacube,
    label_datacube: label_datacube,
    det_err_array: np.ndarray,
    level_maximum: int,
    min_span: int = 3,
    max_span: int = 3,
    lvl_sep_big: int = 6,
    min_reg_size: int = 4,
    verbose: bool = False,
) -> list[interscale_tree]:
    """Attach regions to roots, as a ray task.

    Identical to :func:`interscale_connectivity_serial` in every respect
    except the ``@ray.remote`` decoration, and takes and returns exactly what
    it does. :func:`make_interscale_trees` calls it on patches of roots when
    there are more roots than ``size_patch`` and more than one worker; call it
    through ``.remote()``, not directly.

    See Also
    --------
    interscale_connectivity_serial : The same function, and its full
        parameter documentation.

    Notes
    -----
    The duplication is deliberate and temporary. The pair collapses to one
    function behind ``pmap`` when the ``dawis.parallel`` seam lands.
    """

    interscale_tree_list = []
    n_rejected = 0
    for interscale_maximum in interscale_maximum_list:

        if interscale_maximum.level >= lvl_sep_big :
            pmax_span = 1
            pmin_span = 1

        elif interscale_maximum.level < lvl_sep_big :
            pmax_span = max_span
            pmin_span = min_span

        tube = label_datacube.array[interscale_maximum.coords[:,0], interscale_maximum.coords[:,1], :level_maximum + 1]
        covered_label_list = list(np.unique(tube))

        covered_region_list = [x for x in region_list if x.label in covered_label_list]

        connected_region_list = [x for x in region_list if
                         x.label in covered_label_list and
                         [x.x_max, x.y_max] in interscale_maximum.coords and
                         x.area >= min_reg_size and
                         interscale_maximum.level - x.level <= pmax_span - 1]

        span_levels = np.size(np.unique([ x.level for x in connected_region_list ]))

        if span_levels < pmin_span :
            #log.info('Rejected : level = %d span = %d' %(interscale_maximum.level, span_levels))
            n_rejected += 1
            continue

        # Output list of interscale tree and their clipped segmented label array
        interscale_tree_list.append(interscale_tree(interscale_maximum, connected_region_list, wavelet_datacube, label_datacube, det_err_array))

    return interscale_tree_list

def interscale_connectivity_serial(
    interscale_maximum_list: list[region],
    region_list: list[region],
    wavelet_datacube: wavelet_datacube,
    label_datacube: label_datacube,
    det_err_array: np.ndarray,
    level_maximum: int,
    min_span: int = 3,
    max_span: int = 3,
    lvl_sep_big: int = 6,
    min_reg_size: int = 4,
    verbose: bool = False,
) -> list[interscale_tree]:
    """Attach regions to roots, one tree per surviving root.

    For each root, the label cube is read through the pixels of the root, over
    every scale up to ``level_maximum``. That gives the labels the root
    covers. A covered region joins the tree only if its own intensity peak
    falls inside the footprint of the root — the rule that separates two
    blended structures — and only if it is big enough and close enough in
    scale. A root whose members do not span ``min_span`` distinct scales
    yields no tree.

    Parameters
    ----------
    interscale_maximum_list : list of dawis.detection.regions.region
        The roots to build trees around.
    region_list : list of dawis.detection.regions.region
        Every region detected, on every scale, to draw members from.
    wavelet_datacube : dawis.cubes.wavelet_datacube
        Wavelet planes the regions were measured on.
    label_datacube : dawis.cubes.label_datacube
        Label cube the regions were labelled in.
    det_err_array : numpy.ndarray
        Per-scale detection uncertainty, handed to each tree built.
    level_maximum : int
        Scale index of the roots. Bounds the slice of the label cube read
        through each root footprint.
    min_span : int, optional
        Fewest distinct scales a tree must span to be kept. Default is 3.
    max_span : int, optional
        Most scales a member may sit below its root, root included. Default
        is 3.
    lvl_sep_big : int, optional
        Scale index at and above which both spans are forced to 1, so a very
        coarse root stands alone. Default is 6.
    min_reg_size : int, optional
        Smallest area, in pixels, a region may have to join a tree. Default
        is 4.
    verbose : bool, optional
        Unused; kept because the pipeline passes it. Default is False.

    Returns
    -------
    list of dawis.trees.construction.interscale_tree
        One tree per accepted root, in the order the roots were given.
        Rejected roots are dropped silently.
    """

    interscale_tree_list = []
    n_rejected = 0 
    
    for interscale_maximum in interscale_maximum_list:
        
        if interscale_maximum.level >= lvl_sep_big :
            pmax_span = 1
            pmin_span = 1

        elif interscale_maximum.level < lvl_sep_big :
            pmax_span = max_span
            pmin_span = min_span

        tube = label_datacube.array[interscale_maximum.coords[:,0], interscale_maximum.coords[:,1], :level_maximum + 1]
        covered_label_list = list(np.unique(tube))
        
        covered_region_list = [x for x in region_list if x.label in covered_label_list]

        connected_region_list = [x for x in region_list if
                         x.label in covered_label_list and
                         [x.x_max, x.y_max] in interscale_maximum.coords and
                         x.area >= min_reg_size and
                         interscale_maximum.level - x.level <= pmax_span - 1]
        
        span_levels = np.size(np.unique([ x.level for x in connected_region_list ]))

        if span_levels < pmin_span :
            #log.info('Rejected : level = %d span = %d' %(interscale_maximum.level, span_levels))
            n_rejected += 1
            continue

        interscale_tree_list.append(interscale_tree(interscale_maximum, connected_region_list, wavelet_datacube, label_datacube, det_err_array))
        
    return interscale_tree_list

def make_interscale_trees(
    region_list: list[region],
    wavelet_datacube: wavelet_datacube,
    label_datacube: label_datacube,
    det_err_array: np.ndarray,
    tau: float = 0.8,
    min_span: int = 3,
    max_span: int = 3,
    lvl_sep_big: int = 6,
    min_reg_size: int = 4,
    size_patch: int = 100,
    n_cpus: int = 1,
    verbose: bool = False,
) -> list[interscale_tree] | None:
    """Turn a flat list of regions into interscale trees.

    The entry point of the block, called once per pipeline iteration. It picks
    the roots, then hands them to the connectivity step — serially, or in
    patches of ``size_patch`` roots through ray when there are enough of them
    and more than one worker.

    Roots are chosen from a single scale. The list is sorted by
    noise-normalised peak intensity and walked until a region is found outside
    the rejected planes, the finest and the coarsest; that region fixes both
    the working scale and the threshold ``tau`` times its intensity. Every
    region on that scale above the threshold, and no smaller than
    ``min_reg_size``, becomes a root.

    Parameters
    ----------
    region_list : list of dawis.detection.regions.region
        Every region detected, on every scale. **Sorted in place** by
        descending normalised intensity.
    wavelet_datacube : dawis.cubes.wavelet_datacube
        Wavelet planes the regions were measured on.
    label_datacube : dawis.cubes.label_datacube
        Label cube the regions were labelled in.
    det_err_array : numpy.ndarray
        Per-scale detection uncertainty, handed to each tree built.
    tau : float, optional
        Fraction of the brightest eligible intensity above which a region of
        the working scale becomes a root. Default is 0.8. Lower it to detect
        more structures per iteration.
    min_span : int, optional
        Fewest distinct scales a tree must span to be kept. Default is 3.
    max_span : int, optional
        Most scales a member may sit below its root, root included. Default
        is 3.
    lvl_sep_big : int, optional
        Scale index at and above which both spans are forced to 1. Default
        is 6.
    min_reg_size : int, optional
        Smallest area, in pixels, a region may have to be a root or join a
        tree. Default is 4.
    size_patch : int, optional
        Roots per ray task, and the count below which the serial path is
        taken whatever ``n_cpus`` says. Default is 100.
    n_cpus : int, optional
        Worker count. 1 forces the serial path. Default is 1.
    verbose : bool, optional
        Log how many roots were rejected. Default is False.

    Returns
    -------
    list of dawis.trees.construction.interscale_tree or None
        The trees found, possibly empty. ``None`` when every region sits on a
        rejected plane, which the pipeline reads as a signal to stop
        iterating.

    Notes
    -----
    Only one scale contributes roots per call. Structures on the other scales
    are found in later iterations of the pipeline, once the atoms restored
    from these trees have been subtracted from the residual.

    An empty return and a ``None`` return have the same cause in practice —
    ``tau`` or ``min_span`` set too high — and both log the same advice.
    """

    interscale_tree_list = []

    region_list.sort(key = lambda x: x.norm_max_intensity, reverse = True)

    i = 0
    levels_rejected = [ 0, label_datacube.z_size ]
    threshold_maximum = region_list[0].norm_max_intensity * tau
    level_maximum = region_list[0].level

    if len(region_list) > 1:
        while region_list[i].level in levels_rejected :
            i += 1
            if i >= len(region_list):
                log = logging.getLogger(__name__)
                log.info("No interscale maximum found. Please consider lowering parameters 'tau' or 'min_span' if this keeps happening at every level.")
                return None # Regions are all in rejected levels (0 or last)
                
        threshold_maximum = region_list[i].norm_max_intensity * tau
        level_maximum = region_list[i].level
        
        interscale_maximum_list = [x for x in region_list if 
                            x.norm_max_intensity >= threshold_maximum and
                            x.level == level_maximum and
                            x.area >= min_reg_size]
            
    else:
        interscale_maximum_list = region_list
    

    log = logging.getLogger(__name__)
    log.info('Estimating global interscale maxima: %d found.' %(len(interscale_maximum_list)))

    if (len(interscale_maximum_list) <= size_patch) or (n_cpus == 1):
        log.info('Not activating Ray store.')
        # Connectivity
        interscale_tree_list = interscale_connectivity_serial( interscale_maximum_list = interscale_maximum_list,  \
                                                               region_list = region_list,  \
                                                               wavelet_datacube = wavelet_datacube, \
                                                               label_datacube = label_datacube,  \
                                                               det_err_array = det_err_array, \
                                                               level_maximum = level_maximum,  \
                                                               min_span = min_span,  \
                                                               max_span = max_span,  \
                                                               lvl_sep_big = lvl_sep_big, \
                                                               min_reg_size = min_reg_size, \
                                                               verbose = verbose )

    else:
        logging.info('Activating Ray store with patches of size %d.'%(size_patch))
        id_rl = ray.put(region_list)
        id_wdc = ray.put(wavelet_datacube)
        id_ldc = ray.put(label_datacube)
        id_level_maximum = ray.put(level_maximum)
        id_min_span = ray.put(min_span)
        id_max_span = ray.put(max_span)
        id_lvl_sep_big = ray.put(lvl_sep_big)
        id_min_reg_size = ray.put(min_reg_size)
        id_verbose = ray.put(verbose)
        id_det_err_array = ray.put(det_err_array)
        
        interscale_maximum_patch = []
        interscale_tree_patch = []
        
        for interscale_maximum in interscale_maximum_list:

            interscale_maximum_patch.append(interscale_maximum)
            
            # Launch patches of "size_patch" interscale maxima
            if len(interscale_maximum_patch) >= size_patch:
                interscale_tree_patch.append( interscale_connectivity_para.remote( interscale_maximum_list = interscale_maximum_patch,  \
                                                                                   region_list = id_rl,  \
                                                                                   wavelet_datacube = id_wdc,  \
                                                                                   label_datacube = id_ldc,  \
                                                                                   det_err_array = id_det_err_array, \
                                                                                   level_maximum = id_level_maximum,  \
                                                                                   min_span = id_min_span,  \
                                                                                   max_span = id_max_span,  \
                                                                                   lvl_sep_big = id_lvl_sep_big, \
                                                                                   min_reg_size = id_min_reg_size, \
                                                                                   verbose = id_verbose ) )
                interscale_maximum_patch = []

        # Launch leftovers
        if interscale_maximum_patch:
            interscale_tree_patch.append( interscale_connectivity_para.remote( interscale_maximum_list = interscale_maximum_patch,  \
                                                                               region_list = id_rl,  \
                                                                               wavelet_datacube = id_wdc,  \
                                                                               label_datacube = id_ldc,  \
                                                                               det_err_array = id_det_err_array, \
                                                                               level_maximum = id_level_maximum,  \
                                                                               min_span = id_min_span,  \
                                                                               max_span = id_max_span,  \
                                                                               lvl_sep_big = id_lvl_sep_big, \
                                                                               min_reg_size = id_min_reg_size, \
                                                                               verbose = id_verbose ) )
        
        for id_patch in interscale_tree_patch:
            patch = ray.get( id_patch )
            interscale_tree_list = interscale_tree_list + patch

    if verbose == True:
        log.info('%d rejected.' %(len(interscale_maximum_list) - len(interscale_tree_list)))
    if not interscale_tree_list:
        log.info("No interscale maximum found. Please consider lowering parameters 'tau' or 'min_span' if this keeps happening at every level. ")

    return interscale_tree_list

