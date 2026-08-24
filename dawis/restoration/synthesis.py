"""Atom synthesis: turning interscale trees into the objects we subtract.

The step that costs the run. Detection says where the structures are and
:mod:`dawis.trees` groups them across scales; this module reconstructs each
tree as an image and wraps it as a :class:`restored_object` — an *atom*, in
the vocabulary of ``CONTEXT.md``. The pipeline sums the atoms of an iteration,
subtracts them from the residual, and goes round again.

The solver itself is not here. The conjugate-gradient descent lives on the
tree, as :meth:`~dawis.trees.construction.interscale_tree.CG_minimization`,
because it works on the tubes the tree cuts out of the cubes. What this module
owns is the choice of how to call it.

:func:`restore_object` makes that choice for one tree, on its shape and its
scale: small or elongated structures below ``lvl_sep_lin`` are synthesised
with the Haar filter, everything else with the B3-spline, starting from the
plain sum of planes at or below ``lvl_sep_op`` and from the adjoint operator
above it. It also builds the detection-error image and traps a diverged solve.

:func:`restore_objects_default` runs that over a list of trees. Below
``size_patch_big`` trees, or on one worker, it loops in process. Above it, it
sorts by area, splits into big and small patches, and hands them to ray
through :func:`restore_patch`. #5 measured a 308% flux difference between the
two, so the worker count currently selects an outcome as well as a degree of
parallelism; #56 unifies them onto the published serial path. Until it lands,
a run whose output is being compared with another must hold ``n_cpus`` fixed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from dawis.cubes import label_datacube, wavelet_datacube

# Annotations only. dawis.trees imports the deconvolution of this package, so
# importing it back at runtime would close a cycle. The alias is needed
# because restore_object's first parameter is itself named interscale_tree.
if TYPE_CHECKING:
    from dawis.trees import interscale_tree as interscale_tree_type

try:
    import ray
except ImportError:  # ray is the optional [ray] extra
    from dawis._optional import MissingRay
    ray = MissingRay()


class restored_object(object):
    """One structure reconstructed from an interscale tree — an *atom*.

    The output unit of the pipeline: a synthesised image, the bounding box it
    sits in within the full frame, and the descriptors of the interscale
    maximum it came from. Atoms are raw synthesis building blocks and not yet
    astrophysically meaningful sources; grouping them into source models is
    atom selection, which is a later step.

    Constructed with ``blank = True`` and no other argument, the instance is
    left empty for a reader to fill in — that is how
    :mod:`dawis.io.store` rebuilds one off disk.

    Parameters
    ----------
    image : numpy.ndarray, optional
        Synthesised image of the object, of the shape of ``bbox``.
    det_err_image : numpy.ndarray, optional
        Detection-error image over the same footprint.
    bbox : tuple of int, optional
        Bounding box of the object in the full frame, as
        ``(x_min, y_min, x_max, y_max)``. Stored as a tuple.
    level : int, optional
        Wavelet scale of the interscale maximum.
    eccentricity : float, optional
        Eccentricity of the interscale maximum.
    filter_kw : str, optional
        Which filter synthesised the object, ``'HAAR'`` or ``'BSPL'``. Stored
        as ``str(filter_kw)`` under the attribute ``filter``.
    flag_convergence : bool, optional
        False when the solve produced NaNs and the image was zeroed.
    sum_wr : float, optional
        Sum of the wavelet residual at the last iteration of the solve.
    norm_wr : float, optional
        Norm of the wavelet residual at the last iteration of the solve.
    blank : bool, optional
        If True, set no attributes at all and ignore every other argument.
        Default False.
    """

    def __init__(
        self,
        image: np.ndarray | None = None,
        det_err_image: np.ndarray | None = None,
        bbox: tuple[int, ...] | None = None,
        level: int | None = None,
        eccentricity: float | None = None,
        filter_kw: str | None = None,
        flag_convergence: bool | None = None,
        sum_wr: float | None = None,
        norm_wr: float | None = None,
        blank: bool = False,
    ):
        
        if blank == False:
            
            self.bbox = tuple(bbox)
            self.image = image
            self.det_err_image = det_err_image
            self.level = level
            self.eccentricity = eccentricity
            self.filter = str(filter_kw)
            self.flag_convergence = flag_convergence
            self.sum_wr = sum_wr
            self.norm_wr = norm_wr

@ray.remote
def restore_patch(
    interscale_tree_patch: list[interscale_tree_type],
    wavelet_datacube: wavelet_datacube,
    label_datacube: label_datacube,
    extent_sep: float,
    ecc_sep: float,
    lvl_sep_lin: int,
    lvl_sep_big: int,
    lvl_sep_op: int,
    gamma: float,
    rm_gamma_for_big: bool,
    deconv: bool,
) -> list[restored_object]:
    """Synthesise a patch of trees on a ray worker.

    The parallel counterpart of the in-process loop of
    :func:`restore_objects_default`: the same call to :func:`restore_object`
    per tree, the same attenuation rule, over a slice of the tree list. Called
    through ``restore_patch.remote(...)``, which needs the ``[ray]`` extra.

    Parameters
    ----------
    interscale_tree_patch : list of interscale_tree
        The trees this worker is responsible for.
    wavelet_datacube : dawis.cubes.wavelet_datacube
        Wavelet coefficients of the residual being restored.
    label_datacube : dawis.cubes.label_datacube
        Region labels matching ``wavelet_datacube``.
    extent_sep : float
        Extent below which a tree takes the Haar branch.
    ecc_sep : float
        Eccentricity at or above which a tree takes the Haar branch.
    lvl_sep_lin : int
        Scale at or above which the Haar branch is never taken.
    lvl_sep_big : int
        Scale at or above which an object counts as big, for ``gamma``.
    lvl_sep_op : int
        Scale above which the adjoint synthesis operator replaces the sum.
    gamma : float
        Attenuation applied to the synthesised image.
    rm_gamma_for_big : bool
        If True, skip the attenuation for objects at or above
        ``lvl_sep_big``.
    deconv : bool
        Whether the solver deconvolves each plane by the B3-spline kernel.

    Returns
    -------
    list of restored_object
        One atom per tree in the patch, in the order given.
    """

    object_patch = []
    for tree in interscale_tree_patch:

        image, det_err_image, filter_kw, flag_convergence, sum_wr, norm_wr = restore_object( tree, \
                                                                              wavelet_datacube, \
                                                                              label_datacube, \
                                                                              extent_sep, \
                                                                              ecc_sep, \
                                                                              lvl_sep_lin, \
                                                                              lvl_sep_big, \
                                                                              lvl_sep_op, \
                                                                              deconv )

        if (rm_gamma_for_big == False) or (tree.interscale_maximum.level < lvl_sep_big):
            image = image * gamma  # add attenuation factor



        object_patch.append(restored_object(image, det_err_image, tree.bbox, \
                                            tree.interscale_maximum.level, \
                                            tree.interscale_maximum.eccentricity, \
                                            filter_kw, \
                                            flag_convergence, \
                                            sum_wr, \
                                            norm_wr ))

    return object_patch

def restore_object(
    interscale_tree: interscale_tree_type,
    wavelet_datacube: wavelet_datacube,
    label_datacube: label_datacube,
    extent_sep: float,
    ecc_sep: float,
    lvl_sep_lin: int,
    lvl_sep_big: int,
    lvl_sep_op: int = 2,
    deconv: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, str, bool, float, float]:
    """Synthesise one interscale tree into an image.

    Picks the filter and the starting operator from the shape and the scale of
    the tree, then runs
    :meth:`~dawis.trees.construction.interscale_tree.CG_minimization` with
    them. A tree that is small or elongated — extent below ``extent_sep`` or
    eccentricity at or above ``ecc_sep`` — and sits below ``lvl_sep_lin``
    takes the Haar filter; everything else takes the B3-spline, from the plain
    sum of planes at or below ``lvl_sep_op`` and from the adjoint operator
    above it.

    A solve that produces NaNs is not an error: the image is zeroed and
    ``flag_convergence`` comes back False.

    Parameters
    ----------
    interscale_tree : interscale_tree
        The tree to synthesise.
    wavelet_datacube : dawis.cubes.wavelet_datacube
        Wavelet coefficients of the residual being restored.
    label_datacube : dawis.cubes.label_datacube
        Region labels matching ``wavelet_datacube``.
    extent_sep : float
        Extent below which the Haar branch is taken.
    ecc_sep : float
        Eccentricity at or above which the Haar branch is taken.
    lvl_sep_lin : int
        Scale at or above which the Haar branch is never taken.
    lvl_sep_big : int
        Not used; see Notes.
    lvl_sep_op : int, optional
        Scale above which the adjoint synthesis operator replaces the plain
        sum of planes. Default 2.
    deconv : bool, optional
        Whether the solver deconvolves each plane by the B3-spline kernel.
        Default True.
    verbose : bool, optional
        Passed through to the solver. Default False.

    Returns
    -------
    image : numpy.ndarray
        The synthesised object, zeroed if the solve diverged.
    det_err_image : numpy.ndarray
        Detection-error image over the same footprint, summed over scales.
    filter_kw : str
        ``'HAAR'`` or ``'BSPL'``, whichever branch was taken.
    flag_convergence : bool
        False if the solve produced NaNs.
    sum_wr : float
        Sum of the wavelet residual at the last solver iteration.
    norm_wr : float
        Norm of the wavelet residual at the last solver iteration.

    Notes
    -----
    ``lvl_sep_big`` is accepted so the signature matches the ray path and is
    not read here; the attenuation it governs is applied by the caller.
    """

    bspl = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
    haar = 1 / 2. * np.array([ 1, 0, 1 ])

    flag_convergence = True

    if (( interscale_tree.extent < extent_sep ) or (  interscale_tree.eccentricity >= ecc_sep )) & ( interscale_tree.interscale_maximum.level < lvl_sep_lin ) :
        
        image, sum_wr, norm_wr = interscale_tree.CG_minimization( wavelet_datacube, label_datacube, filter = haar, \
                                                        synthesis_operator = 'SUM', deconv = deconv, verbose = verbose )
        filter_kw = 'HAAR'

    else:
        if interscale_tree.interscale_maximum.level <= lvl_sep_op:
            image, sum_wr, norm_wr = interscale_tree.CG_minimization( wavelet_datacube, label_datacube, filter = bspl, \
                                                    synthesis_operator = 'SUM', deconv = deconv, verbose = verbose )
        else:
            image, sum_wr, norm_wr = interscale_tree.CG_minimization( wavelet_datacube, label_datacube, filter = bspl, \
                                                    synthesis_operator = 'ADJOINT', deconv = deconv, verbose = verbose )
        filter_kw = 'BSPL'

    if interscale_tree.det_err_tube(wavelet_datacube, label_datacube).shape[2] > 1:
        det_err_image = np.sum(interscale_tree.det_err_tube(wavelet_datacube, label_datacube), axis = 2)
    else:
        det_err_image = np.squeeze(interscale_tree.det_err_tube(wavelet_datacube, label_datacube), 2) # from (x, y, 1) to (x, y)    

    # Security
    if np.isnan(np.sum(image)):
        image = np.zeros( image.shape )
        flag_convergence = False

    return image, det_err_image, filter_kw, flag_convergence, sum_wr, norm_wr

def restore_objects_default(
    interscale_tree_list: list[interscale_tree_type],
    oimage: np.ndarray,
    cg_gamma: float,
    niter: int,
    wavelet_datacube: wavelet_datacube,
    label_datacube: label_datacube,
    lvl_sep_big: int,
    rm_gamma_for_big: bool,
    extent_sep: float,
    ecc_sep: float,
    lvl_sep_lin: int,
    lvl_sep_op: int,
    deconv: bool,
    size_patch_small: int = 1,
    gamma: float = 1,
    size_patch_big: int = 1,
    size_big_objects: int = 512,
    n_cpus: int = 1,
) -> list[restored_object]:
    """Synthesise a list of interscale trees into atoms.

    The restoration entry point the pipeline calls once per iteration. Two
    branches, and they do not agree.

    Below ``size_patch_big`` trees, or with ``n_cpus`` at 1, every tree goes
    through :func:`restore_object` in process. Otherwise the list is sorted by
    the area of the interscale maximum, split into big and small patches by
    ``size_big_objects``, and each patch is sent to a ray worker through
    :func:`restore_patch`. Either way the synthesised image is multiplied by
    ``gamma``, unless ``rm_gamma_for_big`` is set and the object sits at or
    above ``lvl_sep_big``.

    The two branches are not interchangeable: #5 measured a 308% flux
    difference between them, and #56 unifies them onto the published serial
    path. Until it lands, a run whose output is being compared with another
    must hold ``n_cpus`` fixed.

    Parameters
    ----------
    interscale_tree_list : list of interscale_tree
        Trees to synthesise. Sorted in place on the parallel branch.
    oimage : numpy.ndarray
        Not used; see Notes.
    cg_gamma : float
        Not used; see Notes.
    niter : int
        Not used; see Notes.
    wavelet_datacube : dawis.cubes.wavelet_datacube
        Wavelet coefficients of the residual being restored.
    label_datacube : dawis.cubes.label_datacube
        Region labels matching ``wavelet_datacube``.
    lvl_sep_big : int
        Scale at or above which an object counts as big, for
        ``rm_gamma_for_big``.
    rm_gamma_for_big : bool
        If True, do not attenuate objects at or above ``lvl_sep_big``.
    extent_sep : float
        Extent below which a tree takes the Haar branch.
    ecc_sep : float
        Eccentricity at or above which a tree takes the Haar branch.
    lvl_sep_lin : int
        Scale at or above which the Haar branch is never taken.
    lvl_sep_op : int
        Scale above which the adjoint synthesis operator replaces the sum.
    deconv : bool
        Whether the solver deconvolves each plane by the B3-spline kernel.
    size_patch_small : int, optional
        Trees per ray task for objects below ``size_big_objects``. Default 1.
    gamma : float, optional
        Attenuation applied to every synthesised image. Default 1, no
        attenuation.
    size_patch_big : int, optional
        Trees per ray task for big objects, and the list length below which
        the serial branch is taken regardless of ``n_cpus``. Default 1.
    size_big_objects : int, optional
        Side in pixels above which a tree counts as a big object for
        patching. Default 512.
    n_cpus : int, optional
        Worker count. 1 forces the serial branch. Default 1.

    Returns
    -------
    list of restored_object
        One atom per tree. In tree order on the serial branch; big objects
        first, then small, on the parallel one.

    Notes
    -----
    ``oimage``, ``cg_gamma`` and ``niter`` are accepted and not read. They are
    kept because ``dawis.synthesis_by_analysis`` passes them positionally.
    """

    if (len(interscale_tree_list) < size_patch_big) or (n_cpus == 1):

        object_list = []
        bspl = 1 / 16. * np.array([ 1, 4, 6, 4, 1 ])
        logging.info('Not parallelizing - %d Objects.'%(len(interscale_tree_list)))
        for tree in interscale_tree_list:
            image, det_err_image, filter_kw, flag_convergence, sum_wr, norm_wr = restore_object( tree, \
                                                                                  wavelet_datacube, \
                                                                                  label_datacube, \
                                                                                  extent_sep, \
                                                                                  ecc_sep, \
                                                                                  lvl_sep_lin, \
                                                                                  lvl_sep_big, \
                                                                                  lvl_sep_op, \
                                                                                  deconv, \
                                                                                  verbose = True )

            if (rm_gamma_for_big == False) or (tree.interscale_maximum.level < lvl_sep_big):
                image = image * gamma  # add attenuation factor

            object_list.append( restored_object(image, det_err_image, tree.bbox, \
                                                tree.interscale_maximum.level, \
                                                tree.interscale_maximum.eccentricity, \
                                                filter_kw, \
                                                flag_convergence, \
                                                sum_wr, \
                                                norm_wr ))

    else:
        logging.info('Size tree patch (%d) greater than %d, activating Ray store.'%(len(interscale_tree_list), size_patch_big ))
        id_wdc = ray.put(wavelet_datacube)
        id_ldc = ray.put(label_datacube)
        id_extent_sep = ray.put(extent_sep)
        id_ecc_sep = ray.put(ecc_sep)
        id_lvl_sep_lin = ray.put(lvl_sep_lin)
        id_lvl_sep_big = ray.put(lvl_sep_big)
        id_rm_gamma_for_big = ray.put(rm_gamma_for_big)
        id_gamma = ray.put(gamma)
        id_deconv = ray.put(deconv)
        id_lvl_sep_op = ray.put(lvl_sep_op)

        interscale_tree_list.sort(key = lambda x: x.interscale_maximum.area, reverse = True)

        small_interscale_tree_patch = []
        small_object_patch = []
        big_interscale_tree_patch = []
        big_object_patch = []

        for tree in interscale_tree_list:

            if tree.x_size > size_big_objects & tree.y_size > size_big_objects:
                big_interscale_tree_patch.append(tree)

                if len(big_interscale_tree_patch) >= size_patch_big:
                    big_object_patch.append( restore_patch.remote( big_interscale_tree_patch, id_wdc, id_ldc, id_extent_sep, id_ecc_sep, id_lvl_sep_lin, id_lvl_sep_big, id_lvl_sep_op, id_gamma, id_rm_gamma_for_big, id_deconv ))
                    big_interscale_tree_patch = []

            else:

                if big_interscale_tree_patch:
                        big_object_patch.append( restore_patch.remote( big_interscale_tree_patch, id_wdc, id_ldc, id_extent_sep, id_ecc_sep, id_lvl_sep_lin, id_lvl_sep_big, id_lvl_sep_op, id_gamma, id_rm_gamma_for_big, id_deconv ))

                small_interscale_tree_patch.append(tree)

                if len(small_interscale_tree_patch) >= size_patch_small:
                    small_object_patch.append( restore_patch.remote( small_interscale_tree_patch, id_wdc, id_ldc, id_extent_sep, id_ecc_sep, id_lvl_sep_lin, id_lvl_sep_big, id_lvl_sep_op, id_gamma, id_rm_gamma_for_big, id_deconv ))
                    small_interscale_tree_patch = []

        if small_interscale_tree_patch:
            small_object_patch.append( restore_patch.remote( small_interscale_tree_patch, id_wdc, id_ldc, id_extent_sep, id_ecc_sep, id_lvl_sep_lin, id_lvl_sep_big, id_lvl_sep_op, id_gamma, id_rm_gamma_for_big, id_deconv ))

        object_list = []
        for id_patch in big_object_patch:
            patch = ray.get( id_patch )
            object_list = object_list + patch
        for id_patch in small_object_patch:
            patch = ray.get( id_patch )
            object_list = object_list + patch

    return object_list
