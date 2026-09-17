"""Atom synthesis: turning interscale trees into the objects we subtract.

The step that costs the run. Detection says where the structures are and
:mod:`dawis.trees` groups them across scales; this module reconstructs each
tree as an image and wraps it as an :class:`Atom` — the *atom* of
``CONTEXT.md``. The pipeline sums the atoms of an iteration, subtracts them
from the residual, and goes round again.

The solver itself is not here. The conjugate-gradient descent lives on the
tree, as :meth:`~dawis.trees.construction.InterscaleTree.CG_minimization`,
because it works on the tubes the tree cuts out of the cubes. What this module
owns is the choice of how to call it.

:func:`restore_object` no longer makes that choice (#103). It used to route:
small or elongated structures below ``lvl_sep_lin`` took the Haar filter,
everything else the B3-spline, from the plain sum of planes at or below
``lvl_sep_op`` and from the adjoint operator above it. One arm now applies to
every tree of every iteration — the one its caller names — and
``extent_sep``, ``ecc_sep``, ``lvl_sep_lin`` and ``lvl_sep_op`` are accepted
and read nowhere. It still builds the detection-error image and traps a
diverged solve. And it *says what it ran*: the algorithm and the synthesis
operator come back with the image and are recorded on the atom (#66), so a
reader never has to re-derive them from the filter, the level and a
configuration value the atom file does not hold.

Routing off, thresholds kept (#103)
-----------------------------------

The thresholds stay in :class:`~dawis.config.Config` and route nothing. They
are kept rather than deleted because whether routing should come back is a
question #96 puts outside its destination, and the evidence that different
arms win on different classes is where that question starts. Haar was a
branch almost nothing took: ``lvl_sep_lin`` was never reached on any of #74's
222 fixtures, and #3's Tier 1 golden has one Haar atom in 76 at the published
defaults. It ships as a value of the filter axis, which a caller sets, rather
than as a branch a small tree earns.

The check that says routing is really off: two runs disagreeing about all
four thresholds are bit-identical on the restored image, the residuals and
the detection-error image. It is a whole run, so it cannot live in the suite
— it is ``.research/103-routing-off/`` on the research branch. What the suite
holds is ``tests/unit/test_routing_off.py``, which pins the same negative on
the solver's arguments over fake trees.

:func:`restore_objects_default` runs that over a list of trees. It cuts the
list into patches, hands them to :func:`dawis.parallel.pmap`, and lays the
atoms that come back out in the order the trees were given.
:func:`restore_patch` is what runs on the far side of that call, wherever
``pmap`` puts it — one call to :func:`restore_object` per tree of the patch.
Ray is named nowhere in this module since #47.

One path, whatever the worker count (#56)
-----------------------------------------

There used to be two. A worker count of 1 looped over the trees in process and
anything else patched them out, and the two did not agree: #5 measured a
**308% difference in total restored flux** between them. #20 settled which
was the science — the published conjugate-gradient path, the one the serial
branch ran — and decided the divergence ends here.

So the worker count no longer reaches any decision this module makes. It is
passed to ``pmap`` and nowhere else, and every tree is restored by the same
:func:`restore_object` call it would get on one worker. What ``n_workers``
changes is where a patch runs and how many run at once; what it cannot change
is the atom that comes back, or where that atom sits in the returned list.
That is a contract Tier 2 asserts bit-for-bit (#57), and the reason it can be
asserted that strictly is that trees are independent — no reduction runs
across them, so no worker count can reassociate a sum.

Patching is scheduling, and only scheduling. Trees wider *and* taller than
``size_big_objects`` are expensive and go into patches of ``size_patch_big``
ahead of everything else, so the long pole is submitted first; the rest
follow in patches of ``size_patch_small``. Fixing the atom order back to tree
order afterwards costs one list assignment and buys a result that does not
depend on how the work was cut up.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from dawis.config import DEFAULT_CG_MAX_ITER, SYNTHESIS_OPERATORS
from dawis.cubes import LabelCube, WaveletCube
from dawis.exceptions import DawisValueError
from dawis.parallel import cut_into_patches, pmap

log = logging.getLogger(__name__)

#: What :attr:`Atom.algorithm` holds for an atom this module reconstructed.
#: The conjugate-gradient descent is the only reconstruction DAWIS has, so it
#: is the only value a run produces; the attribute exists so that a second one
#: would be distinguishable, not because a second one is being added.
CG_ALGORITHM = 'CG'

#: The low-band à trous filters a solve can run with, by the name an atom
#: records under :attr:`Atom.filter`.
#:
#: A mapping rather than a tuple of names beside
#: :data:`~dawis.config.SYNTHESIS_OPERATORS`, because the filter axis is the
#: one whose value the solver takes as an *array*: the name has to resolve to
#: coefficients somewhere, and one place that both names and builds them is
#: what stops a caller passing a filter no atom can be labelled with. #104
#: takes the name onto :class:`~dawis.config.Config`; the arrays stay here.
FILTERS = {
    'BSPL': 1 / 16. * np.array([ 1, 4, 6, 4, 1 ]),
    'HAAR': 1 / 2. * np.array([ 1, 0, 1 ]),
}

#: What :attr:`Atom.algorithm` and :attr:`Atom.synthesis_operator` read on an
#: atom written before #66 gave them to the atom. Class attributes of
#: :class:`Atom` rather than a reader's default, because unpickling restores
#: the stored ``__dict__`` straight onto a new instance and never runs
#: ``__init__``: a name the old dictionary has no key for has to resolve on
#: the class or not at all.
UNKNOWN_PROVENANCE = 'UNKNOWN'

# Annotations only. This package imports dawis.trees for the type alone; the
# runtime import is not needed and is not taken.
if TYPE_CHECKING:
    from dawis.trees import InterscaleTree


class Atom(object):
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
    algorithm : str, optional
        Which reconstruction produced the image. :data:`CG_ALGORITHM` for
        everything DAWIS reconstructs today. Default
        :data:`UNKNOWN_PROVENANCE`.
    synthesis_operator : str, optional
        Which synthesis operator that reconstruction ran with — one of the
        words :meth:`InterscaleTree.CG_minimization
        <dawis.trees.construction.InterscaleTree.CG_minimization>` branches
        on, ``'SUM'``, ``'ADJOINT'`` or ``'TRANSPOSE'``. Default
        :data:`UNKNOWN_PROVENANCE`.
    blank : bool, optional
        If True, set no attributes at all and ignore every other argument.
        Default False.

    Notes
    -----
    ``algorithm`` and ``synthesis_operator`` are also *class* attributes,
    holding :data:`UNKNOWN_PROVENANCE`. That is what an atom written before
    #66 reads: an HDF5 group with no such dataset leaves the blank instance
    with nothing set, and a pickle restores a ``__dict__`` with no such key
    without ever calling this constructor. Either way the lookup falls through
    to the class and answers ``'UNKNOWN'`` instead of raising.
    """

    # Read by an atom whose stored state predates these two fields. See Notes.
    algorithm: str = UNKNOWN_PROVENANCE
    synthesis_operator: str = UNKNOWN_PROVENANCE

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
        algorithm: str = UNKNOWN_PROVENANCE,
        synthesis_operator: str = UNKNOWN_PROVENANCE,
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
            self.algorithm = str(algorithm)
            self.synthesis_operator = str(synthesis_operator)

def restore_patch(
    interscale_tree_patch: list[InterscaleTree],
    wavelet_cube: WaveletCube,
    label_cube: LabelCube,
    extent_sep: float,
    ecc_sep: float,
    lvl_sep_lin: int,
    lvl_sep_big: int,
    lvl_sep_op: int,
    gamma: float,
    rm_gamma_for_big: bool,
    cg_max_iter: int = DEFAULT_CG_MAX_ITER,
    verbose: bool = False,
    # Appended rather than placed beside the thresholds they replace: the
    # arguments of this function are what `restore_objects_default` passes
    # through `pmap`'s `shared` tuple, positionally, so a new one goes on the
    # end or every existing one moves.
    filter_kw: str = 'BSPL',
    synthesis_operator: str = 'SUM',
) -> list[Atom]:
    """Synthesise one patch of trees into atoms.

    The unit of work :func:`restore_objects_default` distributes: the same
    call to :func:`restore_object` per tree, the same attenuation rule, over a
    slice of the tree list. A patch is independent of every other, which is
    the whole reason restoration parallelises.

    Parameters
    ----------
    interscale_tree_patch : list of InterscaleTree
        The trees this worker is responsible for.
    wavelet_cube : dawis.cubes.WaveletCube
        Wavelet coefficients of the residual being restored.
    label_cube : dawis.cubes.LabelCube
        Region labels matching ``wavelet_cube``.
    extent_sep : float
        Passed through unread; see :func:`restore_object`.
    ecc_sep : float
        Passed through unread; see :func:`restore_object`.
    lvl_sep_lin : int
        Passed through unread; see :func:`restore_object`.
    lvl_sep_big : int
        Scale at or above which an object counts as big, for ``gamma``. Read
        here, and the only one of the five that is read anywhere.
    lvl_sep_op : int
        Passed through unread; see :func:`restore_object`.
    gamma : float
        Attenuation applied to the synthesised image.
    rm_gamma_for_big : bool
        If True, skip the attenuation for objects at or above
        ``lvl_sep_big``.
    cg_max_iter : int, optional
        Iteration cap handed to the solver. Default
        :data:`~dawis.config.DEFAULT_CG_MAX_ITER`.
    verbose : bool, optional
        Passed through to the solver, which logs a line when its flux test
        fires. Default False.
    filter_kw : {'BSPL', 'HAAR'}, optional
        The arm's filter, the same for every tree. Default ``'BSPL'``.
    synthesis_operator : {'SUM', 'ADJOINT', 'TRANSPOSE'}, optional
        The arm's synthesis operator, the same for every tree. Default
        ``'SUM'``.

    Returns
    -------
    list of Atom
        One atom per tree in the patch, in the order given.
    """

    object_patch = []
    for tree in interscale_tree_patch:

        # Unpacked into names of their own rather than back over this
        # function's `filter_kw` and `synthesis_operator`. `restore_object`
        # returns its inputs verbatim today, so rebinding them would be a
        # fixed point -- but only today. The moment it normalises a name or
        # substitutes one, the substituted arm would carry to every later
        # tree of the patch, making a tree's atom depend on its position in a
        # patch and so on `size_patch_small` and `n_workers`. That is the one
        # invariant this module's docstring promises (#56, #57).
        image, det_err_image, used_filter, flag_convergence, sum_wr, norm_wr, \
                              algorithm, used_operator = restore_object( tree, \
                                                                              wavelet_cube, \
                                                                              label_cube, \
                                                                              extent_sep, \
                                                                              ecc_sep, \
                                                                              lvl_sep_lin, \
                                                                              lvl_sep_big, \
                                                                              lvl_sep_op, \
                                                                              filter_kw = filter_kw, \
                                                                              synthesis_operator = synthesis_operator, \
                                                                              cg_max_iter = cg_max_iter, \
                                                                              verbose = verbose )

        if (rm_gamma_for_big == False) or (tree.interscale_maximum.level < lvl_sep_big):
            image = image * gamma  # add attenuation factor



        object_patch.append(Atom(image, det_err_image, tree.bbox, \
                                            tree.interscale_maximum.level, \
                                            tree.interscale_maximum.eccentricity, \
                                            used_filter, \
                                            flag_convergence, \
                                            sum_wr, \
                                            norm_wr, \
                                            algorithm, \
                                            used_operator ))

    return object_patch

def restore_object(
    interscale_tree: InterscaleTree,
    wavelet_cube: WaveletCube,
    label_cube: LabelCube,
    extent_sep: float,
    ecc_sep: float,
    lvl_sep_lin: int,
    lvl_sep_big: int,
    lvl_sep_op: int = 2,
    # Keyword-only since #101. `deconv` used to sit here, so a caller written
    # against the old signature who passed it positionally would otherwise
    # bind `cg_max_iter = False` — zero, one CG iteration, and an
    # under-converged atom returned with no error at all. The arm arrives
    # here too (#103), for the same reason: a new argument of an exported
    # function is a positional trap unless it cannot be passed positionally.
    *,
    filter_kw: str = 'BSPL',
    synthesis_operator: str = 'SUM',
    cg_max_iter: int = DEFAULT_CG_MAX_ITER,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, str, bool, float, float, str, str]:
    """Synthesise one interscale tree into an image.

    Runs :meth:`~dawis.trees.construction.InterscaleTree.CG_minimization`
    with the filter and the synthesis operator its caller names, whatever the
    tree is. Since #103 this function chooses nothing: it used to route a
    small or elongated tree below ``lvl_sep_lin`` onto Haar and to swap the
    plain sum of planes for the adjoint above ``lvl_sep_op``, and one arm now
    applies to every tree of every iteration.

    A solve that produces NaNs is not an error: the image is zeroed and
    ``flag_convergence`` comes back False.

    Parameters
    ----------
    interscale_tree : InterscaleTree
        The tree to synthesise.
    wavelet_cube : dawis.cubes.WaveletCube
        Wavelet coefficients of the residual being restored.
    label_cube : dawis.cubes.LabelCube
        Region labels matching ``wavelet_cube``.
    extent_sep : float
        Not used; see Notes.
    ecc_sep : float
        Not used; see Notes.
    lvl_sep_lin : int
        Not used; see Notes.
    lvl_sep_big : int
        Not used; see Notes.
    lvl_sep_op : int, optional
        Not used; see Notes. Default 2.
    filter_kw : {'BSPL', 'HAAR'}, optional
        Which low-band filter every tree is synthesised with —
        :data:`FILTERS`. Default ``'BSPL'``.
    synthesis_operator : {'SUM', 'ADJOINT', 'TRANSPOSE'}, optional
        Which operator every tree's first reconstruction is built with —
        :data:`~dawis.config.SYNTHESIS_OPERATORS`, checked by
        ``CG_minimization`` itself. Default ``'SUM'``.
    cg_max_iter : int, optional
        Iteration cap handed to the solver. Default
        :data:`~dawis.config.DEFAULT_CG_MAX_ITER`, the published 200 — see
        that constant for why it is a knob and who owns it.
    verbose : bool, optional
        Passed through to the solver. Default False.

    Returns
    -------
    image : numpy.ndarray
        The synthesised object, zeroed if the solve diverged.
    det_err_image : numpy.ndarray
        Detection-error image over the same footprint, summed over scales.
    filter_kw : str
        The filter this call was given, back out so the atom can record it.
    flag_convergence : bool
        False if the solve produced NaNs.
    sum_wr : float
        Sum of the wavelet residual at the last solver iteration.
    norm_wr : float
        Norm of the wavelet residual at the last solver iteration.
    algorithm : str
        :data:`CG_ALGORITHM`, the one reconstruction this function calls.
    synthesis_operator : str
        The operator this call was given, back out for the same reason.

    Notes
    -----
    ``extent_sep``, ``ecc_sep``, ``lvl_sep_lin`` and ``lvl_sep_op`` are the
    routing rule, and there is none (#103). They are accepted so that the
    thresholds can stay in :class:`~dawis.config.Config` routing nothing, and
    so that removing them from the middle of an exported signature does not
    become the silent-wrong-answer hazard #101 met. ``lvl_sep_big`` is
    accepted for the older reason: the signature matches :func:`restore_patch`,
    and the attenuation it governs is applied by the caller.

    Reading none of the five is what makes routing testable as *off*:
    ``tests/unit/test_routing_off.py`` inverts each threshold and asserts the
    solver was called identically. The run-level version of that check — two
    whole runs, bit-identical on every image — is a comparison against #93's
    seam and lives in ``.research/103-routing-off/``, not in the suite.

    The last two values are returned rather than left to be re-derived (#66).
    The derivation they replace — Haar, or B3-spline at or below
    ``lvl_sep_op``, took the sum — was folklore even while it was correct: it
    needed ``lvl_sep_op``, which no atom file records, and it stopped being
    correct the moment routing went off. The return grew by two values
    rather than changing into an object, because the attenuation factor is
    applied between this call and the atom's construction; folding the atom
    into this function would drag the attenuation parameters down with it.
    That restructuring belongs with the minimizer interface work.
    """

    # The routing rule, read here and nowhere else before #103 removed it.
    # Deleted rather than left bound, so that a branch reintroduced by
    # accident is a NameError rather than a quiet return to routing.
    del extent_sep, ecc_sep, lvl_sep_lin, lvl_sep_big, lvl_sep_op

    # Both axes are refused here, before `_schedule_patches` and `pmap`.
    # `CG_minimization` checks the operator too (#108) and that check stays,
    # but it fires from inside the first solve -- on a worker process under
    # any backend but the serial one, where the traceback lands in the
    # backend's log rather than the run's. The filter has no second checker
    # at all: this is where the name becomes an array, so a name outside
    # `FILTERS` would otherwise surface as a `KeyError` mid-solve.
    if filter_kw not in FILTERS:
        raise DawisValueError(
            'filter_kw names the low-band filter every tree of the run is '
            'synthesised with -->', filter_kw, list(FILTERS))

    if synthesis_operator not in SYNTHESIS_OPERATORS:
        raise DawisValueError(
            'synthesis_operator names the operator that rebuilds an image '
            'from a tube -->', synthesis_operator, list(SYNTHESIS_OPERATORS))

    flag_convergence = True

    image, sum_wr, norm_wr = interscale_tree.CG_minimization( wavelet_cube, label_cube, filter = FILTERS[filter_kw], \
                                                    synthesis_operator = synthesis_operator, \
                                                    max_iter = cg_max_iter, verbose = verbose )

    if interscale_tree.det_err_tube(wavelet_cube, label_cube).shape[2] > 1:
        det_err_image = np.sum(interscale_tree.det_err_tube(wavelet_cube, label_cube), axis = 2)
    else:
        det_err_image = np.squeeze(interscale_tree.det_err_tube(wavelet_cube, label_cube), 2) # from (x, y, 1) to (x, y)    

    # Security
    if np.isnan(np.sum(image)):
        image = np.zeros( image.shape )
        flag_convergence = False

    return (image, det_err_image, filter_kw, flag_convergence, sum_wr, norm_wr,
            CG_ALGORITHM, synthesis_operator)


def _schedule_patches(
    interscale_tree_list: list[InterscaleTree],
    size_patch_small: int,
    size_patch_big: int,
    size_big_objects: int,
) -> list[list[int]]:
    """Decide which trees are worked together, and in which order.

    Pure scheduling: every tree lands in exactly one patch whatever the
    numbers say, so nothing here can change what is restored — only how the
    work is cut up and which end of it is submitted first. A tree bigger than
    ``size_big_objects`` on *both* axes costs disproportionately more to
    solve, so those go first, in patches of their own size; the rest follow.
    Submitting the long pole first is what keeps the last worker from being
    handed it.

    Returns positions rather than trees, so that the caller can put the atoms
    back in tree order once the patches come home. That reordering is the
    whole reason this is allowed to schedule freely.

    Parameters
    ----------
    interscale_tree_list : list of InterscaleTree
        Trees of this iteration, in the order the caller gave them.
    size_patch_small, size_patch_big : int
        Trees per patch, for ordinary and for big objects.
    size_big_objects : int
        Side in pixels a tree must exceed on both axes to count as big.

    Returns
    -------
    list of list of int
        Patches of positions into ``interscale_tree_list``, big first.

    Notes
    -----
    The pre-#56 version of this had three defects, all of them here rather
    than in the science, and all three are what the tests in
    ``tests/unit/test_restoration_unification.py`` pin. It wrote the size test
    as ``x_size > size & y_size > size``, which Python reads as the chained
    comparison ``x_size > (size & y_size) > size`` and not as the conjunction
    it looks like. It appended the big accumulator to the patch list without
    clearing it, so a big tree followed by *n* small ones was restored *n*
    times over. And it flushed the small accumulator at the end of the list
    but not the big one, so a big tree in the tail was dropped. The first
    inflated big patches, the second duplicated atoms and the third lost
    them — between them a large part of the flux difference #5 measured
    against the serial branch.
    """

    big = []
    small = []
    for position, tree in enumerate(interscale_tree_list):
        if tree.x_size > size_big_objects and tree.y_size > size_big_objects:
            big.append(position)
        else:
            small.append(position)

    return (cut_into_patches(big, size_patch_big)
            + cut_into_patches(small, size_patch_small))


def restore_objects_default(
    interscale_tree_list: list[InterscaleTree],
    oimage: np.ndarray,
    cg_gamma: float,
    niter: int,
    wavelet_cube: WaveletCube,
    label_cube: LabelCube,
    lvl_sep_big: int,
    rm_gamma_for_big: bool,
    extent_sep: float,
    ecc_sep: float,
    lvl_sep_lin: int,
    lvl_sep_op: int,
    size_patch_small: int = 1,
    gamma: float = 1,
    size_patch_big: int = 1,
    size_big_objects: int = 512,
    n_workers: int = 1,
    cg_max_iter: int = DEFAULT_CG_MAX_ITER,
    verbose: bool = False,
    filter_kw: str = 'BSPL',
    synthesis_operator: str = 'SUM',
) -> list[Atom]:
    """Synthesise a list of interscale trees into atoms.

    The restoration entry point the pipeline calls once per iteration, and
    since #56 there is one of it rather than two. The trees are cut into
    patches by :func:`_schedule_patches`, the patches go to
    :func:`dawis.parallel.pmap` as calls to :func:`restore_patch`, and the
    atoms are laid back out in the order the trees were given. Each
    synthesised image is multiplied by ``gamma``, unless ``rm_gamma_for_big``
    is set and the object sits at or above ``lvl_sep_big``.

    ``n_workers`` reaches ``pmap`` and nothing else. It decides how many patches
    run at once and no part of what any of them computes: the atom returned
    for a tree, and its place in the returned list, are the same on one worker
    and on forty. See *One path, whatever the worker count* in the module
    docstring for why that is worth stating, and #57 for the test that holds
    it to the bit.

    Parameters
    ----------
    interscale_tree_list : list of InterscaleTree
        Trees to synthesise. Read, never reordered.
    oimage : numpy.ndarray
        Not used; see Notes.
    cg_gamma : float
        Not used; see Notes.
    niter : int
        Not used; see Notes.
    wavelet_cube : dawis.cubes.WaveletCube
        Wavelet coefficients of the residual being restored.
    label_cube : dawis.cubes.LabelCube
        Region labels matching ``wavelet_cube``.
    lvl_sep_big : int
        Scale at or above which an object counts as big, for
        ``rm_gamma_for_big``.
    rm_gamma_for_big : bool
        If True, do not attenuate objects at or above ``lvl_sep_big``.
    extent_sep : float
        Passed through unread; see :func:`restore_object`.
    ecc_sep : float
        Passed through unread; see :func:`restore_object`.
    lvl_sep_lin : int
        Passed through unread; see :func:`restore_object`.
    lvl_sep_op : int
        Passed through unread; see :func:`restore_object`.
    size_patch_small : int, optional
        Trees per patch for objects below ``size_big_objects``. Default 1.
    gamma : float, optional
        Attenuation applied to every synthesised image. Default 1, no
        attenuation.
    size_patch_big : int, optional
        Trees per patch for big objects. Default 1.
    size_big_objects : int, optional
        Side in pixels a tree must exceed on both axes to count as a big
        object for patching. Default 512.
    n_workers : int, optional
        Worker count, passed straight to :func:`dawis.parallel.pmap`.
        Scheduling only. Default 1.
    cg_max_iter : int, optional
        Iteration cap handed to the solver. Default
        :data:`~dawis.config.DEFAULT_CG_MAX_ITER`, the published 200.
    verbose : bool, optional
        Passed through to the solver, which logs a line per atom when its
        flux test fires. Default False. The pipeline passes True, which is
        what the serial branch always did; the pre-#56 parallel branch could
        not, so a distributed run now writes those lines too — into the
        backend's own worker logs rather than the run directory's.
    filter_kw : {'BSPL', 'HAAR'}, optional
        The arm's filter, applied to every tree of the list. Default
        ``'BSPL'``.
    synthesis_operator : {'SUM', 'ADJOINT', 'TRANSPOSE'}, optional
        The arm's synthesis operator, applied to every tree of the list.
        Default ``'SUM'``. #104 is what carries a :class:`~dawis.config.Config`
        value here; until then the pipeline takes this default.

    Returns
    -------
    list of Atom
        One atom per tree, in the order the trees were given, whatever the
        worker count.

    Notes
    -----
    ``oimage``, ``cg_gamma`` and ``niter`` are accepted and not read. They are
    kept because the pipeline passes them positionally.
    """

    patches = _schedule_patches(interscale_tree_list, size_patch_small,
                                size_patch_big, size_big_objects)
    log.info('Restoring %d interscale trees in %d patches on %d worker(s).',
             len(interscale_tree_list), len(patches), n_workers)

    # Each patch is built as its own list, never as a view into the caller's.
    # That is what `.remote()` used to do implicitly by serialising its
    # argument at the call; a backend that runs a patch in this process sees
    # live tree objects where ray saw a deep copy, and nothing enforces the
    # difference, so `restore_patch` must go on treating what it is handed as
    # read-only.
    atom_patches = pmap(
        restore_patch,
        [[interscale_tree_list[position] for position in patch]
         for patch in patches],
        shared = (wavelet_cube, label_cube, extent_sep, ecc_sep,
                  lvl_sep_lin, lvl_sep_big, lvl_sep_op, gamma,
                  rm_gamma_for_big, cg_max_iter, verbose, filter_kw,
                  synthesis_operator),
        n_workers = n_workers,
    )

    # Patch order is a scheduling decision -- big objects first -- and it
    # stops here. The atoms go back where their trees were, so the returned
    # list reads the same whatever the patching did.
    by_position: dict[int, Atom] = {}
    for patch, atom_patch in zip(patches, atom_patches, strict = True):
        by_position.update(zip(patch, atom_patch, strict = True))

    return [by_position[position]
            for position in range(len(interscale_tree_list))]
