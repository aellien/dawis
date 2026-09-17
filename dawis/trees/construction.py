"""Interscale trees: what a detected structure is before it is restored.

A :class:`InterscaleTree` is one structure seen at every scale at once. It
groups the regions that :mod:`dawis.detection` found on the separate wavelet
planes into a single object, rooted at the brightest of them — the *interscale
maximum*. Which regions join which root is decided in
:mod:`dawis.trees.deblending`; this module holds what a tree *is* once that
decision is made.

Three things live on the class.

The **geometry** is fixed in ``__init__``: the bounding box of the member
regions, widened by half and floored at ``clip`` pixels so the synthesis has
room to work, plus the extent and eccentricity that the restoration reads to
choose its branch.

The **tubes** are the tree cut out of the cubes. :meth:`~InterscaleTree.tubes`
returns the wavelet coefficients, the binary support and the labels of the
member regions, stacked over the scales the tree spans and clipped to its
bounding box. Everything downstream works on tubes rather than on full planes.

The **synthesis** turns a tube back into an image.
:meth:`~InterscaleTree.CG_minimization` is the published solver of
Ellien+2021: it runs a conjugate-gradient descent that looks for the image
whose wavelet transform matches the tube, starting from either of two
synthesis operators — :meth:`~InterscaleTree.synthesis_adjoint`, the
transposed transform, or :meth:`~InterscaleTree.synthesis_sum`, the plain sum
of planes. Its output is the atom.
"""

from __future__ import annotations

import logging

import numpy as np

from dawis.exceptions import DawisValueError
from dawis.config import (
    CG_LINE_DIRECTIONS,
    CG_STEP_SIZES,
    DEFAULT_CG_MAX_ITER,
    SYNTHESIS_OPERATORS,
)
from dawis.cubes import CoarseCube, LabelCube, WaveletCube
from dawis.detection import region
from dawis.wavelets import atrous, atrous_adjoint

log = logging.getLogger(__name__)

# ``InterscaleTree.plot`` moved to
# :func:`dawis.viz.trees.plot_interscale_tree` with #46; nothing outside
# :mod:`dawis.viz` imports matplotlib.


def _analysis_tube(
    image: np.ndarray,
    n_levels: int,
    filter: np.ndarray,
    coarse: bool,
) -> np.ndarray:
    r"""The forward transform of an iterate, shaped like the tube it is
    compared against.

    The wavelet planes, and — when the tube carries the coarse term — the
    coarse channel :math:`c_{L+1}` behind them.

    The channel costs **no extra transform**. :func:`~dawis.wavelets.atrous`
    computes the smooth planes on its way to the wavelet ones and returns
    both, and plane ``n_levels`` of them *is* the image smoothed ``n_levels``
    times.

    It is the same *operator* as the one the tube's channel was cut with, at
    the same depth, but **not the same boundary rule**: the cut comes from
    :func:`~dawis.wavelets.ser_iuwt_decomposition`, which mirrors, and this
    comes from :func:`~dawis.wavelets.atrous` under ``'prolongation'``, which
    clamps. The two agree to 4e-6 in the interior and by up to 9% within
    ``2 * 2 ** level`` of an edge. The published solve already compares its
    wavelet planes across that same mismatch, so this is inherited rather
    than introduced — but a tree box at ``level >= 3`` is smaller than that
    footprint, so for a deep tree the whole channel is boundary, and the
    channel carries the smooth term's amplitude rather than a detail's.
    Carried on #96 as Drift, not fixed here: closing it moves every published
    wavelet plane too.

    Parameters
    ----------
    image : numpy.ndarray
        2D, the iterate to analyse.
    n_levels : int
        Number of **wavelet** planes — the channel is not counted.
    filter : numpy.ndarray
        1D low-band filter.
    coarse : bool
        Whether to append the coarse channel.

    Returns
    -------
    numpy.ndarray
        3D, ``(rows, columns, n_levels)`` — ``n_levels + 1`` with ``coarse``.
    """

    smooth, wavelets = atrous(image, n_levels = n_levels, filter = filter,
                              conditions = 'prolongation')

    if not coarse:
        return wavelets

    return np.concatenate(
        (wavelets, smooth[:, :, n_levels][:, :, None]), axis = 2)


class InterscaleTree(object):
    """One structure, grouped across the wavelet planes it appears on.

    A tree owns a list of :class:`~dawis.detection.regions.region` objects
    found on adjacent scales and connected to a root region, the interscale
    maximum. It carries the bounding box that contains all of them and the
    shape descriptors the restoration branches on. It is the unit handed to
    restoration, which synthesises it into an atom.

    Parameters
    ----------
    interscale_maximum : dawis.detection.regions.region, optional
        The root region of the tree, the brightest of its members. Its level
        is the coarsest scale the tree spans.
    region_list : list of dawis.detection.regions.region, optional
        The connected regions, interscale maximum included.
    wavelet_cube : dawis.cubes.WaveletCube, optional
        Wavelet planes the regions were measured on. Read once, to size the
        extent; not stored.
    label_cube : dawis.cubes.LabelCube, optional
        Label cube the regions were labelled in. Read once, for the cube
        dimensions and the extent; not stored.
    det_err_array : numpy.ndarray, optional
        Per-scale detection uncertainty, one value per plane, as returned by
        :func:`~dawis.detection.deblending.ms_detect_and_deblend`.
    clip : int, optional
        Minimum side of the bounding box, in pixels. A tree smaller than this
        is grown around its centre. Default is 30.
    blank : bool, optional
        Build an empty instance and set no attribute at all. Default is False.
        Used by the object store to rehydrate a tree attribute by attribute.

    Attributes
    ----------
    bbox : tuple of int
        ``(x_min, y_min, x_max, y_max)`` of the tree in image pixels, already
        widened and clipped.
    x_size, y_size : int
        Side lengths of ``bbox``.
    n_regions : int
        Number of member regions.
    span_levels : int
        Number of distinct scales the members sit on.
    extent : float
        Fraction of the *unwidened* bounding box covered by the union of the
        member regions. The restoration reads it to tell compact structures
        from sprawling ones.
    eccentricity : float
        Eccentricity of the interscale maximum, carried up for the same
        reason.
    lowb, highb : float
        Parameter bounds kept for the Bayesian minimizer, which is not part
        of the published path.

    Notes
    -----
    The bounding box is computed twice. The first pass fixes the box that
    ``extent`` is measured against; the second widens it by half its size in
    each direction, floors it at ``clip`` and clamps it to the cube, and is
    the one stored.
    """

    def __init__(
        self,
        interscale_maximum: region | None = None,
        region_list: list[region] | None = None,
        wavelet_cube: WaveletCube | None = None,
        label_cube: LabelCube | None = None,
        det_err_array: np.ndarray | None = None,
        clip: int = 30,
        blank: bool = False,
    ) -> None:

        if blank == False:
             
            self.region_list = region_list
            self.interscale_maximum = interscale_maximum
            self.n_regions = len(region_list)
            self.span_levels = np.size(np.unique([ x.level for x in region_list ]))
            self.clip = clip
            self.eccentricity = self.interscale_maximum.eccentricity
            self.det_err_array = det_err_array
            
            x_min = np.min([ x.bbox[0] for x in self.region_list ])
            y_min = np.min([ x.bbox[1] for x in self.region_list ])
            x_max = np.max([ x.bbox[2] for x in self.region_list ])
            y_max = np.max([ x.bbox[3] for x in self.region_list ])

            # Extent
            clip = np.copy(label_cube.array[ x_min:x_max, \
                                                y_min:y_max, \
                                                    : ])
            self.bbox = (x_min, y_min, x_max, y_max)
            extent_im = np.sum( self.tubes( wavelet_cube, label_cube )[1], axis = 2 )
            n_pix = np.size( np.where( extent_im != 0. )[0] )
            self.extent = n_pix / ( ( x_max - x_min ) * ( y_max - y_min ) )

            # Make vignet larger
            x_min -= int( ( x_max - x_min ) / 2. )
            x_max += int( ( x_max - x_min ) / 2. )
            if x_min < 0: x_min = 0
            if x_max > label_cube.x_size: x_max = label_cube.x_size

            y_min -= int( ( y_max - y_min ) / 2. )
            y_max += int( ( y_max - y_min ) / 2. )
            if y_min < 0: y_min = 0
            if y_max > label_cube.y_size: y_max = label_cube.y_size

            # Clip
            if x_max - x_min < self.clip:
                x_min = int( (x_max + x_min) / 2. - self.clip / 2. + 1.)   # we resize it to the minclip
                x_max = int( (x_max + x_min) / 2. + self.clip / 2. + 1.)   # value.
                if x_min < 0: x_min = 0
                if x_max > label_cube.x_size: x_max = label_cube.x_size

            if y_max - y_min < self.clip:
                y_min = int( (y_max + y_min) / 2. - self.clip / 2. + 1.)   # we resize it to the minclip
                y_max = int( (y_max + y_min) / 2. + self.clip / 2. + 1.)   # value.
                if y_min < 0: y_min = 0
                if y_max > label_cube.y_size: y_max = label_cube.y_size

            self.bbox = (x_min, y_min, x_max, y_max)
            self.x_size = x_max - x_min
            self.y_size = y_max - y_min

            # Bayesian parameter bounds
            self.lowb = 0.
            self.highb = 2.
        
    def det_err_tube(
        self,
        wavelet_cube: WaveletCube,
        label_cube: LabelCube,
    ) -> np.ndarray:
        """Build the detection uncertainty of the tree, plane by plane.

        The support tube says which pixels of the tree are significant at each
        scale; each plane of it is multiplied by that scale's entry of
        ``det_err_array``.

        Parameters
        ----------
        wavelet_cube : dawis.cubes.WaveletCube
            Wavelet planes the tree was built on.
        label_cube : dawis.cubes.LabelCube
            Label cube the tree was built on.

        Returns
        -------
        numpy.ndarray
            Cube of shape ``(x_size, y_size, level + 1)``, zero outside the
            tree and equal to the per-scale uncertainty inside it. The
            restoration sums it into the uncertainty map of the atom.
        """

        wavelet_tube, support_tube, label_tube = self.tubes(wavelet_cube, label_cube)
        det_err_tube = np.copy(support_tube)
        
        for level in range(0, det_err_tube.shape[2]):
                det_err_tube[:,:,level] *= self.det_err_array[level]
        
        return det_err_tube
            
    
    def tubes(
        self,
        wavelet_cube: WaveletCube,
        label_cube: LabelCube,
        coarse_cube: CoarseCube | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Cut the tree out of the cubes.

        Only the pixels of the member regions are copied, each onto the plane
        of its own scale, into arrays clipped to the bounding box of the tree.
        Everything else is zero.

        Handed a coarse cube, the tube gains the tree's **coarse term** as one
        extra channel behind its wavelet planes, and the support gains a
        repeat of its own plane ``L`` behind them.

        Parameters
        ----------
        wavelet_cube : dawis.cubes.WaveletCube
            Wavelet planes to read the coefficients from.
        label_cube : dawis.cubes.LabelCube
            Label cube to read the labels from.
        coarse_cube : dawis.cubes.CoarseCube, optional
            Smooth planes of the same decomposition, as
            :func:`~dawis.wavelets.transforms.bspl_atrous` returns them with
            ``coarse = True``. Plane ``j`` is the image smoothed ``j`` times,
            so the term this tree takes is plane ``L + 1``. It must carry one
            plane more than the wavelet cube and the same
            ``wavelet_type``; both are checked, because both mistakes read a
            wrong number rather than raising. Default None, which cuts the
            tube every result before #99 was taken on.

        Raises
        ------
        dawis.exceptions.DawisValueError
            If the coarse cube is not one plane deeper than the wavelet cube,
            or was built on a different basis.

        Returns
        -------
        wavelet_tube : numpy.ndarray
            Wavelet coefficients of the tree, shape
            ``(x_size, y_size, level + 1)`` — ``(x_size, y_size, level + 2)``
            when a coarse cube is given, the channel last.
        support_tube : numpy.ndarray
            The same cube as a binary mask — 1 on a member pixel, 0 elsewhere.
            The same shape as the wavelet tube.
        label_tube : numpy.ndarray
            The same cube carrying region labels instead of coefficients.
            **Not** widened: the channel is a smooth remainder, and no region
            was ever detected on it.

        Notes
        -----
        The tube spans every scale from 0 up to the level of the interscale
        maximum, including scales on which the tree has no region. Those
        planes stay zero.

        The coarse term is :math:`c_{L+1}` **masked by the tree's support at
        level L** (#88). The unmasked plane over a tree's box is sky,
        intracluster light and neighbours; DAWIS subtracts every atom from
        the field, so an atom carrying that flux removes the light the
        pipeline exists to keep. The mask is repeated onto the support rather
        than applied once here, because the solve restricts its residual to
        the support at every iteration and the channel has to keep obeying
        the same constraint.

        It is not the last plane of the coarse cube, which is the smooth
        remainder of the *full* decomposition and would leave every scale
        between :math:`L` and the last out of the problem. Nor is it that
        plane plus the wavelet planes above :math:`L`: that identity needs a
        telescoping transform and the pipeline's is not one — the detail is
        :math:`w_j = c_j - H_j^2 c_j` while the chain advances as
        :math:`c_{j+1} = H_j c_j` (#89, the sum misses by 15.8-21.0%).
        """

        label_tube = np.zeros((self.bbox[2] - self.bbox[0], \
                               self.bbox[3] - self.bbox[1], \
                               self.interscale_maximum.level + 1))

        wavelet_tube = np.copy(label_tube)

        for region in self.region_list:

            wavelet_tube[ region.coords[:,0] - self.bbox[0], \
                          region.coords[:,1] - self.bbox[1], \
                          region.level] = wavelet_cube.array[ region.coords[:,0], \
                                                          region.coords[:,1], \
                                                          region.level ]

            label_tube[ region.coords[:,0] - self.bbox[0], \
                        region.coords[:,1] - self.bbox[1], \
                        region.level] = label_cube.array[ region.coords[:,0], \
                                                              region.coords[:,1], \
                                                              region.level ]

        support_tube = np.copy(label_tube)
        support_tube[support_tube != 0] = 1.

        if coarse_cube is not None:
            if coarse_cube.z_size != wavelet_cube.z_size + 1:
                raise DawisValueError(
                    'Coarse cube depth against a %d-plane wavelet cube. This '
                    'method reads plane L + 1 of a cascade that starts at the '
                    'unsmoothed image, so it needs one plane more. A cube of '
                    'the same depth is the other live convention, in which '
                    'plane L is already the term this one calls plane L + 1 '
                    '-- read with this convention, every tree but the deepest '
                    'would take a coarse term one scale too smooth, and none '
                    'of them would raise -->'
                    % wavelet_cube.z_size,
                    coarse_cube.z_size, wavelet_cube.z_size + 1)

            if coarse_cube.wavelet_type != wavelet_cube.wavelet_type:
                raise DawisValueError(
                    'Coarse cube basis. The channel is the datum the solve '
                    'compares its forward model against, and the solve '
                    'smooths with the basis it was handed, so the two have '
                    'to be one basis -->',
                    coarse_cube.wavelet_type, wavelet_cube.wavelet_type)

            level = self.interscale_maximum.level
            support_plane = support_tube[:, :, level]
            smooth = coarse_cube.array[self.bbox[0]:self.bbox[2],
                                       self.bbox[1]:self.bbox[3],
                                       level + 1]

            wavelet_tube = np.concatenate(
                (wavelet_tube, (smooth * support_plane)[:, :, None]), axis = 2)
            support_tube = np.concatenate(
                (support_tube, support_plane[:, :, None]), axis = 2)

        return wavelet_tube, support_tube, label_tube

    def synthesis_adjoint(
        self,
        wavelet_tube: np.ndarray,
        filter: np.ndarray,
        coarse: bool = False,
    ) -> np.ndarray:
        r"""Reconstruct an image from a tube with the adjoint operator.

        The tilde operator :math:`\hat{A}` associated with the wavelet
        transform :math:`W`:

        .. math:: \hat{A}(W) = \sum_i H_i * \dots * H_{N-1} * W_i

        with :math:`H` the low-band filter of the à trous algorithm in its
        smooth-only version. Each plane of the tube is smoothed as many times
        as its scale index, and the results are summed.

        Parameters
        ----------
        wavelet_tube : numpy.ndarray
            Wavelet coefficients of a tree, as returned by :meth:`tubes`.
        filter : numpy.ndarray
            1D low-band filter, normally the B3-spline kernel.
        coarse : bool, optional
            Whether the last channel of the tube is the coarse term (#89).
            It is smoothed :math:`L + 1` times, by the same filters that
            built it — mirroring
            :func:`~dawis.wavelets.atrous_adjoint`'s coarse term
            :math:`H_{0}^{\mathsf{T}} \cdots H_{L}^{\mathsf{T}}` with the
            filters the right way round, rather than inventing an
            :math:`H_{L+1}` that appears nowhere in the transform. Default
            False.

        Returns
        -------
        numpy.ndarray
            2D image of the shape of one tube plane, the first approximate
            reconstruction the conjugate gradient starts from.

        See Also
        --------
        synthesis_sum : The cheaper alternative, a plain sum of the planes.
        synthesis_transpose : The genuine transpose of the transform.
        """

        n_levels = wavelet_tube.shape[2] - int(coarse)
        smooth_coeff = np.zeros(wavelet_tube.shape)

        for level in range(n_levels):

            smooth_coeff[:, :, level] = atrous( wavelet_tube[:, :, level], \
                                               n_levels = level + 1, \
                                               filter = filter, \
                                               conditions = 'prolongation' )[0][:, :, level]

        if coarse:
            smooth_coeff[:, :, n_levels] = atrous( wavelet_tube[:, :, n_levels], \
                                                  n_levels = n_levels, \
                                                  filter = filter, \
                                                  conditions = 'prolongation' )[0][:, :, n_levels]

        return np.sum( smooth_coeff, axis = 2 )

    def synthesis_sum(self, wavelet_tube: np.ndarray) -> np.ndarray:
        """Reconstruct an image from a tube by summing its planes.

        Parameters
        ----------
        wavelet_tube : numpy.ndarray
            Wavelet coefficients of a tree, as returned by :meth:`tubes`.

        Returns
        -------
        numpy.ndarray
            2D image of the shape of one tube plane.

        See Also
        --------
        synthesis_adjoint : The cascaded-smoothing alternative.
        synthesis_transpose : The genuine transpose of the transform.

        Notes
        -----
        This operator takes no ``coarse`` flag, and needs none: the coarse
        term is a plane of the tube and this operator adds the planes (#89).
        """
        return np.sum( wavelet_tube, axis = 2 )

    def synthesis_transpose(
        self,
        wavelet_tube: np.ndarray,
        filter: np.ndarray,
        coarse: bool = False,
    ) -> np.ndarray:
        r"""Reconstruct an image from a tube with the true adjoint operator.

        The transpose :math:`W^{\mathsf{T}}` of the à trous transform the tube
        holds, and a third operator rather than a renaming of either published
        one. :meth:`synthesis_sum` adds the planes, which is the starlet's
        *reconstruction* dual; :meth:`synthesis_adjoint` cascades each plane
        through the low-pass filters, which is what the published lineage
        adopted *after* the true adjoint produced ring artefacts ([LU3CW9P8]
        p.272, #76) — a rejection #85 measured on this instrument and did not
        reproduce.

        The tree is read for nothing but the tube it is handed, so the
        arithmetic lives with the transform it transposes, in
        :func:`~dawis.wavelets.atrous_adjoint`.

        Parameters
        ----------
        wavelet_tube : numpy.ndarray
            Wavelet coefficients of a tree, as returned by :meth:`tubes`.
        filter : numpy.ndarray
            1D low-band filter, normally the B3-spline kernel.
        coarse : bool, optional
            Whether the last channel of the tube is the coarse term (#89).
            The transpose then includes the coarse row of the transform, at
            the cost of one extra smoothing transpose. Default False.

        Returns
        -------
        numpy.ndarray
            2D image of the shape of one tube plane.

        See Also
        --------
        dawis.wavelets.atrous_adjoint : The transpose itself, and the
            inner-product identity that pins it.
        synthesis_adjoint : The cascaded-smoothing alternative.
        synthesis_sum : The cheaper alternative, a plain sum of the planes.
        """
        return atrous_adjoint( wavelet_tube, filter, coarse = coarse )

    def CG_minimization(
        self,
        wavelet_cube: WaveletCube,
        label_cube: LabelCube,
        filter: np.ndarray,
        synthesis_operator: str = 'ADJOINT',
        step_size: str = 'FR',
        line_direction: str = 'as_written',
        sigma_flux: float = 1E-3,
        max_iter: int = DEFAULT_CG_MAX_ITER,
        verbose: bool = False,
        coarse_cube: CoarseCube | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Synthesise the tree into an image by conjugate gradient descent.

        Modified from Starck, Murtagh & Bijaoui (1998), and the published path
        of Ellien+2021. A first reconstruction is made with the chosen
        synthesis operator, then refined: at each step the tube is compared
        with the wavelet transform of the current image, restricted to the
        support of the tree, and the image is moved along a conjugate
        direction by the ratio of the two residual norms. The image is
        clipped at zero after every step, and the loop stops when the total
        flux changes by less than ``sigma_flux`` relatively, or when
        ``max_iter`` steps have been taken.

        Parameters
        ----------
        wavelet_cube : dawis.cubes.WaveletCube
            Wavelet planes the tree was built on.
        label_cube : dawis.cubes.LabelCube
            Label cube the tree was built on.
        filter : numpy.ndarray
            1D low-band filter of the à trous transform, normally B3-spline.
        synthesis_operator : {'ADJOINT', 'SUM', 'TRANSPOSE'}, optional
            Which operator rebuilds an image from a tube —
            :data:`~dawis.config.SYNTHESIS_OPERATORS`. Default is
            ``'ADJOINT'``. ``'TRANSPOSE'`` is the genuine transpose of the
            transform (#84, #98) — see :meth:`synthesis_transpose`.
        step_size : {'FR', 'PRP', 'HS', 'CD'}, optional
            Formula for the conjugate direction coefficient — Fletcher-Reeves,
            Polak-Ribière-Polyak, Hestenes-Stiefel or conjugate descent —
            :data:`~dawis.config.CG_STEP_SIZES`. Default is ``'FR'``, the
            published choice, which is what a run still gets:
            :attr:`dawis.config.Config.step_size` defaults to the ``'HS'``
            #82 chose, and **nothing carries it here yet** — #104 wires all
            six arm axes at once, and until it does this argument is set by
            its callers or not at all.
        line_direction : {'as_written', 'textbook'}, optional
            Which reading of the conjugate direction recursion to take.
            ``'as_written'``, the default, is ``v = fr + beta * old_fr`` —
            what the published pseudocode prints and what DAWIS has always
            run. ``'textbook'`` is ``v = fr + beta * v``, what both published
            CG algorithms on this problem do (#76). They agree on the first
            update, because ``v`` starts as a copy of ``fr``, and part
            company on the second. #74 swept both and #82 ruled the
            as-written form a **documented variant, not a defect** — the
            textbook form is better on neither metric on any class. Set here
            or not at all until #104, like ``step_size`` above.
        sigma_flux : float, optional
            Relative flux change below which the descent stops. Default is
            1E-3.
        max_iter : int, optional
            Iteration cap. Default :data:`~dawis.config.DEFAULT_CG_MAX_ITER`,
            the published 200, at which the ``sigma_flux`` test does the
            stopping rather than the cap. `7c5ba8e` cut it to 3 in 2024 and
            #32 reverted that drift; **#26 owns the question of what it should
            be** and sweeps it. A run sets it through
            :attr:`dawis.config.Config.cg_max_iter`, which reaches here
            through :func:`~dawis.restoration.synthesis.restore_object`.
        verbose : bool, optional
            Print the convergence line when the flux test fires. Default is
            False.

        Other Parameters
        ----------------
        coarse_cube : dawis.cubes.CoarseCube, optional
            Smooth planes of the same decomposition. Given one, the tube
            carries the tree's coarse term :math:`c_{L+1}` as a last channel
            and the solve reconstructs it alongside the wavelet planes
            (#88, #89). Default None, which solves the published problem
            every pre-#99 result was taken on. Last in the signature rather
            than beside the other axes, so a positional caller of the
            published method keeps binding what it always bound.

        Raises
        ------
        dawis.exceptions.DawisValueError
            If ``synthesis_operator``, ``step_size`` or ``line_direction``
            names a branch the solve does not have, or if the coarse cube is
            not one plane deeper than the wavelet cube — see :meth:`tubes`.

        Returns
        -------
        reconstructed : numpy.ndarray
            The synthesised image of the tree, clipped at zero — the atom.
        sum_wr : float
            Sum of the final wavelet residual over the support of the tree.
        norm_wr : float
            Euclidean norm of that residual.

        Notes
        -----
        Both residual quantities are recomputed once after the loop, so they
        describe the image actually returned rather than the last iterate.

        The coarse channel costs the analysis step no extra transform:
        :func:`~dawis.wavelets.atrous` returns the smooth planes alongside
        the wavelet ones, and index ``n_levels`` of them *is* the iterate
        smoothed ``n_levels`` times. What the channel does cost is one extra
        smoothing transpose per ``TRANSPOSE`` synthesis, and one extra
        smoothing cascade per ``ADJOINT`` one.
        """

        # All three axes are refused here, ahead of the tube build:
        # `tubes` raises a `DawisValueError` of its own on a coarse cube of
        # the wrong depth or basis, and the call most likely to trip these
        # guards -- a positional caller written before #101 -- is as likely
        # to trip that one. Reported in the other order, the axis error hides
        # behind the coarse-cube one, which is the failure being guarded.
        #
        # The three `if`/`elif` chains the operator selects between have no
        # `else`, so a value naming none of them matched nothing, left
        # `reconstructed` unassigned and surfaced as an `UnboundLocalError`
        # several frames from the cause. #101 made that worth guarding rather
        # than noting: removing `deconv` shifted this into the fourth
        # positional slot, so a caller outside the tree who passed
        # positionally now binds here what they meant as `deconv`.
        if synthesis_operator not in SYNTHESIS_OPERATORS:
            raise DawisValueError(
                'synthesis_operator names the operator that rebuilds an '
                'image from a tube -->',
                synthesis_operator, list(SYNTHESIS_OPERATORS))

        # Both CG axes are read inside the loop, where an unrecognised value
        # costs a whole descent before it shows -- and the line direction
        # would not show at all, because the recursion it selects between has
        # a published form to fall back on. A sweep harness with a misspelt
        # cell would file as-written numbers under `textbook`, so the value is
        # refused here rather than silently reinterpreted.
        if step_size not in CG_STEP_SIZES:
            raise DawisValueError(
                'step_size names the formula the conjugate direction '
                'coefficient is built by -->',
                step_size, list(CG_STEP_SIZES))
        if line_direction not in CG_LINE_DIRECTIONS:
            raise DawisValueError(
                'line_direction names the reading of the conjugate '
                'direction recursion -->',
                line_direction, list(CG_LINE_DIRECTIONS))

        coarse = coarse_cube is not None
        wavelet_tube, support_tube, label_tube = self.tubes(
            wavelet_cube, label_cube, coarse_cube = coarse_cube)
        n_levels = wavelet_tube.shape[2] - int(coarse)

        if synthesis_operator == 'ADJOINT':
            reconstructed = self.synthesis_adjoint(wavelet_tube, filter = filter, coarse = coarse)
        elif synthesis_operator == 'TRANSPOSE':
            reconstructed = self.synthesis_transpose(wavelet_tube, filter = filter, coarse = coarse)
        elif synthesis_operator == 'SUM':
            reconstructed = self.synthesis_sum(wavelet_tube)

        wr = wavelet_tube - _analysis_tube(reconstructed, n_levels, filter,
                                           coarse) * support_tube

        if synthesis_operator == 'ADJOINT':
            fr = self.synthesis_adjoint(wr, filter = filter, coarse = coarse )

        elif synthesis_operator == 'TRANSPOSE':
            fr = self.synthesis_transpose(wr, filter = filter, coarse = coarse )

        elif synthesis_operator == 'SUM':
            fr = self.synthesis_sum(wr)

        v = np.copy(fr)

        iter = 0
        total_flux = 1.0
        while iter <= max_iter:

            iter = iter + 1

            norm_wr = np.linalg.norm(wr)
            if norm_wr == 0:
                norm_wr = 1
            delta = np.linalg.norm(fr) / norm_wr
            reconstructed = reconstructed + delta * v
            reconstructed[ reconstructed < 0.0 ] = 0.0

            old_total_flux = total_flux
            total_flux = np.sum(reconstructed)

            if old_total_flux == 0.0:
                old_total_flux = 1.0

            diff_flux = abs( total_flux - old_total_flux ) / old_total_flux
            if diff_flux < sigma_flux:
                if verbose == True:
                    log.info('flux convergence %s %s %s', diff_flux,
                             np.max(reconstructed), np.shape(reconstructed))
                break

            else:

                old_wr = np.copy(wr)
                old_fr = np.copy(fr)

                wr = wavelet_tube - _analysis_tube(reconstructed, n_levels,
                                                   filter, coarse) * support_tube

                if synthesis_operator == 'ADJOINT':
                    fr = self.synthesis_adjoint(wr, filter = filter, coarse = coarse )

                elif synthesis_operator == 'TRANSPOSE':
                    fr = self.synthesis_transpose(wr, filter = filter, coarse = coarse )

                elif synthesis_operator == 'SUM':
                    fr = self.synthesis_sum(wr)

                squared_norm_old_fr = np.linalg.norm( old_fr )**2
                if squared_norm_old_fr == 0.:
                    squared_norm_old_fr = 1.0

                if step_size == 'FR':
                    beta = np.linalg.norm( fr )**2 / squared_norm_old_fr
                elif step_size == 'PRP':
                        beta = np.sum( (fr - old_fr) * fr ) / np.sum( fr * fr )
                elif step_size == 'HS':
                        beta = np.sum( (fr - old_fr) * fr ) / np.sum( (fr - old_fr) * v )
                elif step_size == 'CD':
                        beta = np.linalg.norm( fr )**2 / np.sum( v * old_fr )

                if line_direction == 'textbook':
                    v = fr + beta * v
                else:
                    v = fr + beta * old_fr

        wr = wavelet_tube - _analysis_tube(reconstructed, n_levels, filter,
                                           coarse) * support_tube

        norm_wr = np.sqrt( np.sum( wr**2 ) )
        sum_wr = np.sum( wr )

        return reconstructed, sum_wr, norm_wr
