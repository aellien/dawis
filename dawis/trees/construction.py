"""Interscale trees: what a detected structure is before it is restored.

A :class:`interscale_tree` is one structure seen at every scale at once. It
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

The **tubes** are the tree cut out of the cubes. :meth:`~interscale_tree.tubes`
returns the wavelet coefficients, the binary support and the labels of the
member regions, stacked over the scales the tree spans and clipped to its
bounding box. Everything downstream works on tubes rather than on full planes.

The **synthesis** turns a tube back into an image.
:meth:`~interscale_tree.CG_minimization` is the published solver of
Ellien+2021: it runs a conjugate-gradient descent that looks for the image
whose wavelet transform matches the tube, starting from either of two
synthesis operators — :meth:`~interscale_tree.synthesis_adjoint`, the
transposed transform, or :meth:`~interscale_tree.synthesis_sum`, the plain sum
of planes. Its output is the atom.
"""

from __future__ import annotations

import numpy as np

from dawis.cubes import label_datacube, wavelet_datacube
from dawis.deconvolution import richardson_lucy_np, twod_B2spline
from dawis.detection import region
from dawis.wavelets import atrous

try:
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib is the optional [viz] extra
    from dawis._optional import MissingDependency
    plt = MissingDependency('matplotlib', 'viz')
try:
    from matplotlib.colors import SymLogNorm
except ImportError:  # matplotlib is the optional [viz] extra
    from dawis._optional import MissingDependency
    SymLogNorm = MissingDependency('matplotlib.colors.SymLogNorm', 'viz')


class interscale_tree(object):
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
    wavelet_datacube : dawis.cubes.wavelet_datacube, optional
        Wavelet planes the regions were measured on. Read once, to size the
        extent; not stored.
    label_datacube : dawis.cubes.label_datacube, optional
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
        wavelet_datacube: wavelet_datacube | None = None,
        label_datacube: label_datacube | None = None,
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
            clip = np.copy(label_datacube.array[ x_min:x_max, \
                                                y_min:y_max, \
                                                    : ])
            self.bbox = (x_min, y_min, x_max, y_max)
            extent_im = np.sum( self.tubes( wavelet_datacube, label_datacube )[1], axis = 2 )
            n_pix = np.size( np.where( extent_im != 0. )[0] )
            self.extent = n_pix / ( ( x_max - x_min ) * ( y_max - y_min ) )

            # Make vignet larger
            x_min -= int( ( x_max - x_min ) / 2. )
            x_max += int( ( x_max - x_min ) / 2. )
            if x_min < 0: x_min = 0
            if x_max > label_datacube.x_size: x_max = label_datacube.x_size

            y_min -= int( ( y_max - y_min ) / 2. )
            y_max += int( ( y_max - y_min ) / 2. )
            if y_min < 0: y_min = 0
            if y_max > label_datacube.y_size: y_max = label_datacube.y_size

            # Clip
            if x_max - x_min < self.clip:
                x_min = int( (x_max + x_min) / 2. - self.clip / 2. + 1.)   # we resize it to the minclip
                x_max = int( (x_max + x_min) / 2. + self.clip / 2. + 1.)   # value.
                if x_min < 0: x_min = 0
                if x_max > label_datacube.x_size: x_max = label_datacube.x_size

            if y_max - y_min < self.clip:
                y_min = int( (y_max + y_min) / 2. - self.clip / 2. + 1.)   # we resize it to the minclip
                y_max = int( (y_max + y_min) / 2. + self.clip / 2. + 1.)   # value.
                if y_min < 0: y_min = 0
                if y_max > label_datacube.y_size: y_max = label_datacube.y_size

            self.bbox = (x_min, y_min, x_max, y_max)
            self.x_size = x_max - x_min
            self.y_size = y_max - y_min

            # Bayesian parameter bounds
            self.lowb = 0.
            self.highb = 2.
        
    def det_err_tube(
        self,
        wavelet_datacube: wavelet_datacube,
        label_datacube: label_datacube,
    ) -> np.ndarray:
        """Build the detection uncertainty of the tree, plane by plane.

        The support tube says which pixels of the tree are significant at each
        scale; each plane of it is multiplied by that scale's entry of
        ``det_err_array``.

        Parameters
        ----------
        wavelet_datacube : dawis.cubes.wavelet_datacube
            Wavelet planes the tree was built on.
        label_datacube : dawis.cubes.label_datacube
            Label cube the tree was built on.

        Returns
        -------
        numpy.ndarray
            Cube of shape ``(x_size, y_size, level + 1)``, zero outside the
            tree and equal to the per-scale uncertainty inside it. The
            restoration sums it into the uncertainty map of the atom.
        """

        wavelet_tube, support_tube, label_tube = self.tubes(wavelet_datacube, label_datacube)
        det_err_tube = np.copy(support_tube)
        
        for level in range(0, det_err_tube.shape[2]):
                det_err_tube[:,:,level] *= self.det_err_array[level]
        
        return det_err_tube
            
    
    def plot(
        self,
        wavelet_datacube: wavelet_datacube,
        label_datacube: label_datacube,
        name: str | None = None,
        show: bool = True,
        save_path: str | None = None,
        **kwargs,
    ) -> None:
        """Draw the tree: its summed wavelet coefficients beside its regions.

        Parameters
        ----------
        wavelet_datacube : dawis.cubes.wavelet_datacube
            Wavelet planes the tree was built on.
        label_datacube : dawis.cubes.label_datacube
            Label cube the tree was built on.
        name : str, optional
            Figure title. Default is the label of the interscale maximum.
        show : bool, optional
            Display the figure and block. Default is True. When False the
            figure is closed instead, which is what saving in a loop wants.
        save_path : str, optional
            Write the figure there as PNG. Default is not to save.
        **kwargs
            Passed to :func:`matplotlib.pyplot.imshow`. A ``norm`` entry is
            honoured; without one a symmetric-log norm is used.

        Notes
        -----
        Requires the ``[viz]`` extra. This method moves to ``dawis.viz`` when
        that block is carved out — no other module here touches matplotlib.
        """

        fig, ax = plt.subplots(1, 2, sharex = True, sharey = True, \
                                  gridspec_kw = {'hspace': 0.5, 'wspace': 0.5})

        if name == None:
            fig.suptitle('Interscale tree of region n°%d' %(self.interscale_maximum.label) )
        else:
            fig.suptitle(name)

        wavelet_image = np.sum(wavelet_datacube.array[ self.bbox[0]:self.bbox[2], \
                                                       self.bbox[1]:self.bbox[3], \
                                                       0:self.interscale_maximum.level + 1 ], axis = 2)

        try:
            norm = kwargs.pop('norm')
            im = ax[0].imshow( wavelet_image, \
                    extent = ( self.bbox[1], self.bbox[3], \
                               self.bbox[0], self.bbox[2] ), \
                    origin = 'lower', \
                    **kwargs )
            plt.colorbar(im, ax = ax[0], orientation = 'vertical')
        except:
            logthresh = 1E-3
            norm = SymLogNorm(logthresh)
            im = ax[0].imshow( wavelet_image, extent = ( self.bbox[1], self.bbox[3], \
                                                         self.bbox[0], self.bbox[2] ), \
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


        label_image = np.zeros((self.bbox[2] - self.bbox[0], \
                                self.bbox[3] - self.bbox[1]))

        for i, region in enumerate(self.region_list):

            ax[0].plot( region.y_max, region.x_max, '+' )

            clip = np.copy(label_datacube.array[ self.bbox[0]:self.bbox[2], \
                                                 self.bbox[1]:self.bbox[3], \
                                                 int(region.level) ])

            clip[np.where(clip != region.label)] = 0.

            try:
                ax[0].contour( clip, region.label,
                                     extent = ( self.bbox[1], self.bbox[3], \
                                                self.bbox[0], self.bbox[2] ), \
                                     levels = 0, \
                                     color = 'cyan')
            except:
                print('Region too small for contours.')

            clip[np.where(clip == region.label)] = 1
            label_image += clip
            ax[1].plot( region.y_max, region.x_max, '+' )

        ax[1].imshow(label_image, extent = ( self.bbox[1], self.bbox[3], \
                                             self.bbox[0], self.bbox[2] ), \
                                  origin = 'lower',
                                  cmap = 'binary'  )

        if save_path != None:
            plt.savefig(save_path, format = 'png')

        if show == True:
            plt.show()
        else:
            plt.close()

    def tubes(
        self,
        wavelet_datacube: wavelet_datacube,
        label_datacube: label_datacube,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cut the tree out of the cubes.

        Only the pixels of the member regions are copied, each onto the plane
        of its own scale, into arrays clipped to the bounding box of the tree.
        Everything else is zero.

        Parameters
        ----------
        wavelet_datacube : dawis.cubes.wavelet_datacube
            Wavelet planes to read the coefficients from.
        label_datacube : dawis.cubes.label_datacube
            Label cube to read the labels from.

        Returns
        -------
        wavelet_tube : numpy.ndarray
            Wavelet coefficients of the tree, shape
            ``(x_size, y_size, level + 1)``.
        support_tube : numpy.ndarray
            The same cube as a binary mask — 1 on a member pixel, 0 elsewhere.
        label_tube : numpy.ndarray
            The same cube carrying region labels instead of coefficients.

        Notes
        -----
        The tube spans every scale from 0 up to the level of the interscale
        maximum, including scales on which the tree has no region. Those
        planes stay zero.
        """

        label_tube = np.zeros((self.bbox[2] - self.bbox[0], \
                               self.bbox[3] - self.bbox[1], \
                               self.interscale_maximum.level + 1))

        wavelet_tube = np.copy(label_tube)

        for region in self.region_list:

            wavelet_tube[ region.coords[:,0] - self.bbox[0], \
                          region.coords[:,1] - self.bbox[1], \
                          region.level] = wavelet_datacube.array[ region.coords[:,0], \
                                                          region.coords[:,1], \
                                                          region.level ]

            label_tube[ region.coords[:,0] - self.bbox[0], \
                        region.coords[:,1] - self.bbox[1], \
                        region.level] = label_datacube.array[ region.coords[:,0], \
                                                              region.coords[:,1], \
                                                              region.level ]

        support_tube = np.copy(label_tube)
        support_tube[support_tube != 0] = 1.

        return wavelet_tube, support_tube, label_tube

    def synthesis_adjoint(
        self,
        wavelet_tube: np.ndarray,
        filter: np.ndarray,
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

        Returns
        -------
        numpy.ndarray
            2D image of the shape of one tube plane, the first approximate
            reconstruction the conjugate gradient starts from.

        See Also
        --------
        synthesis_sum : The cheaper alternative, a plain sum of the planes.
        """

        smooth_coeff = np.zeros(wavelet_tube.shape)

        for level in range(wavelet_tube.shape[2]):

            smooth_coeff[:, :, level] = atrous( wavelet_tube[:, :, level], \
                                               n_levels = level + 1, \
                                               filter = filter, \
                                               conditions = 'prolongation' )[0][:, :, level]
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
        synthesis_adjoint : The transposed-transform alternative.
        """
        return np.sum( wavelet_tube, axis = 2 )

    def CG_minimization(
        self,
        wavelet_datacube: wavelet_datacube,
        label_datacube: label_datacube,
        filter: np.ndarray,
        deconv: bool = False,
        synthesis_operator: str = 'ADJOINT',
        step_size: str = 'FR',
        sigma_flux: float = 1E-3,
        max_iter: int = 3,
        verbose: bool = False,
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
        wavelet_datacube : dawis.cubes.wavelet_datacube
            Wavelet planes the tree was built on.
        label_datacube : dawis.cubes.label_datacube
            Label cube the tree was built on.
        filter : numpy.ndarray
            1D low-band filter of the à trous transform, normally B3-spline.
        deconv : bool, optional
            Deconvolve each reconstruction with a B2-spline PSF by ten
            Richardson-Lucy iterations. Default is False. Only meaningful
            with ``synthesis_operator='ADJOINT'``.
        synthesis_operator : {'ADJOINT', 'SUM'}, optional
            Which operator rebuilds an image from a tube. Default is
            ``'ADJOINT'``.
        step_size : {'FR', 'PRP', 'HS', 'CD'}, optional
            Formula for the conjugate direction coefficient — Fletcher-Reeves,
            Polak-Ribière-Polyak, Hestenes-Stiefel or conjugate descent.
            Default is ``'FR'``, the published choice.
        sigma_flux : float, optional
            Relative flux change below which the descent stops. Default is
            1E-3.
        max_iter : int, optional
            Iteration cap. Default is 3; the published configuration is 200,
            at which the ``sigma_flux`` test does the stopping.
        verbose : bool, optional
            Print the convergence line when the flux test fires. Default is
            False.

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
        """

        wavelet_tube, support_tube, label_tube = self.tubes(wavelet_datacube, label_datacube)
        n_levels = wavelet_tube.shape[2]

        if deconv == True:
            if self.interscale_maximum.level <= 2:
                im_psf = twod_B2spline(xsize = self.x_size, ysize = self.y_size, order = 0)

            else:
                im_psf = twod_B2spline(xsize = self.x_size, ysize = self.y_size, order = 2)
    
        if synthesis_operator == 'ADJOINT':
            reconstructed = self.synthesis_adjoint(wavelet_tube, filter = filter)
            if deconv == True:
                reconstructed = richardson_lucy_np(reconstructed, im_psf, 10)
        elif synthesis_operator == 'SUM':
            reconstructed = self.synthesis_sum(wavelet_tube)

        wr = wavelet_tube - atrous(reconstructed, n_levels = n_levels, \
                                                  filter = filter, \
                                                  conditions = 'prolongation' )[1] * support_tube

        if synthesis_operator == 'ADJOINT':
            fr = self.synthesis_adjoint(wr, filter = filter )
            if deconv == True:
                fr = richardson_lucy_np(fr, im_psf, 10)
                
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
                    print( "flux convergence", diff_flux, np.max(reconstructed), np.shape(reconstructed) )
                break

            else:

                old_wr = np.copy(wr)
                old_fr = np.copy(fr)

                wr = wavelet_tube - atrous( reconstructed, n_levels = n_levels, \
                                                filter = filter, \
                                                conditions = 'prolongation' )[1] * support_tube

                if synthesis_operator == 'ADJOINT':
                    fr = self.synthesis_adjoint(wr, filter = filter )
                    if deconv == True:
                        fr = richardson_lucy_np(fr, im_psf, 10)
                        
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
                v = fr + beta * old_fr

        wr = wavelet_tube - atrous(reconstructed, n_levels = wavelet_tube.shape[2], \
                                                      filter = filter, \
                                                      conditions = 'prolongation' )[1] * support_tube

        norm_wr = np.sqrt( np.sum( wr**2 ) )
        sum_wr = np.sum( wr )

        return reconstructed, sum_wr, norm_wr
