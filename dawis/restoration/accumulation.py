"""The running images of a run: what has been restored, and what is left.

Synthesis by analysis is a subtraction loop. It starts with the image, and
every iteration takes the atoms it restored out of a running residual and adds
them to a running restored image, so that the next iteration analyses only what
the previous one could not explain. :class:`Reconstruction` is those running
images and the one operation that advances them.

Four arrays travel together, which is the reason they are one object rather
than four locals in the pipeline:

* the **residual**, the image minus every atom so far — the array the next
  iteration is analysed from;
* the **restored** image, the sum of every atom so far;
* the same sum kept **per wavelet scale**, so a run can be inspected scale by
  scale afterwards;
* the accumulated **detection error**, the uncertainty image each atom carries.

Every atom updates all four, each inside its own bounding box, and getting one
of the four out of step with the others is exactly the class of bug this
prevents.

The residual is also where inpainting happens. A slightly over-restored atom
digs a hole in it, and the next iteration would detect that hole as structure,
so :meth:`Reconstruction.inpaint` refills it from
:mod:`dawis.restoration.inpainting` before the next pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from dawis.restoration.inpainting import inpaint_with_gaussian_noise

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from dawis.restoration.synthesis import Atom

__all__ = ['Reconstruction']


class Reconstruction:
    """The residual, the restored image, and the two images kept beside them.

    Parameters
    ----------
    image : numpy.ndarray
        The image the run iterates on, with the sky background already
        removed. Copied, not held: the caller's array is never written to.
    n_levels : int
        Number of wavelet scales, which fixes the depth of
        :attr:`restored_per_level`.

    Attributes
    ----------
    residuals : numpy.ndarray
        The image minus every atom restored so far, in the dtype of ``image``.
        This is what each iteration analyses.
    restored : numpy.ndarray
        The sum of every atom restored so far.
    restored_per_level : numpy.ndarray
        The same sum split by the wavelet scale each atom was restored at,
        shaped ``(x, y, n_levels)``.
    det_err : numpy.ndarray
        The accumulated per-atom detection error image.

    Examples
    --------
    >>> import numpy as np
    >>> run = Reconstruction(np.ones((4, 4)), n_levels = 2)
    >>> run.residuals.shape, run.restored_per_level.shape
    ((4, 4), (4, 4, 2))
    """

    def __init__(self, image: np.ndarray, n_levels: int):

        self.residuals = np.copy(image)
        self.restored = np.zeros(image.shape)
        self.restored_per_level = np.zeros((image.shape[0], image.shape[1], n_levels))
        self.det_err = np.zeros(image.shape)

    def add(self, atoms: Sequence[Atom]) -> np.ndarray:
        """Take one iteration's atoms out of the residual and into the rest.

        Parameters
        ----------
        atoms : sequence of Atom
            The atoms one iteration restored. Each is added inside its own
            bounding box, so atoms that overlap accumulate.

        Returns
        -------
        numpy.ndarray
            The sum of these atoms alone, on the full image grid. The
            convergence test and the animation both work from this rather than
            from a difference of the running images, so it is returned rather
            than recomputed.
        """

        atom = np.zeros(self.residuals.shape)

        for o in atoms:
            x_min, y_min, x_max, y_max = o.bbox
            atom[ x_min : x_max, y_min : y_max ] += o.image
            self.restored_per_level[ x_min : x_max, y_min : y_max, o.level ] += o.image
            self.det_err[ x_min : x_max, y_min : y_max ] += o.det_err_image

        self.residuals -= atom
        self.restored += atom

        return atom

    def inpaint(
        self,
        mean: float,
        sigma: float,
        iptd_sigma: float,
        rng: np.random.RandomState | None = None,
    ) -> None:
        """Refill the holes this run's subtractions dug in the residual.

        Parameters
        ----------
        mean, sigma : float
            Noise properties of the image, as
            :func:`~dawis.restoration.inpainting.subtract_background`
            measured them.
        iptd_sigma : float
            Threshold in units of ``sigma``: pixels below minus this many
            sigmas, and NaNs, are replaced with fresh noise draws.
        rng : numpy.random.RandomState, optional
            Generator those draws come from, threaded on to
            :func:`~dawis.restoration.inpainting.inpaint_with_gaussian_noise`.
            Default None uses the global :mod:`numpy.random` state.

        See Also
        --------
        dawis.restoration.inpainting.inpaint_with_gaussian_noise : Does the work.
        """

        self.residuals = inpaint_with_gaussian_noise(
            self.residuals, mean = mean, sigma = sigma, iptd_sigma = iptd_sigma,
            rng = rng)
