"""The units frame a selection converts physical sizes with (#126).

Selection works in **pixels and levels**. That is what the atom index holds —
``bbox`` in pixels, ``level`` as an integer — and a criterion that tested
anything else would have to carry a cosmology around with it. A user who
thinks in kpc converts once, at the call site, with a
:class:`Units` frame built for the run.

The frame is what the store cannot give you. A 1.0 store keeps **only** the
frame's ``shape`` (``dawis/io/store.py``, :func:`~dawis.read_frame_shape`) —
no WCS, no header, no pixel scale — and the CFIS scripts resampled the image
before the run and recorded the factor in an ``f_resize`` header keyword, so
the original pixel scale is not recoverable from a resampled run at all
(#17, point 6). The user supplies it.

Conversion is **kpc to pixels, never kpc to a level.**
``redshift_wavelet_mapping.setwavz`` mapped kpc to a wavelet level, but only
to *resample the image before the run*, which is a separate, pre-run concern
and not something a selection can undo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen = True)
class Units:
    """Pixel scale, redshift and cosmology — built once, used by the criteria.

    Criteria in pixels and levels ignore it; :class:`~dawis.selection.Size`
    and :class:`~dawis.selection.Distance` take it as ``units`` and then read
    their threshold as kpc rather than pixels::

        frame = Units(pixel_scale = 0.168, redshift = 0.3)
        big = Size(10, units = frame)          # 10 kpc, in both bbox sides

    Parameters
    ----------
    pixel_scale : float
        Arcseconds per pixel of the image the run was made on. After any
        pre-run resampling: it is the scale of the pixels in the store.
    redshift : float, optional
        The redshift the physical sizes are at. Needed only by
        :meth:`kpc_to_pix`.
    cosmology : astropy.cosmology.Cosmology, optional
        Defaults to ``astropy.cosmology.Planck18``, and the default is filled
        in on the frame rather than resolved at each call — so a frame's
        ``repr`` names the cosmology a selection used, instead of leaving the
        assumption unwritten.

    Raises
    ------
    ValueError
        If ``pixel_scale`` is not positive — a zero or negative scale turns
        every physical threshold into infinity or into a negative size, and
        both select the wrong atoms without failing.
    """

    pixel_scale: float
    redshift: float | None = None
    cosmology: Any = None

    def __post_init__(self):
        if not self.pixel_scale > 0.:
            raise ValueError('a pixel scale must be positive, in arcsec per pixel: '
                             'got %r' % (self.pixel_scale,))
        if self.cosmology is None:
            from astropy.cosmology import Planck18

            # The frame is frozen, and this is the one write it needs: a
            # default that stays None is an assumption nobody can read back.
            object.__setattr__(self, 'cosmology', Planck18)

    def kpc_to_pix(self, kpc):
        """``kpc`` proper at :attr:`redshift`, in pixels.

        Parameters
        ----------
        kpc : float or array_like
            A proper size, in kiloparsecs.

        Returns
        -------
        float or numpy.ndarray
            The same size in pixels, of the shape of ``kpc``.

        Raises
        ------
        ValueError
            If the frame carries no positive redshift. There is no distance
            to convert with, and a silent zero would make every size zero.
        """

        if self.redshift is None or not self.redshift > 0.:
            raise ValueError('converting kpc to pixels needs a positive redshift on the '
                             'units frame: got %r' % (self.redshift,))
        arcsec_per_kpc = self.cosmology.arcsec_per_kpc_proper(self.redshift).value
        pixels = np.asarray(kpc) * arcsec_per_kpc / self.pixel_scale
        # A scalar in, a scalar out: a 0-d array works in every comparison a
        # criterion makes, but it prints as ``array(59.5)`` in the repr of the
        # criterion that holds it, which is where a threshold is read back.
        return float(pixels) if pixels.ndim == 0 else pixels
