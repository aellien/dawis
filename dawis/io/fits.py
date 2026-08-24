"""The FITS boundary.

DAWIS works on arrays. FITS is how an image reaches it, and the target
architecture (#4) confines that to one thin wrapper so no block below the
pipeline ever opens a file: :func:`~dawis.run` takes an array, and
``run_fits`` (#54) will take a path and call this.

Read only. The pipeline still writes its residual, restored and per-scale
images with astropy directly; that side moves when the object store is
rebuilt in Phase B.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits as _fits


def read_image(filename: str) -> tuple[np.ndarray, _fits.Header]:
    """Read the image and header of a FITS file's primary HDU.

    Parameters
    ----------
    filename : str
        Path to the FITS file.

    Returns
    -------
    image : numpy.ndarray
        The primary HDU's data array, in the file's own dtype and byte order.

    header : astropy.io.fits.Header
        The primary HDU's header, which the pipeline copies onto the images
        it writes.

    Notes
    -----
    The array is read into memory and the file is closed before returning.
    astropy loads data lazily and memory-maps it, so a wrapper that returned
    the lazy array would hand back a view onto a file that closes the moment
    its ``HDUList`` goes out of scope. The pipeline read the file inline
    before this wrapper existed and kept that ``HDUList`` alive by accident,
    in a local it later reused; the copy makes the guarantee explicit rather
    than incidental.
    """

    with _fits.open(filename) as hdu:
        return np.array(hdu[0].data), hdu[0].header.copy()
