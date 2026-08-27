"""The FITS boundary.

DAWIS works on arrays. FITS is how an image reaches it, and the target
architecture (#4) confines that to one thin wrapper so no block below the
pipeline ever opens a file: :func:`~dawis.run` takes an array, and
:func:`~dawis.run_fits` takes a path and calls this (#54).

Which HDU is read: the first one that holds an image, in file order, unless
the caller names an index (#65). A metadata-only primary with the pixels in
an extension is the normal shape of survey data and of anything a cutout tool
wrote, so the search is what makes that file free to run on; the override is
what keeps a file with several images usable.

Read only. The pipeline still writes its residual, restored and per-scale
images with astropy directly; that side moves when the object store is
rebuilt in Phase B.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits as _fits

from dawis.exceptions import ConfigError

#: HDU classes that hold an image, and the whole of the selection rule.
#:
#: Membership is a type test rather than a shape test because a table HDU has
#: ``data`` too, and a :class:`~astropy.io.fits.FITS_rec` is not something to
#: hand a wavelet transform. Testing the class also picks up a tile-compressed
#: image for free, which :class:`~astropy.io.fits.CompImageHDU` makes look like
#: any other array.
IMAGE_HDU_TYPES = (_fits.PrimaryHDU, _fits.ImageHDU, _fits.CompImageHDU)

#: The one image HDU class that does not hold an image.
#:
#: :class:`~astropy.io.fits.GroupsHDU` — random groups, which is how UV-FITS
#: carries interferometric visibilities — *subclasses*
#: :class:`~astropy.io.fits.PrimaryHDU`, so it passes the test above and its
#: ``data`` is the record array that test exists to keep out. Named apart
#: rather than dropped from the tuple, because the exclusion is a fact about
#: astropy's hierarchy and not about what DAWIS wants.
NOT_IMAGE_HDU_TYPES = (_fits.GroupsHDU,)


def _holds_an_image(hdu) -> bool:
    """Whether one HDU is an image HDU that actually carries pixels."""

    return (isinstance(hdu, IMAGE_HDU_TYPES)
            and not isinstance(hdu, NOT_IMAGE_HDU_TYPES)
            and hdu.data is not None)


def _describe(hdulist: _fits.HDUList) -> str:
    """What a file holds, as one line, for the error that says it holds no image."""

    return ', '.join(
        '%d %s %s' % (index, hdu.name or '(unnamed)', type(hdu).__name__)
        for index, hdu in enumerate(hdulist)
    )


def read_image(filename: str, hdu: int | None = None
               ) -> tuple[np.ndarray, _fits.Header]:
    """Read the image and header of the first FITS HDU that holds an image.

    Parameters
    ----------
    filename : str
        Path to the FITS file.
    hdu : int, optional
        Index of the HDU to read. Default searches: the HDUs are walked in
        file order and the first that holds an image is taken. Given an index,
        that HDU is read and no other — a named HDU with no image data is an
        error rather than an invitation to search, because silently reading a
        different image than the one asked for is how a run ends up analysing
        the wrong file.

    Returns
    -------
    image : numpy.ndarray
        The selected HDU's data array, in the file's own dtype and byte order.

    header : astropy.io.fits.Header
        The selected HDU's header, which the pipeline copies onto the images
        it writes.

    Raises
    ------
    ConfigError
        If ``hdu`` names an HDU that is not in the file or holds no image
        data, or if no HDU in the file holds an image. A run pointed at a file
        with no image in it is a run configured in a way that cannot work, the
        same family as a mistyped extension, so it fails the same way and the
        message names the path.

    Notes
    -----
    An HDU "holds an image" if it is a :class:`~astropy.io.fits.PrimaryHDU`,
    an :class:`~astropy.io.fits.ImageHDU` or a
    :class:`~astropy.io.fits.CompImageHDU` *and* its data is not None. A shape
    test alone would not do: a table HDU has data too. The one exception is
    :class:`~astropy.io.fits.GroupsHDU`, which subclasses ``PrimaryHDU`` and
    carries a record array; see :data:`NOT_IMAGE_HDU_TYPES`.

    The test is on the HDU's class, not on the rank of its array. A 1-D or 3-D
    ``ImageHDU`` therefore wins the search if it comes first, and the run fails
    on it in :func:`~dawis.run` with a
    :class:`~dawis.exceptions.DawisDimensionError` naming the rank. That is the
    intended reading of "type test, not shape test" (#65): a file whose first
    image is not the science image is a file whose HDU should be named.

    The header comes from the same HDU as the image, and **the primary header
    is not merged into an extension's**. That is a decision, not an oversight
    (#65). A merged header is a header no file ever contained, and the merge
    rule — who wins on a duplicate card — would be DAWIS inventing FITS
    semantics. The outputs of a run should be traceable to one HDU of one
    input file. A caller who needs cards from the primary can read them; they
    are not lost, only not silently blended.

    The array is read into memory and the file is closed before returning.
    astropy loads data lazily and memory-maps it, so a wrapper that returned
    the lazy array would hand back a view onto a file that closes the moment
    its ``HDUList`` goes out of scope. The pipeline read the file inline
    before this wrapper existed and kept that ``HDUList`` alive by accident,
    in a local it later reused; the copy makes the guarantee explicit rather
    than incidental.
    """

    with _fits.open(filename) as hdulist:
        if hdu is not None:
            try:
                selected = hdulist[hdu]
            except (IndexError, KeyError):
                raise ConfigError(
                    'No such HDU --> %s has no HDU %s, only: %s'
                    % (filename, hdu, _describe(hdulist))
                ) from None

            if not _holds_an_image(selected):
                raise ConfigError(
                    'Named HDU holds no image --> HDU %s of %s is a %s with '
                    'no image data' % (hdu, filename, type(selected).__name__)
                )
        else:
            selected = next(
                (candidate for candidate in hdulist
                 if _holds_an_image(candidate)),
                None,
            )

            if selected is None:
                raise ConfigError(
                    'No image in file --> %s holds no image HDU, only: %s'
                    % (filename, _describe(hdulist))
                )

        return np.array(selected.data), selected.header.copy()
