"""Tier 0 — which HDU the FITS boundary reads (#65).

``read_image`` used to read HDU 0 and nothing else, so a multi-extension file
whose primary carries only a header reached the pipeline as ``np.array(None)``
and died four frames later inside ``inspect_image``. That layout — a metadata
primary with the pixels in an extension — is the normal shape of survey data
and of anything a cutout tool wrote.

What is asserted here is the selection rule and its two refusals:

**Search by default.** With no ``hdu`` given, the first HDU in file order that
is an image *and* holds data wins. A table HDU also has data, so the test is
on the HDU's class, not on its shape.

**The header travels with the data.** When the image comes from extension 1,
extension 1's header is what the outputs carry — and the primary's cards are
not merged in, because a merged header is a header no file ever contained.

**A named ``hdu`` never falls back.** ``hdu = 0`` on a header-only primary is
a ``ConfigError``, not an invitation to go and read a different image.

The files are built in ``tmp_path``: a hand-built HDU list states the layout
under test more clearly than a binary fixture would, and no fixture file has
to enter the repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import dawis
from tinyrun import TINY_PARAMS
from tinyrun import tiny_image as make_tiny_image

from dawis.config import Config
from dawis.exceptions import ConfigError
from dawis.io import read_image

#: The one seed the end-to-end run pins.
SEED = 20260827


def small_image(value: float = 1.0) -> np.ndarray:
    """A 4x4 frame with one recognisable pixel, for identity assertions."""
    image = np.full((4, 4), value, dtype = 'float32')
    image[0, 0] = 42.0
    return image


def write(path: Path, *hdus: fits.hdu.base._BaseHDU) -> str:
    """Write one hand-built HDU list, and answer with its path as a string."""
    fits.HDUList(list(hdus)).writeto(path)
    return str(path)


def header_only_primary() -> fits.PrimaryHDU:
    """A metadata primary — the ``CROP-CONFIG`` HDU of a Euclid VIS cutout."""
    primary = fits.PrimaryHDU()
    primary.header['PRIMONLY'] = 'primary'
    primary.header['OBJECT'] = 'primary'
    return primary


def a_table() -> fits.BinTableHDU:
    """A catalogue extension: data, but nothing to hand a wavelet transform."""
    return fits.BinTableHDU.from_columns(
        [fits.Column(name = 'flux', format = 'E', array = np.arange(3.0))]
    )


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


def test_a_primary_with_data_is_still_read_from_hdu_zero(tmp_path):
    """Today's behaviour, unchanged: the ordinary single-HDU file."""
    image = small_image()
    primary = fits.PrimaryHDU(image)
    primary.header['OBJECT'] = 'primary'
    path = write(tmp_path / 'plain.fits', primary)

    data, header = read_image(path)

    np.testing.assert_array_equal(data, image)
    assert header['OBJECT'] == 'primary'


def test_a_header_only_primary_is_skipped_for_the_extension(tmp_path):
    """The case the ticket came from: the pixels live in extension 1."""
    image = small_image(2.0)
    path = write(tmp_path / 'cutout.fits',
                 header_only_primary(),
                 fits.ImageHDU(image, name = 'CROP'))

    data, _ = read_image(path)

    np.testing.assert_array_equal(data, image)


def test_the_header_comes_from_the_hdu_the_image_came_from(tmp_path):
    """Extension 1's header is where the astrometry is, so it is what travels."""
    extension = fits.ImageHDU(small_image(), name = 'CROP')
    extension.header['OBJECT'] = 'extension'
    path = write(tmp_path / 'cutout.fits', header_only_primary(), extension)

    _, header = read_image(path)

    assert header['OBJECT'] == 'extension'


def test_the_primary_header_is_not_merged_into_the_extensions(tmp_path):
    """A merged header is a header no file ever contained (#65).

    The primary's cards are not lost — a caller who needs them reads them —
    but they are not silently blended into the header the outputs carry,
    because the merge rule on a duplicate card would be DAWIS inventing FITS
    semantics, and a run's outputs should be traceable to one HDU.
    """
    path = write(tmp_path / 'cutout.fits',
                 header_only_primary(),
                 fits.ImageHDU(small_image(), name = 'CROP'))

    _, header = read_image(path)

    assert 'PRIMONLY' not in header


def test_the_search_skips_a_table_extension(tmp_path):
    """A ``BinTableHDU`` has data too, so the test is on the class, not the shape."""
    image = small_image(3.0)
    path = write(tmp_path / 'catalogue.fits',
                 header_only_primary(),
                 a_table(),
                 fits.ImageHDU(image, name = 'CROP'))

    data, _ = read_image(path)

    np.testing.assert_array_equal(data, image)


def test_the_search_reads_a_tile_compressed_image(tmp_path):
    """``CompImageHDU`` is an image the moment it is decompressed."""
    image = small_image(4.0)
    path = write(tmp_path / 'compressed.fits',
                 header_only_primary(),
                 fits.CompImageHDU(image, name = 'CROP'))

    data, _ = read_image(path)

    np.testing.assert_allclose(data, image)


def test_a_file_with_no_image_at_all_is_a_configuration_error(tmp_path):
    """The message names the path and says what the file does hold."""
    path = write(tmp_path / 'catalogue.fits', header_only_primary(), a_table())

    with pytest.raises(ConfigError) as raised:
        read_image(path)

    message = str(raised.value)
    assert path in message
    assert 'BinTableHDU' in message


# ---------------------------------------------------------------------------
# The override
# ---------------------------------------------------------------------------


def test_a_named_hdu_is_the_one_that_is_read(tmp_path):
    """A caller who knows their file names the extension and gets it."""
    wanted = small_image(5.0)
    path = write(tmp_path / 'two_images.fits',
                 fits.PrimaryHDU(small_image(6.0)),
                 fits.ImageHDU(wanted, name = 'CROP'))

    data, _ = read_image(path, hdu = 1)

    np.testing.assert_array_equal(data, wanted)


def test_a_named_hdu_that_holds_no_image_never_falls_back_to_the_search(tmp_path):
    """Explicit beats helpful: reading a different HDU is how a paper goes wrong."""
    image = small_image(7.0)
    path = write(tmp_path / 'cutout.fits',
                 header_only_primary(),
                 fits.ImageHDU(image, name = 'CROP'))

    with pytest.raises(ConfigError) as raised:
        read_image(path, hdu = 0)

    message = str(raised.value)
    assert path in message
    assert 'HDU 0 of' in message


def test_random_groups_are_not_an_image(tmp_path):
    """``GroupsHDU`` subclasses ``PrimaryHDU`` and carries a record array.

    UV-FITS visibilities pass the class test by inheritance, which is the one
    way a :class:`~astropy.io.fits.FITS_rec` could still reach a wavelet
    transform. Named as its own exclusion rather than trusted to the tuple.
    """
    visibilities = fits.GroupData(
        np.arange(12.0, dtype = 'float32').reshape(2, 1, 1, 2, 3),
        parnames = ['UU'],
        pardata = [np.zeros(2, dtype = 'float32')],
        bitpix = -32,
    )
    path = write(tmp_path / 'uv.fits', fits.GroupsHDU(visibilities))

    with pytest.raises(ConfigError, match = 'GroupsHDU'):
        read_image(path)

    with pytest.raises(ConfigError):
        read_image(path, hdu = 0)


def test_a_named_hdu_that_is_not_in_the_file_is_a_configuration_error(tmp_path):
    """Out of range is the same family of mistake, and reads the same way."""
    path = write(tmp_path / 'plain.fits', fits.PrimaryHDU(small_image()))

    with pytest.raises(ConfigError) as raised:
        read_image(path, hdu = 3)

    message = str(raised.value)
    assert path in message
    assert 'no HDU 3' in message
    assert 'PrimaryHDU' in message


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def test_the_wrapper_runs_on_a_header_only_primary_file(tmp_path):
    """The case the traceback came from, driven end to end.

    The others stop at :mod:`dawis.io`. This one proves the fix reaches
    ``run_fits``, which is where the Euclid VIS cutout of #65 died.
    """
    path = write(tmp_path / 'cutout_VIS.fits',
                 header_only_primary(),
                 fits.ImageHDU(make_tiny_image(), name = 'CROP'))

    result = dawis.run_fits(path, Config(seed = SEED, **TINY_PARAMS),
                            outdir = str(tmp_path))

    assert result.restored.shape == (64, 64)
    assert Path('%s.restored.fits' % result.outpath).is_file()


def test_the_wrapper_forwards_the_named_hdu(tmp_path):
    """``hdu`` is a knob of the run, so the wrapper has to carry it down."""
    path = write(tmp_path / 'two_images.fits',
                 fits.PrimaryHDU(np.zeros((64, 64), dtype = 'float32')),
                 fits.ImageHDU(make_tiny_image(), name = 'CROP'))

    result = dawis.run_fits(path, Config(seed = SEED, **TINY_PARAMS),
                            outdir = str(tmp_path), hdu = 1)

    assert result.n_atoms > 0
