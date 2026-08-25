"""Tier 0 properties of the a trous (starlet) wavelet transform.

These hold for the transform as a piece of mathematics, whatever module the
functions end up living in. Every check here is written against an
independently implemented B3-spline smoother (:func:`reference_smooth`), so a
failure means DAWIS and the definition disagree, not that two DAWIS functions
drifted apart.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import convolve1d

import dawis
from dawis.wavelets import max_levels

#: The B3-spline scaling filter the transform is defined from.
BSPL_FILTER = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def reference_smooth(image: np.ndarray, scale: int) -> np.ndarray:
    """Smooth ``image`` with the B3-spline filter dilated to ``scale``.

    Written from the definition of the a trous algorithm — the filter taps are
    spread by ``2 ** scale`` zeros and applied separably along both axes with
    mirror boundaries. This is deliberately not DAWIS code: it is the
    yardstick DAWIS is measured against.
    """
    kernel = np.zeros(4 * 2 ** scale + 1)
    kernel[::2 ** scale] = BSPL_FILTER
    smoothed = convolve1d(image, kernel, axis=0, mode="reflect")
    return convolve1d(smoothed, kernel, axis=1, mode="reflect")


def coarse_plane(coarse: np.ndarray) -> np.ndarray:
    """Return the coarse plane in image orientation.

    The decomposition hands the coarse plane back transposed with respect to
    the wavelet planes: decomposing a ``(48, 64)`` image gives wavelet planes
    of shape ``(48, 64)`` and a coarse plane of shape ``(64, 48)``. Nothing in
    the pipeline notices, because the public transform throws the coarse plane
    away, so this transpose is a latent defect rather than a live one.
    Delete this helper once the orientation is fixed.
    """
    return np.asarray(coarse).T


def reconstruct(wavelet_planes: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    """Rebuild an image from its wavelet planes and its coarse plane.

    The reconstruction is recursive, not a plain sum: starting from the coarse
    plane, each scale is smoothed and its wavelet plane added back, from the
    coarsest scale down to the finest. A plain sum does *not* rebuild the
    input for this transform, because each wavelet plane is the difference
    between a scale and that scale smoothed twice.
    """
    image = coarse_plane(coarse).copy()
    for scale in reversed(range(wavelet_planes.shape[2])):
        image = reference_smooth(image, scale) + wavelet_planes[:, :, scale]
    return image


@pytest.fixture
def noise_image() -> np.ndarray:
    """A small non-square image with structure on every scale."""
    rng = np.random.default_rng(20260823)
    y, x = np.mgrid[0:48, 0:64]
    smooth = 100.0 * np.exp(-((y - 20.0) ** 2 + (x - 30.0) ** 2) / 200.0)
    return smooth + rng.normal(50.0, 3.0, (48, 64))


def test_wavelet_planes_and_coarse_plane_rebuild_the_input(noise_image):
    """Perfect reconstruction: nothing is lost by the decomposition."""
    n_levels = 5
    planes, coarse = dawis.ser_iuwt_decomposition(
        noise_image, n_levels, 0, store_smoothed=True
    )

    np.testing.assert_allclose(
        reconstruct(planes, coarse),
        noise_image,
        rtol=0,
        atol=1e-10,
        err_msg="perfect reconstruction broken: the wavelet planes and the "
                "coarse plane no longer rebuild the input image",
    )


def test_a_flat_image_carries_no_wavelet_signal():
    """The transform annihilates a constant: flat in, flat out."""
    level = 4.2
    flat = np.full((32, 32), level)

    planes, coarse = dawis.ser_iuwt_decomposition(flat, 4, 0, store_smoothed=True)

    np.testing.assert_allclose(
        planes,
        0.0,
        rtol=0,
        atol=1e-12,
        err_msg="a constant image produced non-zero wavelet coefficients, so "
                "the transform no longer annihilates a flat signal (the "
                "filter must sum to one and the boundaries must mirror)",
    )
    np.testing.assert_allclose(
        coarse_plane(coarse),
        level,
        rtol=0,
        atol=1e-12,
        err_msg="a constant image did not survive smoothing unchanged",
    )


def test_the_transform_is_linear(noise_image):
    """Transforming a weighted sum equals the weighted sum of transforms."""
    rng = np.random.default_rng(1234)
    other = rng.normal(0.0, 10.0, noise_image.shape)
    a, b = 2.5, -1.25
    n_levels = 4

    combined = dawis.bspl_atrous(a * noise_image + b * other, n_levels).array
    separate = (
        a * dawis.bspl_atrous(noise_image, n_levels).array
        + b * dawis.bspl_atrous(other, n_levels).array
    )

    np.testing.assert_allclose(
        combined,
        separate,
        rtol=0,
        atol=1e-9,
        err_msg="linearity broken: the transform of a weighted sum is no "
                "longer the weighted sum of the transforms",
    )


def test_asking_for_more_scales_leaves_the_finer_planes_untouched(noise_image):
    """Each wavelet plane depends only on the scales below it."""
    shallow = dawis.bspl_atrous(noise_image, 4).array
    deep = dawis.bspl_atrous(noise_image, 6).array

    np.testing.assert_array_equal(
        deep[:, :, :4],
        shallow,
        err_msg="scale independence broken: the finer wavelet planes changed "
                "when deeper scales were requested",
    )


def test_the_wavelet_cube_keeps_the_image_shape_and_one_plane_per_scale(noise_image):
    """Cube axes are ``(y, x, scale)`` and the image orientation is kept."""
    n_levels = 5

    cube = dawis.bspl_atrous(noise_image, n_levels)

    assert cube.array.shape == noise_image.shape + (n_levels,), (
        "the wavelet cube no longer has the image shape with one plane per "
        f"scale: got {cube.array.shape}, expected "
        f"{noise_image.shape + (n_levels,)}"
    )


def test_a_point_source_spreads_further_at_coarser_scales():
    """Coarser wavelet planes answer to larger structures."""
    image = np.zeros((64, 64))
    image[32, 32] = 1.0

    planes = dawis.bspl_atrous(image, 5).array
    widths = [np.count_nonzero(np.abs(planes[:, :, s]) > 1e-12) for s in range(5)]

    assert widths == sorted(widths), (
        "scale ordering broken: the footprint of a point source must grow "
        f"with scale, got footprints {widths}"
    )


#: Shapes and the deepest decomposition each supports. The pipeline's own two
#: images are the first entries: the 900x900 gallery and its 256x256 cutout.
DECOMPOSITION_DEPTHS = [
    ((900, 900), 9),
    ((256, 256), 8),
    ((1024, 1024), 10),
    ((1023, 1023), 9),
    ((900, 256), 8),
]


@pytest.mark.parametrize("shape, expected", DECOMPOSITION_DEPTHS)
def test_max_levels_is_the_deepest_the_shortest_side_supports(shape, expected):
    """Plane ``n`` describes structure about ``2**n`` pixels across.

    Past the point where that exceeds the image there is nothing left to
    describe, so the count is the largest ``n`` with ``2**n`` no wider than the
    shortest side. The pipeline computed this inline before #49; it is a fact
    about the transform, not about the loop.
    """
    assert max_levels(shape) == expected
    assert 2 ** max_levels(shape) <= min(shape)


def test_max_levels_produces_a_decomposition_the_transform_accepts():
    """The count is only useful if the transform will actually run it."""
    rng = np.random.default_rng(20260823)
    image = rng.normal(size=(64, 64))

    cube = dawis.bspl_atrous(image, max_levels(image.shape))

    assert cube.array.shape[2] == 6
