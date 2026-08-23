"""Tier 0 behaviour of hard thresholding of a wavelet cube.

The cubes here are synthetic, so the answer is known by construction: the
noise plane has an exact spread, and every planted value sits a known
fraction above or below the threshold that spread implies.
"""

from __future__ import annotations

import numpy as np
import pytest

import dawis

#: Exact spread of the synthetic noise plane, in counts.
NOISE_SPREAD = 2.0

#: Shape of the synthetic cubes: 64 by 64 pixels over 6 scales.
CUBE_SHAPE = (64, 64, 6)


def noise_plane(shape: tuple[int, int]) -> np.ndarray:
    """A plane of alternating ``+/- NOISE_SPREAD`` pixels.

    Its mean is exactly zero and its standard deviation exactly
    ``NOISE_SPREAD``, and three-sigma clipping removes nothing, so the spread
    the thresholding measures is known rather than estimated.
    """
    y, x = np.indices(shape)
    return np.where((y + x) % 2 == 0, NOISE_SPREAD, -NOISE_SPREAD).astype(float)


def synthetic_cube() -> np.ndarray:
    """An empty wavelet cube whose finest plane carries the known noise."""
    array = np.zeros(CUBE_SHAPE)
    array[:, :, 0] = noise_plane(CUBE_SHAPE[:2])
    return array


def thresholds(n_sigmas: float) -> np.ndarray:
    """The per-scale threshold the known noise spread implies."""
    return n_sigmas * NOISE_SPREAD * dawis.table_noise_stds(
        CUBE_SHAPE[2], wavelet_type="BSPL"
    )


def test_pure_noise_stays_below_the_threshold():
    """Nothing is flagged in a cube that holds noise and nothing else."""
    cube = dawis.wavelet_datacube(synthetic_cube(), "BSPL")

    support = dawis.hard_threshold(cube, n_sigmas=3)

    assert np.count_nonzero(support.array) == 0, (
        "false detection: pure noise at three sigma produced "
        f"{np.count_nonzero(support.array)} flagged pixels, expected none"
    )


def test_a_value_above_the_threshold_is_flagged_and_one_below_is_not():
    """The threshold cut lands exactly where the noise spread says it does."""
    n_sigmas = 3
    array = synthetic_cube()
    level = thresholds(n_sigmas)
    for scale in range(1, CUBE_SHAPE[2]):
        array[10, 10, scale] = level[scale] * 1.01
        array[20, 20, scale] = level[scale] * 0.99
    cube = dawis.wavelet_datacube(array, "BSPL")

    support = dawis.hard_threshold(cube, n_sigmas=n_sigmas)

    for scale in range(1, CUBE_SHAPE[2]):
        assert support.array[10, 10, scale] == 1.0, (
            f"missed detection at scale {scale}: a value one percent above "
            "the threshold was not flagged"
        )
        assert support.array[20, 20, scale] == 0.0, (
            f"false detection at scale {scale}: a value one percent below "
            "the threshold was flagged"
        )


def test_the_support_holds_only_zeros_and_ones():
    """The support is a mask, not a weighting."""
    array = synthetic_cube()
    array[30, 30, 3] = 1000.0
    cube = dawis.wavelet_datacube(array, "BSPL")

    support = dawis.hard_threshold(cube, n_sigmas=3)

    assert set(np.unique(support.array)) <= {0.0, 1.0}, (
        "the support is no longer binary: it holds values "
        f"{sorted(set(np.unique(support.array)))}"
    )


def test_raising_the_sigma_cut_can_only_remove_detections():
    """The support shrinks monotonically as the cut is raised."""
    rng = np.random.default_rng(20260823)
    array = synthetic_cube()
    array[:, :, 1:] = rng.normal(0.0, 6.0, (64, 64, 5))
    cube = dawis.wavelet_datacube(array, "BSPL")

    loose = dawis.hard_threshold(cube, n_sigmas=3).array
    tight = dawis.hard_threshold(cube, n_sigmas=5).array

    assert np.all(tight <= loose), (
        "monotonicity broken: raising the sigma cut from three to five "
        "flagged pixels that the looser cut had rejected"
    )


@pytest.mark.parametrize("keep_negative, expected", [(False, 0.0), (True, 1.0)])
def test_negative_peaks_are_flagged_only_when_asked_for(keep_negative, expected):
    """``keep_neg`` decides whether the cut is on value or on magnitude."""
    scale = 2
    array = synthetic_cube()
    array[40, 40, scale] = -thresholds(3)[scale] * 1.01
    cube = dawis.wavelet_datacube(array, "BSPL")

    support = dawis.hard_threshold(cube, n_sigmas=3, keep_neg=keep_negative)

    assert support.array[40, 40, scale] == expected, (
        f"with keep_neg={keep_negative} a negative peak one percent beyond "
        f"the threshold should have been flagged {expected}"
    )


def test_the_support_keeps_the_shape_of_the_cube_it_came_from():
    """One mask plane per wavelet plane, over the same pixels."""
    cube = dawis.wavelet_datacube(synthetic_cube(), "BSPL")

    support = dawis.hard_threshold(cube, n_sigmas=3)

    assert support.array.shape == CUBE_SHAPE, (
        f"the support has shape {support.array.shape}, but the cube it came "
        f"from has shape {CUBE_SHAPE}"
    )
