"""Tier 0 properties of the Anscombe variance-stabilising transform.

The Anscombe transform maps Poisson-Gaussian noise, whose spread grows with
the signal, onto noise of constant unit spread. These checks pin that
contract and the closed form it rests on.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.stats import sigma_clip

import dawis
from dawis.wavelets import AnscombeStabilisation


#: Noise parameters covering unit gain, amplification and attenuation.
NOISE_PARAMETERS = [
    (0.0, 0.0, 1.0),
    (2.5, 10.0, 1.0),
    (1.0, 0.0, 3.0),
    (0.0, -5.0, 0.5),
    (3.0, 20.0, 2.5),
]


@pytest.mark.parametrize("sigma, mean, gain", NOISE_PARAMETERS)
def test_the_transform_is_invertible(sigma, mean, gain):
    """Forward then back is the identity, to floating point precision.

    The forward map is ``y = (2 / g) * sqrt(g * x + 3 * g ** 2 / 8 + s ** 2 -
    g * m)``; :func:`dawis.inverse_anscombe` solves it for ``x``. The input
    values are well clear of the point where the square root turns negative,
    so the forward transform fills no pixels with noise draws and the
    roundtrip is exact.
    """
    rng = np.random.default_rng(20260823)
    image = rng.uniform(50.0, 500.0, (64, 64))

    transformed = dawis.anscombe_transform(image, sigma=sigma, mean=mean, gain=gain)
    restored = dawis.inverse_anscombe(transformed, sigma=sigma, mean=mean, gain=gain)

    np.testing.assert_allclose(
        restored,
        image,
        rtol=1e-10,
        atol=0,
        err_msg="the Anscombe roundtrip no longer returns the input, so the "
                "forward transform has stopped matching its closed form",
    )


@pytest.mark.parametrize("photons", [20, 200, 2000, 20000])
def test_poisson_noise_comes_out_with_unit_spread(photons):
    """Variance stabilisation: the spread stops depending on the signal."""
    rng = np.random.default_rng(20260823)
    counts = rng.poisson(photons, (200, 200)).astype(float)

    transformed = dawis.anscombe_transform(counts)

    assert np.std(transformed) == pytest.approx(1.0, abs=0.02), (
        "variance stabilisation broken: Poisson noise of mean "
        f"{photons} transformed to spread {np.std(transformed):.4f} instead "
        "of 1"
    )


def test_the_transform_preserves_order():
    """A brighter pixel stays brighter: the map is strictly increasing.

    The values start well clear of the point where the square root turns
    negative, which for these noise parameters is around ``x = 2``.
    """
    values = np.linspace(10.0, 1000.0, 500).reshape(1, -1)

    transformed = dawis.anscombe_transform(values, sigma=2.0, mean=5.0, gain=1.5)

    assert np.all(np.diff(transformed[0]) > 0), (
        "monotonicity broken: the Anscombe transform reordered pixel values"
    )


@pytest.mark.parametrize("gain", [1.0, 2.5])
@pytest.mark.parametrize("photons", [20, 500])
def test_estimated_noise_parameters_stabilise_the_noise(gain, photons):
    """The estimator returns the parameters that make the noise unit spread.

    That is the whole job of :func:`pg_noise_bissection` — not to recover the
    numbers it was given, but to find numbers under which the transformed
    background has unit spread.
    """
    rng = np.random.default_rng(20260823)
    background = gain * rng.poisson(photons, (128, 128)) + rng.normal(
        0.0, 2.0, (128, 128)
    )
    # A handful of background pixels fall below the point where the square
    # root turns negative, and the transform fills those with draws from the
    # global generator. Pin it so this check is reproducible.
    np.random.seed(20260823)

    sigma, mean, estimated_gain = dawis.pg_noise_bissection(background)
    transformed = dawis.anscombe_transform(background, sigma, mean, estimated_gain)
    spread = np.nanstd(sigma_clip(transformed, sigma=3, masked=False))

    assert spread == pytest.approx(1.0, abs=0.1), (
        "noise estimation broken: the estimated parameters left the "
        f"transformed background with spread {spread:.4f} instead of 1"
    )


def test_the_stabilisation_is_fitted_once_and_reused():
    """``AnscombeStabilisation`` is the map of one run, not of one array.

    The pipeline used to carry the offset and the three noise parameters as
    four loose locals and reapply them by hand every iteration (#49). Fitting
    against the input image and transforming a residual must give exactly what
    that did.
    """
    rng = np.random.default_rng(20260823)
    image = rng.normal(20.0, 3.0, (64, 64))
    residuals = rng.normal(0.0, 3.0, (64, 64))

    stabilisation = AnscombeStabilisation.fit(image, max_err=1e-4, n_sigmas=3)

    shifted = np.copy(image)
    offset = np.nanmin(shifted)
    shifted -= offset
    sigma, mean, gain = dawis.pg_noise_bissection(shifted, max_err=1e-4, n_sigmas=3)
    by_hand = dawis.anscombe_transform(residuals - offset, sigma, mean, gain)

    assert (stabilisation.offset, stabilisation.sigma,
            stabilisation.mean, stabilisation.gain) == (offset, sigma, mean, gain)
    np.testing.assert_array_equal(stabilisation.transform(residuals), by_hand)


def test_the_offset_puts_the_minimum_at_zero():
    """The transform takes a square root, so a negative pixel returns NaN."""
    rng = np.random.default_rng(20260823)
    image = rng.normal(-50.0, 2.0, (64, 64))

    stabilisation = AnscombeStabilisation.fit(image)

    assert stabilisation.offset == np.nanmin(image)
    assert np.isfinite(stabilisation.transform(image)).all()


def test_transforming_does_not_modify_the_image_it_is_given():
    """The residual the loop hands over is the array the loop keeps working on."""
    rng = np.random.default_rng(20260823)
    image = rng.normal(20.0, 3.0, (64, 64))
    residuals = rng.normal(0.0, 3.0, (64, 64))
    untouched = np.copy(residuals)

    AnscombeStabilisation.fit(image).transform(residuals)

    np.testing.assert_array_equal(residuals, untouched)
