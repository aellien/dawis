"""Tier 0 — levelling the image before the loop starts (#49).

Everything downstream of the first step assumes the noise is centred on zero:
the inpainting draws are, and the detection thresholds are quoted in sigmas of
it. ``subtract_background`` is the step that makes that true, and the two
numbers it returns are the ones the inpainting reads for the rest of the run.

It was three inlined lines in the pipeline before this ticket, including an
in-place subtraction whose in-placeness is load-bearing — see the dtype check
below.
"""

from __future__ import annotations

import numpy as np
import pytest

from dawis.restoration import sample_noise, subtract_background


def noisy_image(dtype=float, offset=100.0, sigma=2.0, seed=20260823):
    """Flat noise sitting on a constant background."""
    rng = np.random.default_rng(seed)
    return np.asarray(rng.normal(offset, sigma, (128, 128)), dtype=dtype)


def test_the_background_comes_out_of_the_image():
    image = noisy_image(offset=100.0, sigma=2.0)

    levelled, mean, sigma = subtract_background(image)

    # The measurement mirrors the low half of the histogram about its peak,
    # so it lands within a fraction of a sigma of the truth rather than on it.
    assert mean == pytest.approx(100.0, abs=1.0)
    assert sigma == pytest.approx(2.0, abs=1.0)
    assert levelled.mean() == pytest.approx(0.0, abs=1.0)


def test_the_callers_image_is_not_modified():
    """A run levels a copy, never the array it was handed (#54).

    The array-in entry point takes an image the caller still owns — a fixture,
    a slice of a mosaic, a read-only view — so subtracting the sky out of it
    would change what the caller holds.
    """
    image = noisy_image()
    before = np.copy(image)

    levelled, mean, _ = subtract_background(image)

    assert levelled is not image
    np.testing.assert_array_equal(image, before)
    np.testing.assert_allclose(levelled, before - mean)


def test_a_read_only_image_can_be_levelled():
    """The pipeline fixtures hand out read-only arrays on purpose."""
    image = noisy_image()
    image.setflags(write=False)

    levelled, _, _ = subtract_background(image)

    assert levelled.flags.writeable


def test_a_non_floating_image_is_converted_rather_than_refused():
    """An integer array cannot hold a fractional background (#54).

    Subtracting one in place raises rather than truncating, so the dtype
    warning the input diagnostics emit would be a warning about a run that
    dies anyway. It is converted instead, and the warning says so.
    """
    image = (noisy_image() * 10).astype(np.int32)

    levelled, mean, _ = subtract_background(image)

    assert levelled.dtype == np.float64
    np.testing.assert_allclose(levelled, image - mean)


def test_a_float32_image_stays_float32():
    """Out-of-place subtraction of a float64 mean would promote the image.

    Every number the run computes afterwards would move with it, which is
    exactly the silent change Phase A exists to avoid.
    """
    image = noisy_image(dtype=np.float32)

    levelled, _, _ = subtract_background(image)

    assert levelled.dtype == np.float32


def test_the_numbers_are_those_of_the_sampled_noise_distribution():
    """The mean and sigma are the moments of what ``sample_noise`` returns."""
    image = noisy_image()
    expected_pixels, _ = sample_noise(np.copy(image), n_sigmas=3, bins=200)

    _, mean, sigma = subtract_background(image, n_sigmas=3, bins=200)

    assert mean == np.mean(expected_pixels)
    assert sigma == np.std(expected_pixels)
