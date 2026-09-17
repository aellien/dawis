"""#98 — the true adjoint of the à trous transform, as a third synthesis operator.

A tube is rebuilt into an image by a synthesis operator. ``SUM`` adds the
wavelet planes. ``ADJOINT`` smooths each plane as many times as its scale and
adds the results. Neither is the transpose of the transform the tube holds:
``SUM`` is the starlet's reconstruction dual, and ``ADJOINT`` is what the
published lineage adopted *after* the true transpose produced ring artefacts
([LU3CW9P8] p.272) — a rejection #85 measured on this instrument and did not
reproduce.

``TRANSPOSE`` is the genuine article, ported from
``restoration_tests/arms/adjoint.py`` on ``research/restoration-tests``.

What pins it is the inner-product identity

.. math:: \\langle W x, y \\rangle = \\langle x, W^{\\mathsf{T}} y \\rangle

which holds for the adjoint and for nothing else. #3's goldens are ratcheted
to the arm this map lands, so a green regression suite is not evidence that
this operator is right; the identity is.
"""

from __future__ import annotations

import numpy as np
import pytest

from dawis.restoration.synthesis import FILTERS
from dawis.wavelets import atrous, atrous_adjoint

# ``FILTERS`` is imported rather than transcribed since #103 gave the package
# one place that both names the shipped filters and builds them. A local copy
# would let this identity keep passing on a kernel the solver had stopped
# using.

#: The filters the identity is asserted on. Both shipped kernels are
#: **symmetric**, so on those two a gather and its transpose differ only at
#: the clamped edges — an identity that passed on them alone would not prove
#: the offsets are paired with the taps the way the forward gather pairs them.
#: ``ASYMMETRIC`` is not a filter DAWIS runs; it is there so the identity is
#: load-bearing for the property this module claims it pins.
IDENTITY_FILTERS = dict(
    FILTERS,
    ASYMMETRIC = np.array([0.05, 0.15, 0.4, 0.3, 0.1]),
)


def forward(image: np.ndarray, n_levels: int,
            filter: np.ndarray) -> np.ndarray:
    """The wavelet planes of an image — the transform a tube holds.

    ``'prolongation'`` is the boundary rule every call in the published solve
    passes, and it is the rule the transpose is written against: clamping
    duplicates the edge samples, so re-running the smoothing is *not* the
    transpose of it.
    """

    return atrous(image, n_levels = n_levels, filter = filter,
                  conditions = 'prolongation')[1]


# ---------------------------------------------------------------------------
# The operator is the transpose of the transform.


@pytest.mark.parametrize('filter_name', list(IDENTITY_FILTERS))
@pytest.mark.parametrize('shape', [(16, 16), (12, 21), (25, 9)])
@pytest.mark.parametrize('n_levels', [1, 3, 4])
def test_it_satisfies_the_inner_product_identity(filter_name, shape, n_levels):
    """``<Wx, y> = <x, W^T y>`` on both filters and non-square boxes.

    Non-square boxes are the case that separates a transpose from a
    transpose-with-the-axes-swapped: the forward smoothing applies rows then
    columns, so its transpose applies columns then rows, and a square box
    cannot see the difference.
    """

    filter = IDENTITY_FILTERS[filter_name]
    generator = np.random.default_rng(98)
    image = generator.normal(0.0, 1.0, shape)
    tube = generator.normal(0.0, 1.0, shape + (n_levels,))

    analysed = np.sum(forward(image, n_levels, filter) * tube)
    synthesised = np.sum(image * atrous_adjoint(tube, filter))

    assert analysed == pytest.approx(synthesised, rel = 1e-4)


@pytest.mark.parametrize('filter_name', ['BSPL', 'HAAR'])
@pytest.mark.parametrize('n_levels', [1, 3])
def test_the_identity_holds_with_the_coarse_channel_too(filter_name,
                                                        n_levels):
    """The same identity for the transform that also returns ``c_{L+1}``.

    #99 is what turns this channel on in the pipeline. The transpose is
    written for it now because the channel is a column of :math:`W`, not a
    wavelet plane, and leaving it out would make ``TRANSPOSE`` the transpose
    of a transform the tube no longer holds.
    """

    filter = FILTERS[filter_name]
    shape = (14, 19)
    generator = np.random.default_rng(99)
    image = generator.normal(0.0, 1.0, shape)
    tube = generator.normal(0.0, 1.0, shape + (n_levels + 1,))

    coarse_array, wavelet_array = atrous(image, n_levels = n_levels,
                                         filter = filter,
                                         conditions = 'prolongation')
    analysed = np.sum(wavelet_array * tube[:, :, :n_levels])
    analysed += np.sum(coarse_array[:, :, n_levels] * tube[:, :, n_levels])

    synthesised = np.sum(image * atrous_adjoint(tube, filter, coarse = True))

    assert analysed == pytest.approx(synthesised, rel = 1e-4)


def test_an_even_filter_is_refused_rather_than_transposed_wrongly():
    """An even filter has no centre, and the scatter would drop a tap.

    :func:`~dawis.wavelets.atrous` fails on the same filter at its
    ``matmul``. This fails at the same place, with the reason named.
    """

    from dawis.exceptions import DawisValueError
    from dawis.wavelets import smooth_adjoint

    with pytest.raises(DawisValueError):
        smooth_adjoint(np.zeros((8, 8)), np.ones(4) / 4.0, 0)


def test_a_two_dimensional_filter_is_refused():
    """:func:`~dawis.wavelets.atrous` takes one; the transpose does not.

    Unguarded, a 2D filter whose row length happens to match the image would
    broadcast against it and return a wrong array with no error at all.
    """

    from dawis.exceptions import DawisDimensionError
    from dawis.wavelets import smooth_adjoint

    with pytest.raises(DawisDimensionError):
        smooth_adjoint(np.zeros((8, 8)), np.ones((5, 5)) / 25.0, 0)


def test_a_tube_with_no_wavelet_plane_is_refused():
    """One channel and ``coarse`` True leaves no plane behind the channel.

    The recursion would then read that channel twice, the second time at
    level ``-1``, where ``2 ** -1`` collapses the stencil onto duplicated
    taps — and it would return an array rather than fail.
    """

    from dawis.exceptions import DawisValueError

    with pytest.raises(DawisValueError):
        atrous_adjoint(np.zeros((8, 8, 1)), FILTERS['BSPL'], coarse = True)


# ---------------------------------------------------------------------------
# It is a third value of the synthesis_operator axis.


@pytest.fixture(scope = 'module')
def a_tree():
    """One interscale tree over a small field, and the cubes it sits on.

    Three nested regions, one per wavelet plane, around a single Gaussian
    source on a noisy background — the shape a tree has when the detection
    block hands one to the restoration.
    """

    from dawis.cubes import LabelCube, WaveletCube
    from dawis.detection import region
    from dawis.trees import InterscaleTree

    shape = (32, 32)
    n_levels = 3
    rows, columns = np.indices(shape).astype(float)
    image = 400.0 * np.exp(
        -((rows - 16.0) ** 2 + (columns - 16.0) ** 2) / (2.0 * 3.0 ** 2))
    image += np.random.default_rng(84).normal(10.0, 1.0, shape)

    planes = forward(image, n_levels, FILTERS['BSPL'])
    labels = np.zeros(shape + (n_levels,))

    regions = []
    for level, (low, high) in enumerate(((13, 19), (11, 21), (9, 23))):
        coordinates = np.array([(x, y) for x in range(low, high)
                                for y in range(low, high)])
        labels[low:high, low:high, level] = 1
        regions.append(region(label = 1, level = level, coords = coordinates,
                              x_max = (low + high) // 2,
                              y_max = (low + high) // 2,
                              eccentricity = 0.1,
                              area = coordinates.shape[0],
                              bbox = (low, low, high, high)))

    wavelet_cube = WaveletCube(planes.astype(float), 'BSPL')
    label_cube = LabelCube(labels, lab_counts = [1, 1, 1])
    tree = InterscaleTree(
        interscale_maximum = regions[-1],
        region_list = regions,
        wavelet_cube = wavelet_cube,
        label_cube = label_cube,
        det_err_array = np.zeros((32, 32, n_levels)),
        clip = 16,
    )

    return tree, wavelet_cube, label_cube


def test_the_operator_is_not_an_alias_of_either_published_one(a_tree):
    """``TRANSPOSE`` rebuilds a tube into something neither of the others does.

    Asserted on a tree's own tube rather than on a full-support problem: a
    **sparse** support is what the operator meets in the pipeline, and it is
    where an operator that is not an adjoint differs most from one that is.
    """

    tree, wavelet_cube, label_cube = a_tree
    wavelet_tube = tree.tubes(wavelet_cube, label_cube)[0]
    filter = FILTERS['BSPL']

    transpose = tree.synthesis_transpose(wavelet_tube, filter = filter)

    assert not np.allclose(transpose, tree.synthesis_sum(wavelet_tube))
    assert not np.allclose(transpose,
                           tree.synthesis_adjoint(wavelet_tube,
                                                  filter = filter))


def test_the_axis_takes_the_third_value(a_tree):
    """``CG_minimization`` solves with it, and lands somewhere of its own."""

    tree, wavelet_cube, label_cube = a_tree
    filter = FILTERS['BSPL']

    solved = {}
    for operator in ('SUM', 'ADJOINT', 'TRANSPOSE'):
        reconstructed, sum_wr, norm_wr = tree.CG_minimization(
            wavelet_cube, label_cube, filter = filter,
            synthesis_operator = operator, max_iter = 20)
        assert np.all(np.isfinite(reconstructed))
        assert np.all(reconstructed >= 0.0)
        assert np.isfinite(sum_wr) and np.isfinite(norm_wr)
        solved[operator] = reconstructed

    assert not np.allclose(solved['TRANSPOSE'], solved['SUM'])
    assert not np.allclose(solved['TRANSPOSE'], solved['ADJOINT'])
