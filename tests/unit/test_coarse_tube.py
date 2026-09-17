"""#99 — the coarse term as a tube axis.

A tree's tube carries its wavelet planes. #89 measured that the pipeline's
planes do not telescope — ``w_j = c_j - H_j^2 c_j`` — so the level-``L+1``
smooth plane cannot be recovered by adding the planes above ``L`` back onto
the last one. The coarse term therefore comes from the smoothing chain
itself, and the chain is already there: ``ser_iuwt_decomposition`` walks
``c_0, c_1, ... c_d`` and returns only the last.

``ser_iuwt_cascade`` returns all of them. Keeping them costs no extra
smoothing, which is what decides whether #106 reads a cost or a saving.

What the coarse channel *is* is fixed by #88: :math:`c_{L+1}` cut to the
tree's box and **masked by the tree's support at level L**. The unmasked
plane over that box is sky, intracluster light and neighbours, and DAWIS
subtracts every atom from the field, so an atom carrying that flux removes
the light the pipeline exists to keep.

These tests are read against ``restoration_tests/fixtures/build.py`` on
``research/restoration-tests``, the implementation every #74 and #92 number
was taken with. #3's goldens are ratcheted to the arm this map lands, so a
green regression suite is not evidence here.
"""

from __future__ import annotations

import numpy as np
import pytest

from dawis.cubes import CoarseCube
from dawis.wavelets import (
    bspl_atrous,
    ser_iuwt_cascade,
    ser_iuwt_decomposition,
)

#: Deep enough that a tree's ``L + 1`` is a plane in the middle of the chain
#: rather than its last, which is the case the pipeline actually runs.
DEPTH = 5


def probe_image() -> np.ndarray:
    """A non-square, non-symmetric image, in float64 as the transform is."""

    rng = np.random.default_rng(99)
    return (rng.random((64, 48)) * 100.0).astype(np.float64)


def test_cascade_plane_equals_the_decomposition_run_to_that_depth():
    """``c_j`` is what a decomposition of depth ``j`` returns as its coarse
    plane — which is what ``restoration_tests`` pays ``n_levels`` separate
    decompositions to collect."""

    image = probe_image()
    _, cascade = ser_iuwt_cascade(image, DEPTH, 0)

    for depth in range(1, DEPTH + 1):
        _, coarse_plane = ser_iuwt_decomposition(
            image, depth, 0, store_smoothed = True)

        # ``ser_iuwt_decomposition`` returns its coarse plane transposed
        # relative to its wavelet planes; the cascade does not, so the
        # reference is swapped back rather than the new output bent to the
        # wart.
        assert np.max(np.abs(cascade[:, :, depth]
                             - np.swapaxes(coarse_plane, 0, 1))) == 0.0


def test_cascade_starts_at_the_image():
    """``c_0`` is the image, unsmoothed — so plane ``j`` is the image
    smoothed ``j`` times and the index needs no offset table to read."""

    image = probe_image()
    _, cascade = ser_iuwt_cascade(image, DEPTH, 0)

    assert cascade.shape == (64, 48, DEPTH + 1)
    assert np.max(np.abs(cascade[:, :, 0] - image)) == 0.0


def test_cascade_leaves_the_wavelet_planes_alone():
    """The planes the pipeline detects on are the ones it detected on
    before. The cascade is additive or it is a regression."""

    image = probe_image()
    wavelets, _ = ser_iuwt_cascade(image, DEPTH, 0)
    reference = ser_iuwt_decomposition(image, DEPTH, 0, store_smoothed = False)

    assert np.max(np.abs(wavelets - reference)) == 0.0


def test_bspl_atrous_hands_back_the_cascade_as_a_coarse_cube():
    """The pipeline's own decomposition is where the chain has to come from:
    computing it a second time somewhere else would pay the smoothing twice
    and could drift from the planes the trees were detected on."""

    image = probe_image()
    coarse_cube, wavelet_cube = bspl_atrous(image, DEPTH, coarse = True)

    assert isinstance(coarse_cube, CoarseCube)
    assert coarse_cube.wavelet_type == 'BSPL'

    _, cascade = ser_iuwt_cascade(image, DEPTH, 0)
    assert np.max(np.abs(coarse_cube.array - cascade)) == 0.0


def test_bspl_atrous_wavelet_planes_do_not_move_when_the_cascade_is_asked_for():
    """Every tree in the run is detected on these planes. The coarse term is
    additive or it is a change to detection, which this ticket is not."""

    image = probe_image()
    plain = bspl_atrous(image, DEPTH)
    _, with_cascade = bspl_atrous(image, DEPTH, coarse = True)

    assert np.max(np.abs(plain.array - with_cascade.array)) == 0.0


# ---------------------------------------------------------------------------
# The channel on the tube.

#: The B3-spline kernel, as :func:`dawis.restoration.synthesis.restore_object`
#: writes it.
BSPL = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


@pytest.fixture(scope = 'module')
def a_tree():
    """One interscale tree, the cubes it sits on, and the smooth cascade.

    Three nested regions, one per wavelet plane, around a single Gaussian
    source on a noisy background — the shape a tree has when the detection
    block hands one to the restoration. The same field as #98's fixture, so
    the two tickets are read on one tree.
    """

    from dawis.cubes import LabelCube, WaveletCube
    from dawis.detection import region
    from dawis.trees import InterscaleTree
    from dawis.wavelets import atrous

    shape = (32, 32)
    n_levels = 3
    rows, columns = np.indices(shape).astype(float)
    image = 400.0 * np.exp(
        -((rows - 16.0) ** 2 + (columns - 16.0) ** 2) / (2.0 * 3.0 ** 2))
    image += np.random.default_rng(84).normal(10.0, 1.0, shape)

    smooth, planes = atrous(image, n_levels = n_levels, filter = BSPL,
                            conditions = 'prolongation')
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
    coarse_cube = CoarseCube(smooth.astype(float), 'BSPL')
    tree = InterscaleTree(
        interscale_maximum = regions[-1],
        region_list = regions,
        wavelet_cube = wavelet_cube,
        label_cube = label_cube,
        det_err_array = np.zeros((32, 32, n_levels)),
        clip = 16,
    )

    return tree, wavelet_cube, label_cube, coarse_cube


def test_the_channel_is_appended_and_nothing_below_it_moves(a_tree):
    """A tube of a tree at level ``L`` carries ``L + 2`` planes, and the
    ``L + 1`` below the channel are the ones a plain cut writes.

    Additive or it is a change to every result taken on a plain tube.
    """

    tree, wavelet_cube, label_cube, coarse_cube = a_tree
    level = tree.interscale_maximum.level

    plain, plain_support, plain_labels = tree.tubes(wavelet_cube, label_cube)
    wide, wide_support, wide_labels = tree.tubes(
        wavelet_cube, label_cube, coarse_cube = coarse_cube)

    assert plain.shape[2] == level + 1
    assert wide.shape[2] == level + 2
    assert np.max(np.abs(wide[:, :, :level + 1] - plain)) == 0.0
    assert np.max(np.abs(wide_support[:, :, :level + 1] - plain_support)) == 0.0
    assert np.max(np.abs(wide_labels - plain_labels)) == 0.0


def test_the_channel_is_the_level_above_the_interscale_maximum(a_tree):
    """:math:`c_{L+1}`, not the coarse plane of the whole decomposition.

    ``CoarseCube`` plane ``j`` is the image smoothed ``j`` times, so a tree
    at level ``L`` reads plane ``L + 1``. The last plane of the cube is the
    smooth remainder of the *full* depth and would leave every scale between
    ``L`` and the last out of the problem (#89).
    """

    tree, wavelet_cube, label_cube, coarse_cube = a_tree
    level = tree.interscale_maximum.level
    row_min, col_min, row_max, col_max = tree.bbox

    wide, wide_support, _ = tree.tubes(
        wavelet_cube, label_cube, coarse_cube = coarse_cube)
    channel = wide[:, :, level + 1]
    support = wide_support[:, :, level]

    expected = coarse_cube.array[row_min:row_max, col_min:col_max, level + 1]
    assert np.max(np.abs(channel[support == 1]
                         - expected[support == 1])) == 0.0


def test_the_channel_is_masked_by_the_support_at_level_L(a_tree):
    """Off the tree's support the channel is exactly zero.

    This is the constraint #88 settled, and it is not cosmetic. The coarse
    plane over a tree's box is sky, intracluster light and neighbours; DAWIS
    subtracts each atom from the field, so an atom carrying that flux removes
    the light the pipeline exists to keep.
    """

    tree, wavelet_cube, label_cube, coarse_cube = a_tree
    level = tree.interscale_maximum.level

    wide, wide_support, _ = tree.tubes(
        wavelet_cube, label_cube, coarse_cube = coarse_cube)
    support = wide_support[:, :, level]

    assert np.any(support == 0)
    assert np.all(wide[:, :, level + 1][support == 0] == 0.0)
    assert np.any(wide[:, :, level + 1][support == 1] != 0.0)


def test_the_support_repeats_plane_L_behind_the_channel(a_tree):
    """The mask the channel was cut with is the mask the solve has to keep
    obeying, every iteration, rather than one applied once at cut time."""

    tree, wavelet_cube, label_cube, coarse_cube = a_tree
    level = tree.interscale_maximum.level

    _, wide_support, _ = tree.tubes(
        wavelet_cube, label_cube, coarse_cube = coarse_cube)

    assert wide_support.shape[2] == level + 2
    assert np.max(np.abs(wide_support[:, :, level + 1]
                         - wide_support[:, :, level])) == 0.0


# ---------------------------------------------------------------------------
# The channel through the synthesis operators and the solve.


def zero_channel_tube(tree, wavelet_cube, label_cube):
    """A coarse-carrying tube whose channel is identically zero.

    Cut with a coarse cube of zeros, so the tube is the plain tube with one
    empty channel behind it. Every synthesis operator must then return the
    plain tube's image *exactly*: the channel enters the arithmetic only
    through its values, and an operator that miscounted the levels behind it
    would move the wavelet part too.
    """

    blank = CoarseCube(np.zeros(wavelet_cube.array.shape[:2]
                                + (wavelet_cube.array.shape[2] + 1,)), 'BSPL')

    return (tree.tubes(wavelet_cube, label_cube)[0],
            tree.tubes(wavelet_cube, label_cube, coarse_cube = blank)[0])


def test_an_empty_channel_changes_no_synthesis_operator(a_tree):
    """All three operators, on the same tree."""

    tree, wavelet_cube, label_cube, _ = a_tree
    plain, wide = zero_channel_tube(tree, wavelet_cube, label_cube)

    assert np.max(np.abs(tree.synthesis_sum(wide)
                         - tree.synthesis_sum(plain))) == 0.0
    assert np.max(np.abs(
        tree.synthesis_adjoint(wide, filter = BSPL, coarse = True)
        - tree.synthesis_adjoint(plain, filter = BSPL))) == 0.0
    assert np.max(np.abs(
        tree.synthesis_transpose(wide, filter = BSPL, coarse = True)
        - tree.synthesis_transpose(plain, filter = BSPL))) == 0.0


def test_a_filled_channel_does_change_them(a_tree):
    """The guard on the test above: an operator that ignored the channel
    entirely would also pass it."""

    tree, wavelet_cube, label_cube, coarse_cube = a_tree
    plain = tree.tubes(wavelet_cube, label_cube)[0]
    wide = tree.tubes(wavelet_cube, label_cube,
                      coarse_cube = coarse_cube)[0]

    assert not np.allclose(tree.synthesis_sum(wide),
                           tree.synthesis_sum(plain))
    assert not np.allclose(
        tree.synthesis_adjoint(wide, filter = BSPL, coarse = True),
        tree.synthesis_adjoint(plain, filter = BSPL))
    assert not np.allclose(
        tree.synthesis_transpose(wide, filter = BSPL, coarse = True),
        tree.synthesis_transpose(plain, filter = BSPL))


def test_the_adjoint_smooths_the_channel_with_the_filters_that_built_it(a_tree):
    """``ADJOINT`` smooths plane :math:`j` :math:`j` times. The channel is
    smoothed :math:`L + 1` times, by the same filters that built it — rather
    than by an :math:`H_{L+1}` that appears nowhere in the transform (#89).
    """

    tree, wavelet_cube, label_cube, coarse_cube = a_tree
    wide = tree.tubes(wavelet_cube, label_cube,
                      coarse_cube = coarse_cube)[0]
    n_levels = wide.shape[2] - 1

    from dawis.wavelets import atrous

    expected = tree.synthesis_adjoint(wide[:, :, :n_levels], filter = BSPL)
    expected = expected + atrous(
        wide[:, :, n_levels], n_levels = n_levels, filter = BSPL,
        conditions = 'prolongation')[0][:, :, n_levels]

    assert np.max(np.abs(
        tree.synthesis_adjoint(wide, filter = BSPL, coarse = True)
        - expected)) == 0.0


@pytest.mark.parametrize('operator', ['SUM', 'ADJOINT', 'TRANSPOSE'])
def test_the_solve_runs_on_a_tube_that_carries_the_channel(a_tree, operator):
    """The whole arm, end to end, on every synthesis operator.

    That it runs at all is the level bookkeeping: the analysis inside the
    loop has to ask for ``L + 1`` wavelet planes and put the smooth plane
    behind them. Asking for ``L + 2`` would build an ``L + 3`` channel tube
    and fail to broadcast against the ``L + 2`` one it is subtracted from.
    """

    tree, wavelet_cube, label_cube, coarse_cube = a_tree

    atom, sum_wr, norm_wr = tree.CG_minimization(
        wavelet_cube, label_cube, filter = BSPL,
        synthesis_operator = operator, coarse_cube = coarse_cube)

    assert atom.shape == (tree.bbox[2] - tree.bbox[0],
                          tree.bbox[3] - tree.bbox[1])
    assert np.all(np.isfinite(atom))
    assert np.all(atom >= 0.0)
    assert np.isfinite(sum_wr) and np.isfinite(norm_wr)


def test_the_solve_without_a_coarse_cube_is_the_published_one(a_tree):
    """Every #74 and #92 number on a coarse-off arm was taken on this path.
    The channel is opt-in or those numbers moved underneath their tables."""

    tree, wavelet_cube, label_cube, coarse_cube = a_tree

    off = tree.CG_minimization(wavelet_cube, label_cube, filter = BSPL,
                               synthesis_operator = 'TRANSPOSE')[0]
    on = tree.CG_minimization(wavelet_cube, label_cube, filter = BSPL,
                              synthesis_operator = 'TRANSPOSE',
                              coarse_cube = coarse_cube)[0]

    assert not np.allclose(on, off)


def test_the_reported_residual_is_the_residual_of_the_problem_solved(a_tree):
    """``sum_wr`` and ``norm_wr`` are recomputed once after the loop, and the
    channel has to be the coarse plane of the atom there too.

    Counting the tube's channels as wavelet planes gives an array of the
    right *shape* — so it does not crash — carrying the atom's wavelet plane
    :math:`L+1` where its smooth plane :math:`c_{L+1}` belongs. The atom is
    unaffected, because the loop is already over; what moves is the number
    the pipeline reads to judge the atom.
    """

    from dawis.wavelets import atrous

    tree, wavelet_cube, label_cube, coarse_cube = a_tree
    tube, support, _ = tree.tubes(wavelet_cube, label_cube,
                                  coarse_cube = coarse_cube)
    n_levels = tube.shape[2] - 1

    atom, sum_wr, norm_wr = tree.CG_minimization(
        wavelet_cube, label_cube, filter = BSPL,
        synthesis_operator = 'TRANSPOSE', coarse_cube = coarse_cube)

    smooth, wavelets = atrous(atom, n_levels = n_levels, filter = BSPL,
                              conditions = 'prolongation')
    analysis = np.concatenate(
        (wavelets, smooth[:, :, n_levels][:, :, None]), axis = 2)
    expected = tube - analysis * support

    assert sum_wr == np.sum(expected)
    assert norm_wr == np.sqrt(np.sum(expected ** 2))


# ---------------------------------------------------------------------------
# The two ways a caller can hand `tubes` the wrong coarse cube.


def test_a_coarse_cube_of_the_wrong_depth_is_refused(a_tree):
    """A second plane convention is live in this project.

    ``restoration_tests.fixtures.mock.MockIteration.coarse_cube`` holds
    ``n_levels`` planes with plane ``L`` *already* equal to :math:`c_{L+1}`.
    Read with this method's convention, a tree at level ``L`` would take
    :math:`c_{L+2}` — a coarse term one scale too smooth — and only the
    deepest tree of the run would raise. The failure has to be loud for
    every tree or it is worse than no check.
    """

    from dawis.exceptions import DawisValueError

    tree, wavelet_cube, label_cube, coarse_cube = a_tree
    short = CoarseCube(coarse_cube.array[:, :, 1:], 'BSPL')

    with pytest.raises(DawisValueError):
        tree.tubes(wavelet_cube, label_cube, coarse_cube = short)


def test_a_coarse_cube_from_another_basis_is_refused(a_tree):
    """The channel is data the solve's forward model is compared against, so
    it has to be built by the basis the solve smooths with.

    ``dawis.restoration.synthesis.restore_object`` runs the low-extent branch
    on Haar. A Haar solve against a B3-built channel compares a Haar-smoothed
    model with a B3-smoothed datum, and the two disagree by about 10% — an
    error that lands on the smooth term's amplitude rather than a detail's.
    """

    from dawis.exceptions import DawisValueError

    tree, wavelet_cube, label_cube, coarse_cube = a_tree
    other = CoarseCube(coarse_cube.array, 'HAAR')

    with pytest.raises(DawisValueError):
        tree.tubes(wavelet_cube, label_cube, coarse_cube = other)
