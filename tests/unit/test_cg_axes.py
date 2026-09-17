"""#100 — the line direction and beta as axes of the published solve.

Two things the conjugate gradient does are choices, and until now only one of
them was even visible. ``step_size`` — the beta formula, Fletcher-Reeves,
Polak-Ribière-Polyak, Hestenes-Stiefel or conjugate descent — has always been
an argument of :meth:`~dawis.trees.construction.InterscaleTree.CG_minimization`
and has never been a :class:`~dawis.config.Config` knob, so no run could set
it. The line direction was not an argument at all: ``v = fr + beta * old_fr``
was the only recursion in the tree.

#74 swept both. The published ``as_written`` recursion is ruled a documented
variant rather than a defect by #82 — the ``textbook`` form ``v = fr +
beta * v``, which is what both published CG algorithms on this problem do, is
not better on either metric on any class — so ``as_written`` stays the
default and ``textbook`` lands beside it. ``HS`` becomes the shipping beta,
which is the one place a default *moves*.

These tests are read against ``restoration_tests/arms/cg.py`` on
``research/restoration-tests``, the implementation every #74 and #92 number
was taken with. #3's goldens are ratcheted to the arm this map lands, so a
green regression suite is not evidence here; ``.research/100-cg-axes/`` is.
"""

from __future__ import annotations

import numpy as np
import pytest

from dawis.config import CG_LINE_DIRECTIONS, CG_STEP_SIZES
from dawis.exceptions import DawisValueError

#: The B3-spline kernel, as :func:`dawis.restoration.synthesis.restore_object`
#: writes it.
BSPL = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0

#: Enough steps that the two line directions have somewhere to part company.
#: They agree on the first update by construction — ``v`` starts as a copy of
#: ``fr`` — so a one-step solve cannot tell them apart.
STEPS = 6


@pytest.fixture(scope = 'module')
def a_tree():
    """One interscale tree and the cubes it sits on.

    The same field as #98's and #99's fixtures, so the three tickets are read
    on one tree.
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

    _, planes = atrous(image, n_levels = n_levels, filter = BSPL,
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
    tree = InterscaleTree(
        interscale_maximum = regions[-1],
        region_list = regions,
        wavelet_cube = wavelet_cube,
        label_cube = label_cube,
        det_err_array = np.zeros((32, 32, n_levels)),
        clip = 16,
    )

    return tree, wavelet_cube, label_cube


def solve(a_tree, max_iter = STEPS, **axes):
    """Run the arm on the fixture tree, with the axes under test set."""

    tree, wavelet_cube, label_cube = a_tree
    axes.setdefault('synthesis_operator', 'TRANSPOSE')
    return tree.CG_minimization(wavelet_cube, label_cube, filter = BSPL,
                                max_iter = max_iter, **axes)


# ---------------------------------------------------------------------------
# The line direction
# ---------------------------------------------------------------------------


def test_the_published_recursion_is_what_a_caller_gets_by_default(a_tree):
    """Every pre-#100 result was taken on ``v = fr + beta * old_fr``. Naming
    it must not be what selects it, or those results moved underneath their
    tables."""

    default = solve(a_tree)[0]
    as_written = solve(a_tree, line_direction = 'as_written')[0]

    assert np.max(np.abs(default - as_written)) == 0.0


def test_the_two_line_directions_agree_on_the_first_update(a_tree):
    """``v`` starts as a copy of ``fr``, so ``fr + beta * v`` and ``fr +
    beta * old_fr`` are the same vector at the first step. Any difference
    there would be a difference in something other than the axis.

    ``max_iter = 1`` and not 0: the published loop runs ``max_iter + 1``
    steps, so 0 runs one step and *discards* the ``v`` the branch built
    before anything reads it — an assertion that would hold whatever the
    branch did. At 1 the second step moves the reconstruction along that
    ``v``, which is the first place the two readings could differ and do
    not.
    """

    as_written = solve(a_tree, max_iter = 1, line_direction = 'as_written')[0]
    textbook = solve(a_tree, max_iter = 1, line_direction = 'textbook')[0]

    assert np.max(np.abs(textbook - as_written)) == 0.0


def test_the_two_line_directions_part_company_after_it(a_tree):
    """The second update is where ``old_fr`` and ``v`` stop being the same
    array, and it is the whole of what the axis selects."""

    as_written = solve(a_tree, line_direction = 'as_written')[0]
    textbook = solve(a_tree, line_direction = 'textbook')[0]

    assert not np.allclose(textbook, as_written)


def test_the_textbook_direction_is_the_textbook_recursion(a_tree):
    """Not merely *different* — the specific recursion #76 found in both
    published algorithms on this problem, replayed step for step outside the
    method.

    The solve is rerun here from its own returned pieces rather than
    predicted: what is asserted is that one line of the loop is
    ``v = fr + beta * v`` and everything around it is unchanged.
    """

    tree, wavelet_cube, label_cube = a_tree
    tube, support, _ = tree.tubes(wavelet_cube, label_cube)
    n_levels = tube.shape[2]

    from dawis.trees.construction import _analysis_tube

    def synthesise(cube):
        return tree.synthesis_transpose(cube, filter = BSPL)

    reconstructed = synthesise(tube)
    wr = tube - _analysis_tube(reconstructed, n_levels, BSPL, False) * support
    fr = synthesise(wr)
    v = np.copy(fr)
    total_flux = 1.0

    for _ in range(STEPS + 1):
        norm_wr = np.linalg.norm(wr)
        if norm_wr == 0:
            norm_wr = 1
        reconstructed = reconstructed + (np.linalg.norm(fr) / norm_wr) * v
        reconstructed[reconstructed < 0.0] = 0.0

        old_total_flux = total_flux or 1.0
        total_flux = np.sum(reconstructed)
        if abs(total_flux - old_total_flux) / old_total_flux < 1e-3:
            break

        old_fr = np.copy(fr)
        wr = tube - _analysis_tube(reconstructed, n_levels, BSPL,
                                   False) * support
        fr = synthesise(wr)

        squared_norm_old_fr = np.linalg.norm(old_fr) ** 2 or 1.0
        beta = np.linalg.norm(fr) ** 2 / squared_norm_old_fr

        v = fr + beta * v

    atom = solve(a_tree, line_direction = 'textbook', step_size = 'FR')[0]

    assert np.max(np.abs(atom - reconstructed)) == 0.0


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('step_size', CG_STEP_SIZES)
@pytest.mark.parametrize('line_direction', CG_LINE_DIRECTIONS)
def test_every_cell_of_the_two_axes_solves(a_tree, step_size, line_direction):
    """The eight cells #74 swept all run here, and all return an atom."""

    atom, sum_wr, norm_wr = solve(a_tree, step_size = step_size,
                                  line_direction = line_direction)

    assert atom.shape == (tree_size(a_tree), tree_size(a_tree))
    assert np.all(np.isfinite(atom))
    assert np.all(atom >= 0.0)
    assert np.isfinite(sum_wr) and np.isfinite(norm_wr)


def tree_size(a_tree):
    """The side of the fixture tree's box."""

    tree = a_tree[0]
    return tree.bbox[2] - tree.bbox[0]


def test_the_beta_formula_moves_the_atom(a_tree):
    """The shipping ``HS`` and the published ``FR`` are different arms, not
    two spellings of one. #82 chose between them on that basis."""

    published = solve(a_tree, step_size = 'FR')[0]
    shipping = solve(a_tree, step_size = 'HS')[0]

    assert not np.allclose(shipping, published)


def test_the_method_still_defaults_to_the_published_beta(a_tree):
    """The default *moves* on :class:`~dawis.config.Config`, not on the
    method. A positional caller of the published method keeps the published
    arm; what a run gets is #104's business."""

    default = solve(a_tree)[0]
    published = solve(a_tree, step_size = 'FR')[0]

    assert np.max(np.abs(default - published)) == 0.0


@pytest.mark.parametrize('axis, value', [
    ('step_size', 'FLETCHER'),
    ('step_size', 'fr'),
    ('line_direction', 'text_book'),
    ('line_direction', 'Textbook'),
    ('synthesis_operator', 'transpose'),
    ('synthesis_operator', 'ADJ'),
])
def test_an_axis_value_the_solver_has_no_branch_for_is_refused(a_tree, axis,
                                                               value):
    """And refused *before* the descent, not part-way through it.

    ``Config`` guards the configuration path, but the sweeps read this method
    directly: #74's and #92's arms call it cell by cell. An unrecognised line
    direction has a published form to fall through to, so a misspelt cell
    would file as-written numbers under ``textbook`` and nothing would say
    so. An unrecognised synthesis operator matches none of the three arms
    that build ``reconstructed``, so it surfaced as an ``UnboundLocalError``
    several frames from the cause (#108). All three axes are checked at the
    door instead, and the message names the value and what would have been
    accepted.
    """

    with pytest.raises(DawisValueError) as raised:
        solve(a_tree, max_iter = 1, **{axis: value})

    assert axis in str(raised.value)
    assert value in str(raised.value)
