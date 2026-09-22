"""#102 — the minimizer axis, and the three proximal arms that fill it.

Until now ``dawis/`` had one algorithm for solving a tree: the published
conjugate gradient of Ellien+2021. #74 measured four more against it on one
fixture set and #82 kept the axis user-settable with **no shortlist**, so
FISTA, projected gradient and Condat-Vu land beside it —
:func:`dawis.trees.minimize` is the one door over all four.

The port itself is proved bit-exact against
``restoration_tests/arms/proximal.py`` on ``research/restoration-tests``, the
implementation every #74, #81 and #92 number was taken with; #3's goldens are
ratcheted to the arm this map lands, so a green regression suite is not the
evidence for that and ``.research/102-minimizer-axis/`` is.

What is pinned **here** is what that proof cannot see, because the reference
implementation has the same shape:

* the two tables that say which minimizers exist, and which axes each one
  carries, agree with each other;
* the door adds no layer to the path a run already takes — ``'bijaoui_rue'``
  is :meth:`~dawis.trees.construction.InterscaleTree.CG_minimization` with its
  own published defaults;
* every value of the axis reaches a different algorithm;
* an axis a minimizer does not carry is **refused**, not quietly dropped —
  the failure #96 has met three times already;
* the measured step errs on the side that keeps the convergence argument
  (#86), which is the one number in the proximal arms that can be wrong in a
  direction no equality test would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from dawis.config import MINIMIZERS
from dawis.exceptions import DawisValueError
from dawis.trees import minimize
from dawis.trees.minimizers import (
    DEFAULT_DIVERGENCE_FACTOR,
    DEFAULT_STEP_SAFETY,
    MINIMIZER_AXES,
    _finish,
    spectral_radius,
)

#: The B3-spline kernel, as :func:`dawis.restoration.synthesis.restore_object`
#: writes it.
BSPL = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0

#: Enough steps that the four minimizers have somewhere to part company, and
#: few enough that a unit test is a unit test. The proximal arms take smaller
#: steps than the conjugate gradient, so they need more of them than #100's
#: axes did to separate.
STEPS = 12


@pytest.fixture(scope = 'module')
def a_tree():
    """One interscale tree and the cubes it sits on.

    The same field as #98's, #99's and #100's fixtures, so the four tickets
    are read on one tree.
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


@pytest.fixture(scope = 'module')
def a_coarse_cube(a_tree):
    """The smooth planes of the same decomposition the tree was built on.

    #99's own path: the coarse term is the decomposition's running smooth
    value, one plane deeper than the wavelet cube, not a second decomposition
    arranged for a test.
    """

    from dawis.cubes import CoarseCube
    from dawis.wavelets import atrous

    _, wavelet_cube, _ = a_tree
    shape = wavelet_cube.array.shape[:2]
    n_levels = wavelet_cube.array.shape[2]

    rows, columns = np.indices(shape).astype(float)
    image = 400.0 * np.exp(
        -((rows - 16.0) ** 2 + (columns - 16.0) ** 2) / (2.0 * 3.0 ** 2))
    image += np.random.default_rng(84).normal(10.0, 1.0, shape)

    smooth, _ = atrous(image, n_levels = n_levels, filter = BSPL,
                       conditions = 'prolongation')

    return CoarseCube(smooth.astype(float), 'BSPL')


def solve(a_tree, minimizer, max_iter = STEPS, **axes):
    """Run one minimizer on the fixture tree, through the door."""

    tree, wavelet_cube, label_cube = a_tree
    return minimize(tree, wavelet_cube, label_cube, filter = BSPL,
                    minimizer = minimizer, max_iter = max_iter, **axes)


# ---------------------------------------------------------------------------
# The two tables
# ---------------------------------------------------------------------------


def test_the_axis_table_names_exactly_the_minimizers_that_exist():
    """:data:`~dawis.config.MINIMIZERS` is stated in ``dawis.config``, which
    imports no block of the package, and :data:`MINIMIZER_AXES` is stated
    beside the branches it describes. The two are therefore written twice and
    can disagree, which is what this asserts they do not — a minimizer in one
    and not the other is either a value the door cannot dispatch or an axis
    table nothing reads."""

    assert set(MINIMIZER_AXES) == set(MINIMIZERS)


def test_only_the_conjugate_gradient_carries_the_conjugate_gradient_axes():
    """The line direction and beta are readings of the CG recursion. No
    proximal arm has a conjugate direction, so neither axis means anything on
    one, and a table that handed them one would be describing a knob the solve
    cannot honour."""

    for minimizer, axes in MINIMIZER_AXES.items():
        carries_cg_axes = {'step_size', 'line_direction'} <= set(axes)
        assert carries_cg_axes == (minimizer == 'bijaoui_rue')


def test_the_public_carrier_table_agrees_with_the_solver_table():
    """:data:`~dawis.config.AXIS_CARRIERS` against :data:`MINIMIZER_AXES`.

    The same fact in two shapes, written twice for the reason the tuple above
    is: ``dawis.config`` is shared vocabulary and imports no block of the
    package. One is the rule a knob is refused against at construction and
    the other is the rule the solve settles an axis with, and a run that
    passed the first and failed the second would be a configuration the
    package accepted and could not run.

    Only the three optional axes are compared. Minimizer and filter are
    carried by every arm, so there is nothing for a carrier table to say
    about them.
    """

    from dawis.config import AXIS_CARRIERS

    # Both directions. Iterating the public table alone would pass an axis
    # added to the solver's and forgotten here, and `Config._check_arm` would
    # then never refuse that axis on a minimizer that has not got it -- the
    # accepted-and-unhonoured knob this pair of tables exists to prevent.
    assert set(AXIS_CARRIERS) == {axis
                                  for axes in MINIMIZER_AXES.values()
                                  for axis in axes}

    for axis, carriers in AXIS_CARRIERS.items():
        assert set(carriers) == {minimizer
                                 for minimizer, axes in MINIMIZER_AXES.items()
                                 if axis in axes}


def test_condat_vu_carries_no_synthesis_operator():
    """The whole point of #74's note on this arm: a primal-dual method moves
    between the two spaces with the genuine transpose and never with a
    synthesis operator, so it dissolves the ``SUM``/``ADJOINT`` fork rather
    than making a choice within it."""

    assert 'synthesis_operator' not in MINIMIZER_AXES['condat_vu']


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------


def test_the_default_minimizer_is_the_published_solve_untouched(a_tree):
    """Every pre-#102 result was taken on ``CG_minimization`` called directly.
    The door has to be that call and not a reimplementation of it, or those
    results moved underneath their tables the moment a caller went through
    it."""

    tree, wavelet_cube, label_cube = a_tree
    direct = tree.CG_minimization(wavelet_cube, label_cube, filter = BSPL,
                                  max_iter = STEPS)
    through_the_door = solve(a_tree, 'bijaoui_rue')

    assert np.max(np.abs(through_the_door[0] - direct[0])) == 0.0
    assert through_the_door[1] == direct[1]
    assert through_the_door[2] == direct[2]


def test_the_door_defaults_to_the_published_minimizer(a_tree):
    """Naming ``'bijaoui_rue'`` must not be what selects it. The default that
    **moves** is #104's business, on ``Config``, not this door's."""

    named = solve(a_tree, 'bijaoui_rue')[0]

    tree, wavelet_cube, label_cube = a_tree
    unnamed = minimize(tree, wavelet_cube, label_cube, filter = BSPL,
                       max_iter = STEPS)[0]

    assert np.max(np.abs(unnamed - named)) == 0.0


@pytest.mark.parametrize('minimizer', MINIMIZERS)
def test_every_minimizer_solves_the_tree(minimizer, a_tree):
    """An axis whose values are accepted and dispatched to nothing is the
    defect this whole ticket is the other half of. Every value has to come
    back with an atom the tree's own shape, non-negative, and finite on a cell
    that is not one of the diverging ones."""

    tree, _, _ = a_tree
    operator = None if minimizer == 'condat_vu' else 'TRANSPOSE'
    atom, sum_wr, norm_wr = solve(a_tree, minimizer,
                                  synthesis_operator = operator)

    assert atom.shape == (tree.bbox[2] - tree.bbox[0],
                          tree.bbox[3] - tree.bbox[1])
    assert np.all(atom >= 0.0)
    assert np.all(np.isfinite(atom))
    assert np.isfinite(sum_wr) and np.isfinite(norm_wr)


def test_the_four_minimizers_are_four_algorithms(a_tree):
    """Not merely four accepted strings. Four solves of one tree that agreed
    would mean the door dispatched on nothing, and the equality proof against
    the research arms would pass just as happily."""

    atoms = {}
    for minimizer in MINIMIZERS:
        operator = None if minimizer == 'condat_vu' else 'TRANSPOSE'
        atoms[minimizer] = solve(a_tree, minimizer,
                                 synthesis_operator = operator)[0]

    for one in MINIMIZERS:
        for other in MINIMIZERS:
            if one < other:
                assert np.max(np.abs(atoms[one] - atoms[other])) > 0.0


def test_the_momentum_is_what_separates_the_two_forward_backward_arms(a_tree):
    """FISTA and projected gradient are one loop differing in one line, so a
    port that dropped the extrapolation would still be a working solver and
    still pass every shape and finiteness check above."""

    with_momentum = solve(a_tree, 'fista', synthesis_operator = 'TRANSPOSE')[0]
    without = solve(a_tree, 'projected_gradient',
                    synthesis_operator = 'TRANSPOSE')[0]

    assert np.max(np.abs(with_momentum - without)) > 0.0


# ---------------------------------------------------------------------------
# What is refused at the door
# ---------------------------------------------------------------------------


def test_an_unknown_minimizer_is_refused(a_tree):
    """Including ``'wolfe'``, which #74 measured and #20 deleted from the 1.0
    tree. A value that names no branch used to be the caller's problem several
    frames later; here it is refused before anything is solved."""

    with pytest.raises(DawisValueError) as refusal:
        solve(a_tree, 'wolfe')

    assert 'wolfe' in str(refusal.value)
    assert 'bijaoui_rue' in str(refusal.value)


@pytest.mark.parametrize('axis, value', [
    ('synthesis_operator', 'SUM'),
    ('step_size', 'HS'),
    ('line_direction', 'textbook'),
])
def test_condat_vu_refuses_every_axis_it_does_not_carry(axis, value, a_tree):
    """Accepting one of these and solving anyway is the failure #96 has met
    three times: a surface that claims an intent no code carries. A caller who
    asked Condat-Vu for ``SUM`` and got a table saying ``SUM`` would have no
    way of knowing the solve never used one."""

    with pytest.raises(DawisValueError) as refusal:
        solve(a_tree, 'condat_vu', **{axis: value})

    assert axis in str(refusal.value)


@pytest.mark.parametrize('minimizer', ['fista', 'projected_gradient'])
@pytest.mark.parametrize('axis, value', [
    ('step_size', 'HS'),
    ('line_direction', 'textbook'),
])
def test_the_proximal_arms_refuse_the_conjugate_gradient_axes(
        minimizer, axis, value, a_tree):
    """No proximal arm builds a conjugate direction, so neither axis has
    anything to select."""

    with pytest.raises(DawisValueError):
        solve(a_tree, minimizer, **{axis: value})


@pytest.mark.parametrize('minimizer', ['fista', 'projected_gradient'])
def test_a_misspelt_operator_is_refused_on_a_proximal_arm(minimizer, a_tree):
    """The branch chain in :func:`~dawis.trees.minimizers._synthesis` falls
    through to ``SUM``, so a misspelt ``'ADJONT'`` would silently solve a
    different cell and file its numbers under the one that was asked for —
    the same silent reinterpretation #100 found on the line direction."""

    with pytest.raises(DawisValueError) as refusal:
        solve(a_tree, minimizer, synthesis_operator = 'ADJONT')

    assert 'ADJONT' in str(refusal.value)


def test_every_axis_a_minimizer_carries_is_accepted(a_tree):
    """The other half of the refusals: a guard that refused everything would
    pass all of them. Each minimizer's own axes have to go through."""

    solve(a_tree, 'bijaoui_rue', synthesis_operator = 'SUM',
          step_size = 'HS', line_direction = 'textbook')
    solve(a_tree, 'fista', synthesis_operator = 'ADJOINT')
    solve(a_tree, 'projected_gradient', synthesis_operator = 'SUM')
    solve(a_tree, 'condat_vu')


# ---------------------------------------------------------------------------
# The measured step
# ---------------------------------------------------------------------------


def test_the_radius_brackets_a_symmetric_map_from_above():
    """The whole reason the step is measured rather than assumed, and the
    reason the estimate adds its geometric tail.

    A power iteration climbs to the radius **from below**, and low is the
    dangerous direction: the step is :data:`DEFAULT_STEP_SAFETY` over the
    estimate, so an estimate below the true radius makes the step longer than
    ``1 / L`` and costs the forward-backward convergence argument. On a
    symmetric map the true radius is computable, so this is checkable rather
    than argued.

    Bracketing is asserted to a relative slack of ``1e-6`` and not exactly.
    The estimate lands on the radius to the last few digits on a map this
    well-behaved, and which side of it those digits fall on is round-off
    rather than the property. What is not slack is the second assertion: the
    step has to sit strictly inside ``1 / L``, and the safety factor leaves it
    a tenth of the bound to do that in — four orders of magnitude more than
    the round-off.
    """

    rng = np.random.default_rng(102)
    root = rng.normal(size = (16, 16))
    matrix = root @ root.T

    def operator(vector):
        return matrix @ vector

    true_radius = float(np.max(np.abs(np.linalg.eigvalsh(matrix))))
    measured = spectral_radius(operator, (16, 16))

    assert measured >= true_radius * (1.0 - 1e-6)
    assert DEFAULT_STEP_SAFETY / measured < 1.0 / true_radius


def test_the_radius_keeps_the_peak_on_a_map_that_never_settles():
    """#86's defect, and the one number in the proximal arms that can be wrong
    in a direction no equality test would notice.

    On ``[[0, -3], [1, 0]]`` the iteration oscillates ``2.2798, 1.3159,
    2.2798, ...`` for ever. Returning the norm it stopped on returns the dip,
    1.316 against a true radius of 1.732 — a step 18.5% above the bound.
    Keeping the largest norm seen returns 2.280, which is above the radius, so
    the step is short rather than long.
    """

    matrix = np.array([[0.0, -3.0], [1.0, 0.0]])
    true_radius = np.sqrt(3.0)

    measured = spectral_radius(lambda vector: matrix @ vector, (2,))

    assert measured > true_radius
    assert measured == pytest.approx(2.2798, abs = 1e-4)


@pytest.mark.parametrize('operator, n_iterations', [
    (lambda vector: np.zeros_like(vector), 100),
    (lambda vector: vector, 0),
])
def test_the_radius_falls_back_to_one_with_nothing_to_measure(operator,
                                                              n_iterations):
    """An operator that annihilates the starting vector, and a non-positive
    cap. A step length is still needed in both cases and no step is better
    than another, so the fallback is a number rather than an exception — a
    raise here would turn a degenerate tube into a crashed run."""

    assert spectral_radius(operator, (8, 8), n_iterations) == 1.0


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('minimizer', MINIMIZERS)
def test_every_minimizer_takes_the_coarse_channel(minimizer, a_tree,
                                                  a_coarse_cube):
    """#99 made the coarse term a channel of the tube, and every minimizer has
    to carry it — Condat-Vu included, which has no operator axis but still
    moves through ``W`` and its transpose.

    Three places branch on ``coarse`` inside this module, and a channel-count
    or ``n_levels`` slip in any of them reads the channel as a wavelet plane:
    right shape, wrong quantity, no exception. That is the class of defect
    #99's own proof caught, so the channel has to move the atom rather than
    merely be accepted.
    """

    tree, wavelet_cube, label_cube = a_tree
    operator = None if minimizer == 'condat_vu' else 'TRANSPOSE'

    plain = solve(a_tree, minimizer, synthesis_operator = operator)
    with_coarse = solve(a_tree, minimizer, synthesis_operator = operator,
                        coarse_cube = a_coarse_cube)

    assert with_coarse[0].shape == plain[0].shape
    assert np.all(np.isfinite(with_coarse[0]))
    assert np.max(np.abs(with_coarse[0] - plain[0])) > 0.0


def test_a_diverged_solve_is_a_zeroed_atom_and_nan_residuals(a_tree):
    """``fista.HAAR.SUM`` with the coarse term on diverges on 51 of #81's 54
    fixtures where ``projected_gradient.HAAR.SUM`` converges on all 54 — the
    split #96 carries as a sighting and that ships on this non-default cell.

    What is pinned here is not the split but the **handling**: ``SUM`` is not
    an adjoint of anything, the map has an eigenvalue no positive step can
    contract, and the iterate grows to ``1e27`` without ever reaching an
    ``inf``. A finiteness guard alone would let that atom through as a real
    one, so the divergence factor is what raises the flag, and a flagged solve
    comes back the way ``restore_object`` reports a NaN image: a zeroed atom
    and NaN residuals, never an exception.
    """

    haar = np.array([1.0, 0.0, 1.0]) / 2.0
    tree, wavelet_cube, label_cube = a_tree

    atom, sum_wr, norm_wr = minimize(
        tree, wavelet_cube, label_cube, filter = haar, minimizer = 'fista',
        synthesis_operator = 'SUM', max_iter = 200)

    assert np.all(atom == 0.0)
    assert np.isnan(sum_wr)
    assert np.isnan(norm_wr)


def test_the_iterate_that_is_returned_is_the_one_that_is_tested(a_tree):
    """#110. The loop's guard reads a residual measured **before** the step,
    so iterate *k* is only scored at iteration *k + 1* — and a loop that ends
    on ``max_iter`` or on the flux rule has no iteration *k + 1*. The last
    iterate was therefore the one nothing scored, and ``Config.cg_max_iter``
    is a user knob this repo has benchmarked at 3.

    On this tree ``fista.HAAR.SUM`` crosses the divergence factor at step 24.
    Before #110 the guard fired at 25 and a cap of 24 handed back an atom
    with ``flag_convergence`` left True; :func:`_finish` now scores the
    iterate it is about to return, where its residual is computed for
    ``norm_wr`` anyway.

    The pair is what makes this a test of the *hole* and not of the factor:
    23 is under the factor and still comes back as an atom, so the repair has
    not simply started flagging everything.
    """

    haar = np.array([1.0, 0.0, 1.0]) / 2.0
    tree, wavelet_cube, label_cube = a_tree

    def solve(max_iter):
        return minimize(tree, wavelet_cube, label_cube, filter = haar,
                        minimizer = 'fista', synthesis_operator = 'SUM',
                        max_iter = max_iter)

    atom, sum_wr, norm_wr = solve(24)
    assert np.all(atom == 0.0)
    assert np.isnan(sum_wr) and np.isnan(norm_wr)

    atom, sum_wr, norm_wr = solve(23)
    assert np.any(atom > 0.0)
    assert np.isfinite(sum_wr) and np.isfinite(norm_wr)


def test_a_returned_iterate_over_the_factor_is_flagged():
    """#110's guard, on its own, without a tree to build it a residual.

    :func:`_finish` is where the returned iterate is scored, so this is the
    unit: a finite image whose residual has grown past
    :data:`DEFAULT_DIVERGENCE_FACTOR` comes back zeroed with NaN residuals,
    the same one under the factor comes back untouched, and a solve that took
    no step at all — ``max_iter`` of zero, so no starting residual was ever
    measured — has no growth to test and is returned as it is.
    """

    wavelet_tube = np.zeros((8, 8, 2))
    support_tube = np.ones((8, 8, 2))
    image = np.zeros((8, 8))
    image[4, 4] = 10.0

    # With a zero tube the residual is minus the analysis of the image, so a
    # point source has a non-zero residual and the ratio is what is tested. A
    # flat image would not: its detail planes are zero.
    _, _, norm_wr = _finish(wavelet_tube, support_tube, image, BSPL, False,
                            False, None)
    assert np.isfinite(norm_wr) and norm_wr > 0.0

    atom, sum_wr, flagged = _finish(wavelet_tube, support_tube, image, BSPL,
                                    False, False,
                                    norm_wr / (DEFAULT_DIVERGENCE_FACTOR * 2))
    assert np.all(atom == 0.0)
    assert np.isnan(sum_wr) and np.isnan(flagged)

    atom, _, kept = _finish(wavelet_tube, support_tube, image, BSPL, False,
                            False, norm_wr)
    assert np.array_equal(atom, image)
    assert kept == norm_wr

    # No starting residual switches off the growth half of the test and
    # nothing else. A `max_iter` of zero takes no step, so there is no growth
    # to measure — but an initial iterate that overflowed is still not an
    # atom, and `restore_object`'s NaN test does not catch an `inf`.
    atom, _, _ = _finish(wavelet_tube, support_tube, image, BSPL, False,
                         False, None)
    assert np.array_equal(atom, image)

    overflowed = image.copy()
    overflowed[0, 0] = np.inf
    atom, sum_wr, norm_wr = _finish(wavelet_tube, support_tube, overflowed,
                                    BSPL, False, False, None)
    assert np.all(atom == 0.0)
    assert np.isnan(sum_wr) and np.isnan(norm_wr)
