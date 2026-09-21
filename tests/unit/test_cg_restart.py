"""#109 — the conjugate direction restarts when its denominator vanishes.

The published solve substituted 1.0 for ``||old_fr||**2`` when it was zero
and read that substitution from the ``FR`` branch alone. The other three
betas divided unguarded: ``PRP`` by ``||fr||**2``, ``HS`` by
``(fr - old_fr) . v`` and ``CD`` by ``v . old_fr``. On a tube whose wavelet
coefficients vanish on the support, ``fr`` and ``old_fr`` are both exactly
zero, all three of those are ``0/0``, and the ``nan`` propagates into ``v``
and from there into every later iterate — ``restore_object`` returns a zeroed
atom and ``flag_convergence = False`` with nothing saying why.

The cheapest tube that reaches the failure is one that is exactly zero, which
is what this file builds. It is a degenerate tube rather than one the gallery
produced — `.research/109-beta-denominators/` is where the question of how
often a *real* tube gets there is measured — but it is the shape of tube the
guard exists for, and it fails on all three unguarded betas without it.
"""

import logging

import numpy as np
import pytest

from dawis.config import CG_LINE_DIRECTIONS, CG_STEP_SIZES

#: The B3-spline kernel, as :func:`dawis.restoration.synthesis.restore_object`
#: writes it.
BSPL = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0

#: Enough steps to get past the first update, which computes no beta at all.
STEPS = 6


@pytest.fixture(scope = 'module')
def a_vanishing_tree():
    """A tree whose wavelet coefficients are exactly zero on its support.

    The same geometry as #100's fixture — so the two files describe one tree
    at two amplitudes — with the planes replaced by zeros. That is what makes
    ``fr`` and ``old_fr`` exactly zero rather than merely small, and an exact
    zero is what the denominators need to be for the division to be ``0/0``
    rather than a large number.
    """

    from dawis.cubes import LabelCube, WaveletCube
    from dawis.detection import region
    from dawis.trees import InterscaleTree

    shape = (32, 32)
    n_levels = 3
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

    wavelet_cube = WaveletCube(np.zeros(shape + (n_levels,)), 'BSPL')
    label_cube = LabelCube(labels, lab_counts = [1, 1, 1])
    tree = InterscaleTree(
        interscale_maximum = regions[-1],
        region_list = regions,
        wavelet_cube = wavelet_cube,
        label_cube = label_cube,
        det_err_array = np.zeros(shape + (n_levels,)),
        clip = 16,
    )

    return tree, wavelet_cube, label_cube


def solve(a_vanishing_tree, **axes):
    """Run the arm on the vanishing tree, with the axes under test set."""

    tree, wavelet_cube, label_cube = a_vanishing_tree
    axes.setdefault('synthesis_operator', 'TRANSPOSE')
    return tree.CG_minimization(wavelet_cube, label_cube, filter = BSPL,
                                max_iter = STEPS, **axes)


@pytest.mark.parametrize('step_size', CG_STEP_SIZES)
@pytest.mark.parametrize('line_direction', CG_LINE_DIRECTIONS)
def test_a_vanishing_tube_returns_an_atom_on_every_beta(a_vanishing_tree,
                                                        step_size,
                                                        line_direction):
    """Not only on ``FR``, which is the one the published guard covered — and
    the one a run still solves until #104 wires the axis. ``HS`` is what #82
    chose and what #104 wires, so the unguarded denominator becomes the
    production path on that ticket."""

    atom, sum_wr, norm_wr = solve(a_vanishing_tree, step_size = step_size,
                                  line_direction = line_direction)

    assert np.all(np.isfinite(atom))
    assert np.isfinite(sum_wr) and np.isfinite(norm_wr)


@pytest.mark.parametrize('step_size', CG_STEP_SIZES)
def test_the_restart_is_what_kept_it_finite(a_vanishing_tree, step_size,
                                            caplog):
    """Without this the test above passes on a solve that broke out of the
    loop before computing a beta at all, and would go on passing with the
    guard deleted."""

    with caplog.at_level(logging.DEBUG, logger = 'dawis.trees.construction'):
        solve(a_vanishing_tree, step_size = step_size)

    assert any('conjugate direction restarted' in record.message
               for record in caplog.records)
