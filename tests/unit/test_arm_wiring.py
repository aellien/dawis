"""Tier 0 checks that a run solves the arm its configuration names (#104).

The six axes — minimizer, filter, synthesis operator, line direction, beta
and the coarse term — are :class:`~dawis.config.Config` fields, and this is
where the package is held to honouring them. What is under test is the whole
carry: ``Config`` to the pipeline, the pipeline to
``restore_objects_default``, that to ``restore_patch`` through ``pmap``'s
shared tuple, and that to :func:`~dawis.trees.minimizers.minimize`.

Driven as **whole runs** rather than against fakes, which the rest of this
module's siblings do. The failure this ticket exists to close is a knob
accepted, validated and carried nowhere —
:data:`~dawis.config.UNWIRED_KNOBS` names the ones that were, and #96 met the
same class of defect from four sides. A fake tree would prove that
``restore_objects_default`` passes on what it is given and say nothing about
whether a run ever gives it anything, which is exactly the half that was
missing. A 64x64 frame capped at two iterations costs about a second.

The recorder wraps :func:`~dawis.trees.minimizers.minimize` and delegates, so
the run under it is a real one: what is asserted is what the solver was
actually called with, on every tree of every iteration.

What these cannot show is that the arm is *correct* — that ``dawis/`` with
the arm selected natively reproduces #93's seam on the same field. That is
the equality proof, and it lives in ``.research/104-arm-config/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from tinyrun import TINY_NAME, TINY_PARAMS, run_quiet, tiny_image  # noqa: E402

from dawis.config import Config  # noqa: E402
from dawis.restoration.synthesis import FILTERS, NO_OPERATOR  # noqa: E402

#: The arm #82 chose, as the six values a stock ``Config`` should carry to
#: every solve. Spelled out rather than read off ``Config()``, so that a
#: default moving is a failure here and not a tautology.
SHIPPING_ARM = dict(
    minimizer = 'bijaoui_rue',
    filter_kw = 'BSPL',
    synthesis_operator = 'TRANSPOSE',
    step_size = 'HS',
    line_direction = 'as_written',
    coarse = True,
)

#: An arm that shares no value with :data:`SHIPPING_ARM` on any axis it
#: carries, so a knob read from the wrong place cannot pass by coincidence.
#: Condat-Vu is the one minimizer with no operator axis, which is the case
#: ``None`` exists for.
OTHER_ARMS = [
    dict(minimizer = 'fista', filter_kw = 'HAAR',
         synthesis_operator = 'ADJOINT', step_size = None,
         line_direction = None, coarse = False),
    dict(minimizer = 'condat_vu', filter_kw = 'HAAR',
         synthesis_operator = None, step_size = None,
         line_direction = None, coarse = False),
    dict(minimizer = 'projected_gradient', filter_kw = 'BSPL',
         synthesis_operator = 'SUM', step_size = None,
         line_direction = None, coarse = True),
]


def _run_recording(arm, tmp_path):
    """Run DAWIS on the tiny frame, recording every call to the solver.

    Returns
    -------
    calls : list of dict
        One per tree per iteration: the keywords
        :func:`~dawis.trees.minimizers.minimize` was called with, plus the
        ``n_planes`` of the tube that call built and the ``n_planes_plain``
        the same tree would have had with no coarse cube. The channel count
        is what separates a coarse cube reaching the solver from the coarse
        term reaching the solve.
    atoms : list of dawis.restoration.synthesis.Atom
        Every atom the run produced. ``RunResult`` carries the restored image
        and not the atoms, so they are collected as the pipeline returns them.
    """

    import dawis.pipelines.synthesis_by_analysis as pipeline
    import dawis.restoration.synthesis as synthesis

    calls = []
    atoms = []
    real_minimize = synthesis.minimize
    real_restore = pipeline.restore_objects_default

    def recording(tree, wavelet_cube, label_cube, filter, **keywords):
        tube, _, _ = tree.tubes(wavelet_cube, label_cube,
                                coarse_cube = keywords.get('coarse_cube'))
        plain, _, _ = tree.tubes(wavelet_cube, label_cube)
        calls.append(dict(keywords, filter = filter,
                          level = int(tree.interscale_maximum.level),
                          n_planes = int(tube.shape[2]),
                          n_planes_plain = int(plain.shape[2])))

        return real_minimize(tree, wavelet_cube, label_cube, filter,
                             **keywords)

    def collecting(*arguments, **keywords):
        restored = real_restore(*arguments, **keywords)
        atoms.extend(restored)

        return restored

    # Undone here rather than at teardown: a test that records two runs would
    # otherwise stack the second recorder on the first, and every solve of the
    # second run would be appended to both lists.
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(synthesis, 'minimize', recording)
        patched.setattr(pipeline, 'restore_objects_default', collecting)

        config = Config(**dict(TINY_PARAMS, cg_max_iter = 3, seed = 7, **arm))
        run_quiet(tiny_image(), config, outdir = str(tmp_path),
                  name = TINY_NAME)

    return calls, atoms


def test_a_stock_run_solves_the_arm_82_chose(tmp_path):
    """Nothing overridden: every solve is the shipping arm, natively."""

    calls, _ = _run_recording({}, tmp_path)

    assert calls, 'the tiny run restored no tree, so it proves nothing'

    for call in calls:
        assert call['minimizer'] == SHIPPING_ARM['minimizer']
        assert call['synthesis_operator'] == SHIPPING_ARM['synthesis_operator']
        assert call['step_size'] == SHIPPING_ARM['step_size']
        assert call['line_direction'] == SHIPPING_ARM['line_direction']
        assert np.array_equal(call['filter'],
                              FILTERS[SHIPPING_ARM['filter_kw']])
        assert call['coarse_cube'] is not None


@pytest.mark.parametrize('arm', OTHER_ARMS,
                         ids = [arm['minimizer'] for arm in OTHER_ARMS])
def test_a_configured_arm_reaches_the_solver_unchanged(arm, tmp_path):
    """Every axis moved off its default arrives moved.

    Run per arm rather than per axis: the axes are not independent — three of
    them are refused on a minimizer that does not carry them — so the unit
    that can vary is the arm.
    """

    calls, _ = _run_recording(arm, tmp_path)

    assert calls, 'the tiny run restored no tree, so it proves nothing'

    for call in calls:
        assert call['minimizer'] == arm['minimizer']
        assert call['synthesis_operator'] == arm['synthesis_operator']
        assert call['step_size'] == arm['step_size']
        assert call['line_direction'] == arm['line_direction']
        assert np.array_equal(call['filter'], FILTERS[arm['filter_kw']])
        assert (call['coarse_cube'] is not None) == arm['coarse']


def test_the_coarse_knob_decides_the_channel_count(tmp_path):
    """``coarse`` is a tube axis, so it has to be visible in the tube.

    A coarse cube reaching the solver is not yet the coarse term reaching the
    solve. The term is appended to the tube as exactly one more channel (#89),
    so the count against the same tree's plain tube is what says it was built
    rather than merely offered.
    """

    plain, _ = _run_recording(dict(coarse = False), tmp_path)
    coarse, _ = _run_recording(dict(coarse = True), tmp_path)

    assert plain and coarse

    for call in plain:
        assert call['n_planes'] == call['n_planes_plain']

    for call in coarse:
        assert call['n_planes'] == call['n_planes_plain'] + 1


@pytest.mark.parametrize('arm', [{}] + OTHER_ARMS,
                         ids = ['stock'] + [arm['minimizer']
                                            for arm in OTHER_ARMS])
def test_the_atom_records_the_arm_the_run_solved(arm, tmp_path):
    """An atom's provenance is the configuration's, not a constant.

    ``Atom.algorithm`` held ``'CG'`` on every atom DAWIS ever wrote, which was
    true while there was one algorithm and became false at #102. It records
    the minimizer now, and the operator records :data:`NO_OPERATOR` where the
    minimizer has no operator axis — a word rather than ``None``, because a
    provenance column read months later has to say which it was.
    """

    _, atoms = _run_recording(arm, tmp_path)
    expected = dict(SHIPPING_ARM, **arm)

    assert atoms, 'the tiny run produced no atom, so it proves nothing'

    for atom in atoms:
        assert atom.algorithm == expected['minimizer']
        assert atom.filter == expected['filter_kw']
        assert atom.synthesis_operator == (
            expected['synthesis_operator']
            if expected['synthesis_operator'] is not None else NO_OPERATOR)


class _FakeTree:
    """A tree that solves to whatever the patched solver returns."""

    def det_err_tube(self, wavelet_cube, label_cube):
        return np.zeros((4, 4, 2))


@pytest.mark.parametrize('image, norm_wr, expected', [
    (np.ones((4, 4)), 1.0, True),
    (np.full((4, 4), np.nan), np.nan, False),
    (np.zeros((4, 4)), np.nan, False),
])
def test_a_solve_that_left_the_numbers_is_flagged(image, norm_wr, expected):
    """The third row is the one #104 added, and the one that was passing.

    The two minimizer families fail differently. The conjugate gradient
    carries its blow-up into the reconstruction, so a NaN sum finds it. A
    proximal arm zeroes the image itself before returning, and a zeroed image
    is finite — so testing the image alone recorded an empty atom as a
    converged one, on the cell ``fista.HAAR.SUM``, which diverges on nearly
    every fixture it is offered. ``norm_wr`` is what both families leave NaN.
    """

    import dawis.restoration.synthesis as synthesis

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(synthesis, 'minimize',
                        lambda *arguments, **keywords: (image, 0.0, norm_wr))
        returned = synthesis.restore_object(_FakeTree(), None, None,
                                            0.1, 0.95, 2, 5)

    assert returned[3] is expected
    if not expected:
        assert not np.any(returned[0])
