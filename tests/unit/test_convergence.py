"""Tier 0 — the stopping test of the greedy loop (#49).

``dawis.convergence`` decides when a wavelet scale has stopped paying. The
decision used to be eight inlined lines in the pipeline, so it had never been
checked on its own; these pin the contract the loop leans on.

Three things matter and none of them are the arithmetic. The window is per
scale and the residual's standard deviation is not — a level starts fresh but
is still measured against what the previous one left. A level always runs at
least one iteration, whatever threshold it is given. And the trace rows carry
the columns, in the order, that ``<outpath>.cpar.txt`` is written in and
``dawis.viz`` reads back.
"""

from __future__ import annotations

import numpy as np
import pytest

from dawis.convergence import CONVERGENCE_HEADER, ConvergenceMonitor

#: A quiet residual and a loud one, so that a change between them is large.
QUIET = np.full((8, 8), 0.1)
LOUD = np.arange(64, dtype=float).reshape(8, 8)


def measure(monitor, residuals, level=2, iteration=1, n_atoms=1):
    """Run one iteration through the monitor with plausible images."""
    atom = np.ones(residuals.shape)
    return monitor.update(
        level=level,
        iteration=iteration,
        atom=atom,
        residuals=residuals,
        restored=np.full(residuals.shape, 2.0),
        n_atoms=n_atoms,
        elapsed="0:00:01",
    )


def test_a_level_always_runs_at_least_one_iteration():
    """``begin_level`` puts the average above any threshold a caller can set.

    The loop is written ``while monitor.avnormeps > ceps``, so a monitor that
    started a level at anything else would let a scale be skipped entirely
    depending on what the previous scale happened to end on.
    """
    monitor = ConvergenceMonitor()
    measure(monitor, QUIET, n_atoms=8)
    assert monitor.avnormeps < 1.0

    monitor.begin_level()

    assert monitor.avnormeps == 1.0


def test_the_window_is_per_level_but_the_residual_is_not():
    """A new level clears the window and keeps the standard deviation.

    The residual a level starts on is what the previous level left, so the
    first iteration of a level is measured against the last iteration of the
    one before. Resetting that too would report a spurious epsilon of zero.
    """
    monitor = ConvergenceMonitor()
    measure(monitor, LOUD)
    carried = monitor._std

    monitor.begin_level()
    assert monitor._window == []
    assert monitor._std == carried

    step = measure(monitor, LOUD, level=3)
    assert step.eps == 0.0, "the residual did not change, so epsilon is zero"


def test_the_window_averages_the_last_five_iterations_only():
    """A single quiet iteration cannot stop a level on its own."""
    monitor = ConvergenceMonitor(window_size=5)
    monitor.begin_level()

    for iteration in range(1, 8):
        residuals = QUIET if iteration % 2 else LOUD
        measure(monitor, residuals, iteration=iteration)

    assert len(monitor._window) == 5
    assert monitor.avnormeps == pytest.approx(np.mean(monitor._window))


def test_epsilon_is_normalised_by_level_and_atom_count():
    """Neither the scale nor how many atoms turned up says anything about
    convergence, so both are divided out."""
    one = ConvergenceMonitor()
    many = ConvergenceMonitor()

    thin = measure(one, LOUD, level=4, n_atoms=1)
    fat = measure(many, LOUD, level=4, n_atoms=8)

    assert thin.eps == fat.eps
    assert fat.normeps == pytest.approx(thin.normeps / 8)


def test_scale_lvl_eps_scales_the_normalised_epsilon():
    plain = measure(ConvergenceMonitor(scale_lvl_eps=1), LOUD)
    scaled = measure(ConvergenceMonitor(scale_lvl_eps=3), LOUD)

    assert scaled.normeps == pytest.approx(3 * plain.normeps)


def test_a_row_carries_the_columns_the_trace_file_is_written_in():
    """The trace on disk is what ``dawis.viz`` and the golden harness read.

    Nine fields, in this order; the header line names only seven of them,
    which is why the row order is pinned here rather than parsed from it.
    """
    monitor = ConvergenceMonitor()
    step = measure(monitor, LOUD, level=5, iteration=7, n_atoms=3)

    row, = monitor.rows

    assert len(row) == 9
    assert all(isinstance(field, str) for field in row)
    assert row[0] == "5"
    assert row[1] == "7"
    assert row[2] == str(step.avnormeps)
    assert row[3] == str(step.normeps)
    assert row[4] == str(step.eps)
    assert row[5] == str(step.flux_rec)
    assert row[6] == str(step.flux_res)
    assert row[7] == "3"
    assert row[8] == "0:00:01"


def test_the_header_is_a_comment_line():
    """Readers skip it by the leading ``#``, so it must keep one."""
    assert CONVERGENCE_HEADER.startswith("#")
