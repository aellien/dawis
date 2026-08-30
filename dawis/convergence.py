"""The stopping test of the greedy loop, and the trace it leaves behind.

DAWIS restores structure iteratively, so something has to decide when an
iteration has stopped being worth running. That decision is one number per
iteration and one comparison against a threshold, and this module holds both.

The number is a *normalised epsilon*. Each iteration takes atoms out of the
residual, so the residual gets quieter; ``eps`` is the fractional change in the
standard deviation of the residual from one iteration to the next. Two things
are then divided out of it, because neither says anything about convergence:
the wavelet scale being worked at, and how many atoms the iteration happened to
restore. What is left is averaged over a short window of iterations, so that a
single quiet iteration cannot stop a level on its own, and that average is what
is compared against ``ceps``.

The window is per level: each level starts its own, because the residual a
level starts on is what the previous level left, and the two levels are not
comparable. :meth:`ConvergenceMonitor.begin_level` is that reset. The standard
deviation itself carries across levels, so the first iteration of a level is
measured against the last iteration of the one before.

Two flux fractions ride along. They are recorded and plotted, never read by
the stopping test.

The trace is the second half. Every :meth:`ConvergenceMonitor.update` appends
one row to :attr:`ConvergenceMonitor.rows`, which the pipeline writes to
``<outpath>.cpar.txt`` and :func:`dawis.viz.rebuild_convergence_plot` draws.
The rows are built here rather than in the pipeline so the column order and the
header cannot drift apart from each other. The pipeline writes each row as the
iteration that made it ends, so an interrupted run leaves a trace of what it
did get through (#66); the monitor is still where every row comes from, which
is what keeps a resumed run from writing one twice.

This module sits at the top level, beside :mod:`dawis.cubes` and
:mod:`dawis.parallel`, for the reason those do: it is vocabulary the pipeline
and the viz block share, and no block owns it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

__all__ = ['CONVERGENCE_COLUMNS', 'CONVERGENCE_FIELDS', 'CONVERGENCE_HEADER',
           'ConvergenceMonitor', 'ConvergenceStep']

#: Comment line the pipeline writes at the top of ``<outpath>.cpar.txt``.
#:
#: It names seven fields where nine are written — the two flux fractions are
#: run together and the atom count is missing. Kept verbatim because Phase A
#: does not change what a run puts on disk; the authoritative column list is
#: the order :meth:`ConvergenceMonitor.update` builds a row in.
CONVERGENCE_HEADER = '# level, it, window_normeps, normeps, eps, n_obj, time_it'

#: The columns of ``<outpath>.cpar.txt``, in the order
#: :meth:`ConvergenceMonitor.update` builds a row in.
#:
#: :data:`CONVERGENCE_HEADER` names only seven of the nine — the two flux
#: fractions are run together and the atom count is missing — and is kept
#: verbatim because Phase A does not change what a run puts on disk. So this
#: is the authoritative list, and it sits beside the method that builds the
#: row so the two cannot drift apart.
CONVERGENCE_COLUMNS = ('level', 'iteration', 'window_normeps', 'normeps',
                       'eps', 'flux_rec', 'flux_res', 'n_atoms', 'time_it')

#: Fields in one row of ``<outpath>.cpar.txt``. Derived, not written down: a
#: column added to :data:`CONVERGENCE_COLUMNS` moves this with it.
#:
#: A reader needs it to tell a row truncated by a kill mid-write from a row
#: that is corrupt; see :func:`dawis.viz.read_convergence_table`.
CONVERGENCE_FIELDS = len(CONVERGENCE_COLUMNS)

#: How many iterations the window average covers.
WINDOW_SIZE = 5


def _flux_fraction(atom: np.ndarray, image: np.ndarray) -> float:
    """How close one iteration's flux came to the total flux of an image.

    One at equality, falling away in both directions. Recorded in the trace
    and plotted; the stopping test never reads it.

    ``sqrt(x**2)`` is the absolute value, written the long way because that is
    what the published pipeline computes and the two differ in the last bits
    for large pixels. Phase A does not move a number.
    """

    return 1 - ( np.abs(np.sum(atom) - np.sum(np.sqrt(image**2)))) / np.sum(np.sqrt(image**2))


@dataclass(frozen=True)
class ConvergenceStep:
    """What one iteration measured. Returned by :meth:`ConvergenceMonitor.update`.

    Attributes
    ----------
    level : int
        Wavelet scale the iteration worked at.
    iteration : int
        Iteration number, counted across the whole run rather than per level.
    eps : float
        Fractional change in the standard deviation of the residual since the
        previous iteration.
    normeps : float
        ``eps`` with the level and the atom count divided out.
    avnormeps : float
        ``normeps`` averaged over the window — the number the stopping test
        reads.
    flux_rec : float
        Flux of the iteration's atoms as a fraction of the restored image.
        Recorded and plotted, never read by the stopping test.
    flux_res : float
        The same fraction against the residual.
    n_atoms : int
        Atoms the iteration restored.
    """

    level: int
    iteration: int
    eps: float
    normeps: float
    avnormeps: float
    flux_rec: float
    flux_res: float
    n_atoms: int


@dataclass
class ConvergenceMonitor:
    """The stopping test, and the trace of every iteration that ran.

    Parameters
    ----------
    scale_lvl_eps : float, optional
        Factor applied to ``normeps``, raising or lowering the effective
        threshold without touching ``ceps``. Default 1.
    window_size : int, optional
        Iterations the window average covers. Default :data:`WINDOW_SIZE`.

    Attributes
    ----------
    avnormeps : float
        The window average, which the loop compares against ``ceps``. Starts
        at 1 — above any usable threshold — so a level always runs at least
        one iteration.
    rows : list of list of str
        One row per iteration, in the order they ran, as the pipeline writes
        them to ``<outpath>.cpar.txt``.

    Examples
    --------
    >>> monitor = ConvergenceMonitor()
    >>> monitor.begin_level()
    >>> while monitor.avnormeps > 1e-4:  # doctest: +SKIP
    ...     atom = run.add(atoms)
    ...     monitor.update(level, it, atom, run.residuals, run.restored,
    ...                    len(atoms), elapsed)
    """

    scale_lvl_eps: float = 1
    window_size: int = WINDOW_SIZE
    avnormeps: float = 1.
    rows: list[list[str]] = field(default_factory = list)
    _std: float = 1
    _window: list[float] = field(default_factory = list)

    def begin_level(self) -> None:
        """Start a fresh window for a new wavelet scale.

        Clears the window and puts :attr:`avnormeps` back above any threshold,
        so the level runs at least one iteration. The standard deviation of the
        residual is deliberately *not* reset: it is a property of the image the
        level inherits, not of the level.
        """

        self._window = []
        self.avnormeps = 1.

    def update(
        self,
        level: int,
        iteration: int,
        atom: np.ndarray,
        residuals: np.ndarray,
        restored: np.ndarray,
        n_atoms: int,
        elapsed: timedelta,
    ) -> ConvergenceStep:
        """Measure one iteration, update the window, and append its row.

        Parameters
        ----------
        level : int
            Wavelet scale the iteration worked at.
        iteration : int
            Iteration number.
        atom : numpy.ndarray
            Sum of the atoms this iteration restored, as
            :meth:`~dawis.restoration.accumulation.Reconstruction.add` returns
            it.
        residuals : numpy.ndarray
            The residual after this iteration's atoms were subtracted.
        restored : numpy.ndarray
            The sum of every atom restored so far.
        n_atoms : int
            Atoms this iteration restored. Must not be zero; an iteration that
            restored nothing breaks out of the loop before reaching here.
        elapsed : datetime.timedelta
            Wall time the iteration took. Recorded in the trace and nowhere
            else — the one field of the trace that is not a function of the
            input.

        Returns
        -------
        ConvergenceStep
            What was measured, for the caller to log.
        """

        flux_res = _flux_fraction(atom, residuals)
        flux_rec = _flux_fraction(atom, restored)

        old_std = self._std
        self._std = np.std(residuals)
        eps = np.sqrt( ( old_std - self._std )**2 ) / np.sqrt( old_std**2 )
        normeps = self.scale_lvl_eps * eps * level / n_atoms

        self._window.append(normeps)
        if len(self._window) > self.window_size:
            self._window.pop(0)
        self.avnormeps = np.mean(self._window)

        # `elapsed` last, and last for a reason a reader depends on: it is the
        # one field that can hold a space. A `timedelta` of a day or more
        # stringifies as `1 day, 0:00:01`, so a row is split from the left into
        # `CONVERGENCE_FIELDS` fields and the wall time is whatever is left.
        row = [ str(level), str(iteration), str(self.avnormeps),
                str(normeps), str(eps), str(flux_rec),
                str(flux_res), str(n_atoms), str(elapsed) ]
        assert len(row) == CONVERGENCE_FIELDS, CONVERGENCE_COLUMNS
        self.rows.append(row)

        return ConvergenceStep(level = level, iteration = iteration, eps = eps,
                               normeps = normeps, avnormeps = self.avnormeps,
                               flux_rec = flux_rec, flux_res = flux_res,
                               n_atoms = n_atoms)
