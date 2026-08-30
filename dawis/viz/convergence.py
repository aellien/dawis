"""The convergence trace of a run.

Every run writes one line per iteration to ``<outpath>.cpar.txt``: the level
and iteration, the raw, normalised and window-averaged epsilon the stopping
test reads, the restored and residual flux fractions, the atom count and the
wall time. :func:`plot_convergence` draws that trace.

The pipeline used to draw it itself, at the end of every run. It no longer
does, for the same reason it no longer writes animation frames: the numbers
are already on disk, so the picture can be made afterwards by anyone who wants
it, and the solver keeps no dependency on a plotting library.
:func:`rebuild_convergence_plot` is that afterwards.

The run does not have to have finished. The trace is written a line at a time
as the loop goes (#66), so a run that was killed has one, and
:func:`read_convergence_table` drops a trailing half-line rather than refusing
the file.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt

from dawis.convergence import CONVERGENCE_FIELDS

__all__ = ['plot_convergence', 'read_convergence_table', 'rebuild_convergence_plot']


def plot_convergence( cparl: Sequence[Sequence[str]], outpath: str, **kwargs ) -> None:
    '''Draw the convergence trace of a run.

    Parameters
    ----------
    cparl : list of sequence of str
        One row per iteration, as the pipeline collects them and writes them
        to ``<outpath>.cpar.txt``: level, iteration, window-normalised
        epsilon, normalised epsilon, epsilon, restored flux fraction,
        residual flux fraction, atom count, wall time.
    outpath : str
        Path stem of the run. The figure is written to ``<outpath>.conv.pdf``.
    **kwargs
        Passed to :func:`matplotlib.pyplot.plot`.
    '''

    level = []
    it = []
    avnormeps = []
    normeps = []
    eps = []
    flux_rec = []
    flux_res = []
    nobj = []
    time = []

    for cpar in cparl:

        level.append(int(cpar[0]))
        it.append(int(cpar[1]))
        avnormeps.append(float(cpar[2]))
        normeps.append(float(cpar[3]))
        eps.append(float(cpar[4]))
        flux_rec.append(float(cpar[5]))
        flux_res.append(float(cpar[6]))
        nobj.append(int(cpar[7]))
        time.append(cpar[8])

    fig = plt.figure(figsize = (9, 6), dpi = 500)
    plt.plot(it, avnormeps, label = 'window normalized eps', **kwargs)
    plt.plot(it, normeps, label = 'normalized eps', **kwargs)
    plt.plot(it, eps, label = 'eps', **kwargs)
    plt.plot(it, flux_rec, label = 'flux restored fraction', **kwargs)
    plt.plot(it, flux_res, label = 'flux residuals fraction', **kwargs)
    plt.grid(linestyle='-', linewidth=2, alpha=0.2)

    plt.legend()
    plt.yscale('log')
    plt.savefig(''.join(( outpath, '.conv.pdf' )), format = 'pdf' )
    plt.close()


def read_convergence_table(outpath: str) -> list[list[str]]:
    """Read the convergence trace a run wrote.

    Parameters
    ----------
    outpath : str
        Path stem of the run. Reads ``<outpath>.cpar.txt``.

    Returns
    -------
    list of list of str
        One row of fields per iteration, in the order the run wrote them.
        Comment lines are dropped; the fields are left as strings, which is
        the form :func:`plot_convergence` reads.

    Raises
    ------
    ValueError
        If a row anywhere but at the end of the file does not hold
        :data:`~dawis.convergence.CONVERGENCE_FIELDS` fields.

    Notes
    -----
    A *trailing* row of the wrong width is dropped instead, silently. That is
    the shape a run killed mid-write leaves — the pipeline flushes the trace
    per iteration (#66), so the last line can be half a line — and reading a
    run that did not finish is the whole reason this is lenient. A short row
    anywhere earlier cannot have been left by an interruption: it is
    corruption, and it still raises.

    A row is split from the left into
    :data:`~dawis.convergence.CONVERGENCE_FIELDS` fields, so the last one
    keeps whatever is left of the line. The wall time is that last column and
    it is the one field that can hold a space: a ``timedelta`` of a day or
    more stringifies as ``1 day, 0:00:01``. Splitting on every space would
    make an iteration that ran for a day look like corruption.
    """

    numbered = []
    with open(''.join(( outpath, '.cpar.txt' ))) as trace:
        for number, line in enumerate(trace, start = 1):
            line = line.strip()
            if line and not line.startswith('#'):
                numbered.append((number, line.split(None, CONVERGENCE_FIELDS - 1)))

    rows = []
    for position, (number, fields) in enumerate(numbered):

        if len(fields) != CONVERGENCE_FIELDS:

            if position == len(numbered) - 1:
                break

            raise ValueError(
                "%s.cpar.txt line %d: %d fields where %d were expected --> the "
                "trace is corrupt, not truncated"
                % (outpath, number, len(fields), CONVERGENCE_FIELDS)
            )

        rows.append(fields)

    return rows


def rebuild_convergence_plot(outpath: str, marker: str = 'o',
                             linewidth: float = 2, **kwargs) -> None:
    """Draw the convergence trace of a run, from its ``.cpar.txt``.

    The run need not have finished: the trace is written a line at a time as
    the loop goes (#66), so an interrupted run has one too.

    Parameters
    ----------
    outpath : str
        Path stem of the run, as handed to the pipeline —
        ``<outdir>/<infile without .fits>``.
    marker, linewidth
        Line style, defaulting to what the pipeline used when it drew this
        plot itself.
    **kwargs
        Passed to :func:`matplotlib.pyplot.plot`.
    """

    plot_convergence(read_convergence_table(outpath), outpath,
                     marker = marker, linewidth = linewidth, **kwargs)
