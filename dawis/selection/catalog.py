"""A selection's catalog, and the two ways to put an error on a flux (#128).

:func:`to_table` gives one row per source model (#17, point 10). No fraction
column: a ratio between two source models was CFIS's, and is one line of user
arithmetic.

Two error methods ship, and only one is a bootstrap (point 11).
:func:`bootstrap` draws the group's N atoms N times **with** replacement and
returns the standard deviation of the flux; it is the default.
:func:`resample_borderline` is what the CFIS papers did
(``make_synthesis_sdssrm_xcs.py:112-152``): samples of variable size, drawn
**without** replacement, with borderline atoms swapped in, reported as a median
and its 5th and 95th percentiles.

Both resample atoms, never pixels. Without a floor a draw's flux is a sum over
the index; with one, the draw's image is synthesised and floored, as CFIS
floored it inside its loop (``:170``).
"""

from __future__ import annotations

from typing import Iterable, Mapping

import warnings

import numpy as np
from astropy.table import Table

from dawis.selection.source_model import SourceModel, require_columns


def _draw_flux(model: SourceModel, rows: np.ndarray, draw: np.ndarray,
               floor: float | None) -> float:
    # ponytail: with a floor every draw re-reads its atoms off the store; keep
    # the group's pixels in memory if a floored error on a big group is slow.
    return SourceModel(model.name, rows[draw], model.outpath).flux(floor)


def bootstrap(model: SourceModel, n: int = 100, floor: float | None = None,
              rng: np.random.Generator | int | None = None) -> float:
    """The bootstrap error on a source model's flux.

    Draws the model's N atoms N times with replacement, ``n`` times over, and
    returns the standard deviation of the ``n`` fluxes (``CONTEXT.md``,
    *Bootstrap error*).

    Parameters
    ----------
    model : SourceModel
    n : int, optional
        Number of resamplings, at least 2.
    floor : float or None, optional
        Passed to :meth:`SourceModel.flux` for every draw.
    rng : numpy.random.Generator, int or None, optional
        The generator, or a seed for one.

    Returns
    -------
    float
        Zero for a model of fewer than two atoms.
    """

    if n < 2:
        raise ValueError('a standard deviation needs n >= 2 resamplings, not %d' % n)
    rng = np.random.default_rng(rng)
    size = len(model.rows)
    if size < 2:
        return 0.
    fluxes = [_draw_flux(model, model.rows, rng.integers(0, size, size), floor)
              for _ in range(n)]
    return float(np.std(fluxes, ddof = 1))


def resample_borderline(model: SourceModel, borderline: np.ndarray, n: int = 50,
                        spread: float = 0.1, floor: float | None = None,
                        rng: np.random.Generator | int | None = None
                        ) -> tuple[float, float, float]:
    """The CFIS scheme: a flux median with its 5th and 95th percentiles.

    Each of ``n`` samples takes a size ``s`` uniform in
    ``[N - spread N, N + spread N)``, bounds cast to int as CFIS cast them
    (``:112``), and a count ``r`` uniform in
    ``[0, min(spread s, len(borderline), N))`` — CFIS had no ``N`` there, and
    ``N - r`` went negative without it. Below N it is ``s`` of the
    model's atoms; from N up it is ``N - r`` of them plus ``s - N + r``
    borderline atoms, or every borderline atom if there are too few. All draws
    are without replacement (``CONTEXT.md``, *Borderline resampling*).

    Parameters
    ----------
    model : SourceModel
    borderline : numpy.ndarray
        Index rows of the same run that may be swapped in — CFIS took the
        atoms one level below its level cut, inside the mask of interest. An
        atom also in ``model`` warns, as it can then be drawn twice.
    n : int, optional
        Number of samples.
    spread : float, optional
        Relative spread of the sample size, in ``[0, 1]``.
    floor : float or None, optional
        Passed to :meth:`SourceModel.flux` for every sample.
    rng : numpy.random.Generator, int or None, optional
        The generator, or a seed for one.

    Returns
    -------
    median, p5, p95 : float
    """

    if not 0. <= spread <= 1.:
        raise ValueError('spread must lie in [0, 1], not %r' % spread)
    shared = np.intersect1d(model.rows['offset'], borderline['offset']).size
    if shared:
        warnings.warn('%d borderline atom%s also in the source model, and can be drawn twice'
                      % (shared, ' is' if shared == 1 else 's are'), stacklevel = 2)
    rng = np.random.default_rng(rng)
    size, spare = len(model.rows), len(borderline)
    rows = np.concatenate([model.rows, borderline])
    fluxes = []
    for s in rng.uniform(int(size - size * spread), int(size + size * spread), n).astype(int):
        r = int(rng.uniform(0, min(int(s * spread), spare, size)))
        if s < size:
            draw = rng.choice(size, s, replace = False)
        else:
            draw = np.concatenate([
                rng.choice(size, size - r, replace = False),
                size + rng.choice(spare, min(s - size + r, spare), replace = False)])
        fluxes.append(_draw_flux(model, rows, draw, floor))
    median, p5, p95 = np.percentile(fluxes, [50., 5., 95.])
    return float(median), float(p5), float(p95)


def to_table(models: Mapping[str, SourceModel] | Iterable[SourceModel], n: int = 100,
             floor: float | None = None,
             rng: np.random.Generator | int | None = None) -> Table:
    """The catalog: one row per source model.

    ``flux`` and ``flux_err`` are :meth:`SourceModel.flux` and
    :func:`bootstrap`, both at ``floor``, which the table records in its
    ``meta``. The floor moves those two only: ``det_err`` adds the model's whole
    ``det_err`` image in quadrature over its pixels, and the centroid is the
    flux-weighted mean of the atoms' centroids, read off the index. That is the
    image's own centroid when every atom's flux is positive; an atom that is not
    has no centroid and is left out of it. ``bbox``
    holds every atom's box. Empty models give zeros, NaN centroids and ``-1``.

    Parameters
    ----------
    models : mapping of str to SourceModel, or iterable of SourceModel
        As :func:`~dawis.selection.source_models` returns them.
    n : int, optional
        Resamplings for ``flux_err``.
    floor : float or None, optional
        The flux floor.
    rng : numpy.random.Generator, int or None, optional
        The generator, or a seed for one.

    Returns
    -------
    astropy.table.Table
    """

    rng = np.random.default_rng(rng)
    if isinstance(models, Mapping):
        models = models.values()
    columns = {name: [] for name in ('name', 'n_atoms', 'flux', 'flux_err', 'det_err',
                                     'centroid_x', 'centroid_y', 'bbox', 'level_min',
                                     'level_max')}
    for model in models:
        rows = model.rows
        require_columns(rows, ('flux', 'centroid_x', 'centroid_y'))
        placed = np.isfinite(rows['centroid_x']) & np.isfinite(rows['centroid_y'])
        weight = rows['flux'][placed]
        if placed.any():
            centroid = [float(weight @ rows[c][placed] / weight.sum())
                        for c in ('centroid_x', 'centroid_y')]
        else:
            centroid = [np.nan, np.nan]
        if len(rows):
            box = rows['bbox']
            bbox = [*box[:, :2].min(axis = 0), *box[:, 2:].max(axis = 0)]
            levels = [rows['level'].min(), rows['level'].max()]
        else:
            bbox, levels = [-1] * 4, [-1, -1]
        det_err = float(np.sqrt((model.det_err_image() ** 2).sum())) if len(rows) else 0.
        for name, value in zip(columns, (model.name, len(rows), model.flux(floor),
                                         bootstrap(model, n, floor, rng), det_err,
                                         *centroid, bbox, *levels)):
            columns[name].append(value)
    columns['bbox'] = np.array(columns['bbox'], dtype = int).reshape(-1, 4)
    return Table(columns, meta = {'floor': floor, 'n': n, 'error': 'bootstrap'})
