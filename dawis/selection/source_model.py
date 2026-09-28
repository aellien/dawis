"""Source models: the groups a selection made, and the images they add up to (#127).

A selection partitions a run's index rows into named groups. A
:class:`SourceModel` is one of those groups with its name and the run it came
from, so it is complete on its own and reads pixels only when an image is
asked for (#17, point 7). The atoms no rule claimed are no source model: they
synthesise to an image through :func:`~dawis.io.read_synthesised_image` and
get no measurements (point 9).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from dawis.io.store import read_synthesised_image
from dawis.selection.cascade import UNCLASSIFIED
from dawis.selection.criteria import MissingColumn


def require_columns(rows: np.ndarray, columns: tuple[str, ...]) -> None:
    """Refuse, by name, index rows from a store written without ``columns``."""

    missing = [c for c in columns if c not in (rows.dtype.names or ())]
    if missing:
        raise MissingColumn('this reads %s, which this atom store does not carry: it was '
                            'written before the index gained the columns measured off the '
                            'pixels, so re-run to measure it' % ', '.join(missing))


@dataclass(eq = False)
class SourceModel:
    """A named group of whole atoms from one run.

    Parameters
    ----------
    name : str
        What the selection called the group.
    rows : numpy.ndarray
        Its index rows, from :func:`dawis.read_atom_index`.
    outpath : str
        Path stem of the run; the store is ``<outpath>.atoms.hdf5``.
    """

    name: str
    rows: np.ndarray
    outpath: str

    def image(self) -> np.ndarray:
        """The sum of the group's atoms, on the run's frame."""
        return read_synthesised_image(self.outpath, self.rows)

    def det_err_image(self) -> np.ndarray:
        """The sum of the group's atoms' detection error images."""
        return read_synthesised_image(self.outpath, self.rows, det_err = True)

    def flux(self, floor: float | None = None) -> float:
        """The sum of the group's image.

        Read off the index, with no pixel read, unless a ``floor`` is given:
        then only the image's pixels at or above it count, as CFIS counted them
        (``make_synthesis_sdssrm_xcs.py:170``). The floor is an argument and
        never state, so a flux cannot depend on something set elsewhere (#17,
        point 12).
        """

        if floor is None:
            require_columns(self.rows, ('flux',))
            return float(self.rows['flux'].sum())
        image = self.image()
        return float(image[image >= floor].sum())


def source_models(groups: Mapping[str, np.ndarray], outpath: str,
                  unclassified: str = UNCLASSIFIED) -> dict[str, SourceModel]:
    """One :class:`SourceModel` per group, leaving the unclassified group out.

    A :class:`~dawis.selection.Cascade` or :func:`~dawis.selection.split`
    partitions by construction, but groups built by hand need not, so an atom
    found more than once across ``groups`` — the unclassified group included —
    warns (#17, point 3). A warning and not an error: the user may mean it, and
    what is worth catching is the atom counted twice without anyone knowing.

    Parameters
    ----------
    groups : mapping of str to numpy.ndarray
        Group name to index rows, as a cascade returns them.
    outpath : str
        Path stem of the run the rows come from.
    unclassified : str, optional
        The name of the group that is not a source model.

    Returns
    -------
    dict of str to SourceModel
        In the order of ``groups``.
    """

    nonempty = [r for r in groups.values() if len(r)]
    if nonempty:
        # An atom's pixel offset names it within its store. Only an atom with
        # no pixel could share one, and it adds nothing to an image.
        _, counts = np.unique(np.concatenate(nonempty)['offset'], return_counts = True)
        shared = int((counts > 1).sum())
        if shared:
            warnings.warn('%d atom%s found more than once across the groups, and will be '
                          'counted once per appearance'
                          % (shared, ' is' if shared == 1 else 's are'),
                          stacklevel = 2)
    return {name: SourceModel(name, r, outpath)
            for name, r in groups.items() if name != unclassified}
